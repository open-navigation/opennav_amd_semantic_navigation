#!/usr/bin/env python3
# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HSA_XNACK", "1")

# Cap CPU thread pools BEFORE any C-extension import. SAM3's inference path
# is GPU-bound; py-spy shows ~1 dispatcher thread is enough. The PyTorch/
# OpenCV defaults (== all logical cores) over-subscribe the scheduler and
# compete with co-resident ROS nodes (nav2_controller_server, costmap_2d).
# Override via env (OMP_NUM_THREADS=N etc.) if you need more.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import threading

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from opennav_sam3_msgs.srv import ChangePrompt
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from vision_msgs.msg import LabelInfo, VisionClass


# Module-level cache so the same class_id always draws the same color
# across frames and across prompt-reconfigurations.
_COLOR_CACHE: dict[int, np.ndarray] = {}


def _infer_imgsz(onnx_dir: str) -> int:
    """Extract resolution from onnx_dir path name (e.g. 'onnx_files_504' -> 504)."""
    match = re.search(r'(504|1008)', os.path.basename(onnx_dir))
    return int(match.group(1)) if match else 504


class Sam3InferenceNode(Node):
    """ROS 2 node that runs SAM3 text-prompted instance segmentation on an image topic.

    Uses SAM3Live (streaming video model with MIGraphX acceleration) for inference.
    """

    def __init__(self):
        super().__init__('sam3_inference')

        self.declare_parameter('checkpoint', 'models/sam3')
        self.declare_parameter('onnx_dir', 'onnx_files_504')
        self.declare_parameter('prompts', ['object'])
        self.declare_parameter('class_ids', [1])
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('score_threshold', 0.5)
        self.declare_parameter('max_objects_per_prompt', 5)
        self.declare_parameter('redetect_interval_ms', 0.0)
        # Bound GPU memory growth by periodically dropping the session's
        # accumulated per-frame raw pixel buffer + tracker per-obj history.
        # Default 30s; set <= 0 to disable.
        self.declare_parameter('reset_tracking_every_seconds', 30.0)
        self.declare_parameter('queue_depth', 5)
        self.declare_parameter('start_enabled', True)
        # CPU thread-pool cap for torch + cv2. Default 1: SAM3 inference is
        # GPU-bound and uses only ~1 dispatcher thread (py-spy verified).
        # Capping prevents the 32-default pools from competing with co-resident
        # ROS nodes (nav2_controller_server, costmap_2d) for CPU scheduling
        # and CCX-local cache. Raise this if profiling shows CPU starvation.
        # NOTE: OMP/MKL env caps at module top must be raised separately
        # via env var (OMP_NUM_THREADS=N) — those are import-time locked.
        self.declare_parameter('cpu_threads', 1)

        checkpoint = self.get_parameter('checkpoint').value
        onnx_dir = self.get_parameter('onnx_dir').value
        prompts = list(self.get_parameter('prompts').value)
        class_ids = list(self.get_parameter('class_ids').value)
        self._prompts, self._class_ids = self._validate_prompt_config(prompts, class_ids)
        device_param = self.get_parameter('device').value
        device = device_param or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._score_threshold = self.get_parameter('score_threshold').value
        max_objects = self.get_parameter('max_objects_per_prompt').value
        self._redetect_interval_ms = self.get_parameter('redetect_interval_ms').value
        self._reset_tracking_every_s = self.get_parameter(
            'reset_tracking_every_seconds').value
        queue_depth = self.get_parameter('queue_depth').value
        self._enabled = self.get_parameter('start_enabled').value

        # Apply CPU thread caps from parameter (see declare_parameter above).
        # Runtime setters are required: torch ignores TORCH_NUM_THREADS env
        # and cv2 only reads thread count at first parallel call.
        n_cpu_threads = int(self.get_parameter('cpu_threads').value)
        torch.set_num_threads(n_cpu_threads)
        torch.set_num_interop_threads(n_cpu_threads)
        cv2.setNumThreads(n_cpu_threads)
        self.get_logger().info(
            f'CPU thread pools capped: torch={n_cpu_threads} cv2={n_cpu_threads} '
            f"(env OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')})"
        )

        imgsz = _infer_imgsz(onnx_dir)

        self.get_logger().info(
            f'Loading SAM3 from {checkpoint} on {device} '
            f'(imgsz={imgsz}, onnx_dir={onnx_dir})'
        )

        _boot = int(os.environ.get("SAM3_BOOTSTRAP_FRAMES", "5"))
        _boot_min = float(os.environ.get("SAM3_BOOTSTRAP_MIN_SCORE", "0.3"))

        if os.environ.get("SAM3_USE_HYBRID", "0") == "1":
            from opennav_sam3_inference.tracker.hybrid_inference import SAM3HybridLive
            _kfe = int(os.environ.get("SAM3_KEYFRAME_EVERY", "10"))
            self.get_logger().info(
                f'SAM3_USE_HYBRID=1: instantiating SAM3HybridLive '
                f'(keyframe_every={_kfe}, bootstrap_frames={_boot})'
            )
            self._live = SAM3HybridLive(
                checkpoint=checkpoint,
                prompts=self._prompts,
                onnx_dir=onnx_dir,
                imgsz=imgsz,
                dtype=torch.float16,
                device=device,
                mig=True,
                keyframe_every=_kfe,
                max_objects_per_prompt=max_objects,
                bootstrap_frames=_boot,
                bootstrap_min_score=_boot_min,
            )
        else:
            from opennav_sam3_inference.tracker.live_inference import SAM3Live
            self.get_logger().info(
                f'instantiating SAM3Live (bootstrap_frames={_boot})'
            )
            self._live = SAM3Live(
                checkpoint=checkpoint,
                prompts=self._prompts,
                onnx_dir=onnx_dir,
                imgsz=imgsz,
                dtype=torch.float16,
                device=device,
                mig=True,
                max_objects_per_prompt=max_objects,
                redetect_every=1,
                bootstrap_frames=_boot,
                bootstrap_min_score=_boot_min,
            )

        self._prompt_to_class_id = dict(zip(self._prompts, self._class_ids))

        # ROS time-based redetection tracking
        self._last_detect_time = self.get_clock().now()
        self._last_reset_time = self.get_clock().now()

        self._bridge = CvBridge()
        self._infer_lock = threading.Lock()

        self._pub = self.create_publisher(Image, '~/segmentation_mask', queue_depth)
        self._label_mask_pub = self.create_publisher(Image, '~/label_mask', queue_depth)
        label_info_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._label_info_pub = self.create_publisher(LabelInfo, '~/label_info', label_info_qos)
        sub_qos = QoSProfile(depth=queue_depth, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._image_cb_group = ReentrantCallbackGroup()
        self._sub = self.create_subscription(
            Image, '~/image', self._on_image, sub_qos,
            callback_group=self._image_cb_group)
        self._change_prompt_srv = self.create_service(
            ChangePrompt, '~/change_prompt', self._on_change_prompt
        )
        self._enable_srv = self.create_service(
            SetBool, '~/enable', self._on_enable
        )

        self.add_on_set_parameters_callback(self._on_param_update)
        self._publish_label_info()

        self.get_logger().info(
            f'SAM3 Inference Node ready, starting enabled: {self._enabled}'
        )

    @staticmethod
    def _validate_prompt_config(prompts, class_ids):
        if not prompts or not class_ids:
            raise ValueError('prompts and class_ids must both be non-empty')
        if len(prompts) != len(class_ids):
            raise ValueError(
                f'prompts ({len(prompts)}) and class_ids ({len(class_ids)}) '
                'must have the same length'
            )
        normalized_prompts = []
        normalized_ids = []
        for p, cid in zip(prompts, class_ids):
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f'prompt must be a non-empty string, got {p!r}')
            cid_int = int(cid)
            if cid_int <= 0 or cid_int > 255:
                raise ValueError(
                    f'class_id must be in [1, 255] (0 is reserved for "no detection"), '
                    f'got {cid_int} for prompt {p!r}'
                )
            normalized_prompts.append(p.strip())
            normalized_ids.append(cid_int)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError(f'duplicate class_ids in {normalized_ids}')
        if len(set(normalized_prompts)) != len(normalized_prompts):
            raise ValueError(f'duplicate prompts in {normalized_prompts}')
        return normalized_prompts, normalized_ids

    def _publish_label_info(self):
        msg = LabelInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.class_map = [
            VisionClass(class_id=cid, class_name=name)
            for name, cid in zip(self._prompts, self._class_ids)
        ]
        msg.threshold = float(self._score_threshold)
        self._label_info_pub.publish(msg)

    def _on_param_update(self, params):
        threshold_changed = False
        for p in params:
            if p.name == 'score_threshold':
                if not isinstance(p.value, (int, float)) or not 0.0 <= float(p.value) <= 1.0:
                    reason = f'{p.name} must be in [0.0, 1.0], got {p.value}'
                    self.get_logger().warn(f'Rejected parameter update: {reason}')
                    return SetParametersResult(successful=False, reason=reason)
                self._score_threshold = float(p.value)
                self.get_logger().info(f'Setting score_threshold to {self._score_threshold}')
                threshold_changed = True
        if threshold_changed:
            self._publish_label_info()
        return SetParametersResult(successful=True)

    def _on_change_prompt(self, request, response):
        try:
            prompts, class_ids = self._validate_prompt_config(
                list(request.prompts), list(request.class_ids)
            )
        except ValueError as exc:
            self.get_logger().warn(f'Rejected prompt update: {exc}')
            response.success = False
            response.message = str(exc)
            return response
        old_n = len(self._prompts)
        new_n = len(prompts)
        if new_n != old_n:
            self.get_logger().warn(
                f'Prompt count changed to {new_n}; there will be added latency '
                'on the next image. Wait for a fresh ~/label_mask before '
                'assuming the new prompts are live.'
            )
        self._live.reset_prompts(prompts)
        self._prompts = prompts
        self._class_ids = class_ids
        self._prompt_to_class_id = dict(zip(prompts, class_ids))
        self._publish_label_info()
        mapping = ', '.join(f'{n}={c}' for n, c in zip(self._prompts, self._class_ids))
        response.success = True
        response.message = f'Prompts updated: {mapping}'
        self.get_logger().info(response.message)
        return response

    def _on_enable(self, request, response):
        self._enabled = request.data
        state = 'enabled' if self._enabled else 'disabled'
        self.get_logger().info(f'Inference {state}')
        response.success = True
        response.message = f'Inference {state}'
        return response

    def _should_detect(self) -> bool:
        """Time-based detection policy using ROS clock."""
        if self._redetect_interval_ms <= 0.0:
            return True
        now = self.get_clock().now()
        elapsed_ms = (now - self._last_detect_time).nanoseconds / 1e6
        if elapsed_ms >= self._redetect_interval_ms:
            self._last_detect_time = now
            return True
        return False

    def _on_image(self, msg: Image):
        self.get_logger().info(  # TODO back to debug
            f'Received image with timestamp {msg.header.stamp.sec}.'
            f'{msg.header.stamp.nanosec} and frame_id "{msg.header.frame_id}"'
        )
        if not self._enabled:
            return

        # Non-blocking try; if inference is in flight, drop this frame
        if not self._infer_lock.acquire(blocking=False):
            return

        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            prompts = self._prompts
            class_ids = self._class_ids
            prompt_to_class_id = self._prompt_to_class_id

            full_detection = self._should_detect()
            result = self._live.infer(bgr, full_detection=full_detection)

            H, W = bgr.shape[:2]
            label_map = np.zeros((H, W), dtype=np.uint8)
            score_map = np.zeros((H, W), dtype=np.float32)
            instances = []

            for prompt_text, obj_ids in result['prompt_to_obj_ids'].items():
                class_id = prompt_to_class_id.get(prompt_text, 0)
                for oid in obj_ids:
                    score = result['scores'][oid]
                    if score < self._score_threshold:
                        continue
                    mask = result['masks'][oid]
                    update = mask & (score > score_map)
                    label_map[update] = class_id
                    score_map[update] = score
                    instances.append((mask, class_id))

            self.get_logger().info(  # TODO back to debug
                f'Found {len(instances)} instances across {len(prompts)} prompts'
            )

            label_msg = self._bridge.cv2_to_imgmsg(label_map, encoding='mono8')
            label_msg.header = msg.header
            self._label_mask_pub.publish(label_msg)

            if self._pub.get_subscription_count() > 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                overlay = _render_overlay(rgb, instances)
                out_msg = self._bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
                out_msg.header = msg.header
                self._pub.publish(out_msg)

            # Periodic reset_tracking() to bound the session's per-frame
            # raw pixel buffer + tracker per-obj history (which otherwise
            # grows ~1.5 MB/frame at imgsz=504, exhausting GPU after
            # ~20 min at 30 fps). Runs inside the image callback AFTER
            # publish so it serializes with inference (no race) and the
            # current frame's mask is already delivered to downstream
            # consumers (label_mask publishes class_id per pixel, so
            # obj_id renumbering is invisible). Next inference is slightly
            # faster than steady-state (empty session, no propagation
            # overhead) before ramping back over a few frames.
            if self._reset_tracking_every_s > 0.0:
                now = self.get_clock().now()
                elapsed_s = (now - self._last_reset_time).nanoseconds / 1e9
                if elapsed_s >= self._reset_tracking_every_s:
                    self._live.reset_tracking()
                    self._last_reset_time = now
                    self.get_logger().debug(
                        f'reset_tracking() to bound GPU memory '
                        f'(elapsed {elapsed_s:.0f}s since last reset)'
                    )
        except Exception as exc:
            self.get_logger().error(f'Segmentation failed: {exc}')
        finally:
            self._infer_lock.release()

        self.get_logger().info(  # TODO back to debug
            f'Finished processing image with timestamp {msg.header.stamp.sec}.'
            f'{msg.header.stamp.nanosec}'
        )


def _color_for_class_id(class_id: int) -> np.ndarray:
    cached = _COLOR_CACHE.get(class_id)
    if cached is not None:
        return cached
    color = np.random.default_rng(class_id).integers(64, 256, size=3, dtype=np.uint8)
    _COLOR_CACHE[class_id] = color
    return color


def _render_overlay(rgb: np.ndarray, instances, alpha: float = 0.5) -> np.ndarray:
    out = rgb.copy()
    for mask, class_id in instances:
        color = _color_for_class_id(class_id)
        out[mask] = ((1.0 - alpha) * out[mask] + alpha * color).astype(np.uint8)
    return out


def main(args=None):
    rclpy.init(args=args)
    node = Sam3InferenceNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HSA_XNACK", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/cache/inductor")

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from opennav_sam3_msgs.srv import ChangePrompt
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from transformers import Sam3Model, Sam3Processor
from vision_msgs.msg import LabelInfo, VisionClass


SAM3_INPUT_SIZE = 1008

# Module-level cache so the same class_id always draws the same color
# across frames and across prompt-reconfigurations.
_COLOR_CACHE: dict[int, np.ndarray] = {}


"""ROS 2 node that runs SAM3 text-prompted instance segmentation on an image topic."""
class Sam3InferenceNode(Node):
    def __init__(self):
        super().__init__('sam3_inference')

        self.declare_parameter('model_name', 'facebook/sam3')
        self.declare_parameter('prompts', ['object'])
        self.declare_parameter('class_ids', [1])
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('score_threshold', 0.5)
        self.declare_parameter('mask_threshold', 0.5)
        self.declare_parameter('queue_depth', 5)
        self.declare_parameter('start_enabled', True)
        self.declare_parameter('compile_cache_path', '/cache/sam3_compiled.pt')

        self._model_name = self.get_parameter('model_name').value
        prompts = list(self.get_parameter('prompts').value)
        class_ids = list(self.get_parameter('class_ids').value)
        self._prompts, self._class_ids = self._validate_prompt_config(prompts, class_ids)
        self._score_threshold = self.get_parameter('score_threshold').value
        self._mask_threshold = self.get_parameter('mask_threshold').value
        queue_depth = self.get_parameter('queue_depth').value
        self._enabled = self.get_parameter('start_enabled').value
        self._compile_cache_path = self.get_parameter('compile_cache_path').value

        device_param = self.get_parameter('device').value
        self._device = device_param or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._dtype = torch.bfloat16 if self._device == 'cuda' else torch.float32

        torch._dynamo.config.capture_scalar_outputs = True
        if self._device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self._bridge = CvBridge()

        self.get_logger().info(f'Loading {self._model_name} on {self._device} (dtype={self._dtype})')
        self._model = Sam3Model.from_pretrained(
            self._model_name,
            torch_dtype=self._dtype,
            attn_implementation='sdpa',
        ).to(self._device)
        self._model.eval()
        self._model = self._model.to(memory_format=torch.channels_last)
        self._processor = Sam3Processor.from_pretrained(self._model_name)

        if os.path.exists(self._compile_cache_path):
            self.get_logger().info(f'Loading compiled model state from {self._compile_cache_path}')
            self._model.load_state_dict(torch.load(self._compile_cache_path, map_location=self._device))
            self._model = torch.compile(self._model, mode='max-autotune-no-cudagraphs', fullgraph=True) # TODO -no-cudagraphs perf change?
            self.get_logger().info('Skipping warmup compile (inductor cache should populate from TORCHINDUCTOR_CACHE_DIR)')
        else:
            self.get_logger().info('Compiling model (first run, this will take a few minutes)...')
            self._model = torch.compile(self._model, mode='max-autotune-no-cudagraphs', fullgraph=True) # TODO -no-cudagraphs perf change?
            dummy_image = np.zeros((SAM3_INPUT_SIZE, SAM3_INPUT_SIZE, 3), dtype=np.uint8)
            warmup_probe = self._processor(images=dummy_image, text=self._prompts[0], return_tensors='pt')
            warmup_inputs = {
                k: (v.to(self._device) if torch.is_tensor(v) else v)
                for k, v in warmup_probe.items()
            }
            with torch.inference_mode():
                self._model(**warmup_inputs)
            if self._device == 'cuda':
                torch.cuda.synchronize()
            os.makedirs(os.path.dirname(self._compile_cache_path), exist_ok=True)
            torch.save(self._model._orig_mod.state_dict(), self._compile_cache_path)
            self.get_logger().info(f'Saved compiled model state to {self._compile_cache_path}')

        self._device_buffers = None
        self._non_tensor = None

        self.get_logger().info(f'Loaded {self._model_name} on {self._device}')

        self._pub = self.create_publisher(Image, '~/segmentation_mask', queue_depth)
        self._label_mask_pub = self.create_publisher(Image, '~/label_mask', queue_depth)
        # Latched: late-joining subscribers receive the most recent mapping.
        label_info_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._label_info_pub = self.create_publisher(LabelInfo, '~/label_info', label_info_qos)
        sub_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._sub = self.create_subscription(Image, '~/image', self._on_image, sub_qos)
        self._change_prompt_srv = self.create_service(
            ChangePrompt, '~/change_prompt', self._on_change_prompt
        )
        self._enable_srv = self.create_service(
            SetBool, '~/enable', self._on_enable
        )

        # Allow `score_threshold` and `mask_threshold` to be changed live via
        # the built-in parameter service (e.g. `ros2 param set ...`).
        self.add_on_set_parameters_callback(self._on_param_update)

        self._publish_label_info()

        self.get_logger().info(f'SAM3 model {self._model_name} ready to process, starting enabled: {self._enabled}')


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
            # 0 is reserved for "no detection" in the label mask; VisionClass.class_id is uint16.
            if cid_int <= 0 or cid_int > 65535:
                raise ValueError(
                    f'class_id must be in [1, 65535] (0 is reserved for "no detection"), '
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
            if p.name in ('score_threshold', 'mask_threshold'):
                if not isinstance(p.value, (int, float)) or not 0.0 <= float(p.value) <= 1.0:
                    reason = f'{p.name} must be in [0.0, 1.0], got {p.value}'
                    self.get_logger().warn(f'Rejected parameter update: {reason}')
                    return SetParametersResult(successful=False, reason=reason)
                if p.name == 'score_threshold':
                    self._score_threshold = float(p.value)
                    self.get_logger().info(f'Setting score_threshold to {self._score_threshold}')
                    threshold_changed = True
                else:
                    self._mask_threshold = float(p.value)
                    self.get_logger().info(f'Setting mask_threshold to {self._mask_threshold}')
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
        self._prompts = prompts
        self._class_ids = class_ids
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

    def _on_image(self, msg: Image):
        self.get_logger().debug(
            f'Received image with timestamp {msg.header.stamp.sec}.{msg.header.stamp.nanosec} and frame_id "{msg.header.frame_id}"'
        )
        if not self._enabled:
            return
        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            orig_h, orig_w = rgb.shape[:2]

            scale = SAM3_INPUT_SIZE / max(orig_h, orig_w)
            if scale < 1.0:
                infer_image = cv2.resize(
                    rgb,
                    (int(round(orig_w * scale)), int(round(orig_h * scale))),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                infer_image = rgb

            # Snapshot the current prompt mapping so a mid-frame service call
            # doesn't tear the label mask across classes.
            prompts = self._prompts
            class_ids = self._class_ids

            instances = []  # list of (mask_bool: np.ndarray[H,W], score: float, class_id: int)
            for prompt_text, class_id in zip(prompts, class_ids):
                # We can actually prompt with any/all:
                # - text: Text to loop for in the image
                # - points: Lists of points and labels
                # - boxes: Bounding boxes as localization priors
                # - Masks: Feed previous masks to refine
                cpu_inputs = self._processor(
                    images=infer_image,
                    text=prompt_text,
                    return_tensors='pt',
                )

                if self._device_buffers is None or any(
                    k not in self._device_buffers or self._device_buffers[k].shape != v.shape
                    for k, v in cpu_inputs.items() if torch.is_tensor(v)
                ):
                    self._device_buffers = {
                        k: torch.empty_like(
                            v,
                            device=self._device,
                            dtype=(self._dtype if v.is_floating_point() else v.dtype),
                        )
                        for k, v in cpu_inputs.items() if torch.is_tensor(v)
                    }
                    self._non_tensor = {
                        k: v for k, v in cpu_inputs.items() if not torch.is_tensor(v)
                    }
                    self.get_logger().debug('Rebuilt device input buffers')

                for k, buf in self._device_buffers.items():
                    src = cpu_inputs[k]
                    if src.is_floating_point():
                        src = src.to(self._dtype)
                    buf.copy_(src, non_blocking=True)
                inputs = {**self._non_tensor, **self._device_buffers}

                with torch.inference_mode():
                    outputs = self._model(**inputs)

                results = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self._score_threshold,
                    mask_threshold=self._mask_threshold,
                    target_sizes=[[orig_h, orig_w]],
                )[0]

                masks = results['masks']
                scores = results['scores']
                if len(masks) == 0:
                    continue
                masks_np = masks.cpu().numpy().astype(bool)
                scores_np = scores.cpu().numpy()
                for mask, score in zip(masks_np, scores_np):
                    instances.append((mask, float(score), class_id))

            self.get_logger().debug(
                f'Found {len(instances)} instances across {len(prompts)} prompts'
            )

            # Max-score-wins label mask (mono16).
            label_map = np.zeros((orig_h, orig_w), dtype=np.uint16)
            score_map = np.zeros((orig_h, orig_w), dtype=np.float32)
            for mask, score, class_id in instances:
                update = mask & (score > score_map)
                label_map[update] = class_id
                score_map[update] = score

            label_msg = self._bridge.cv2_to_imgmsg(label_map, encoding='mono16')
            label_msg.header = msg.header
            self._label_mask_pub.publish(label_msg)

            if self._pub.get_subscription_count() > 0:
                overlay = _render_overlay(rgb, [(m, cid) for m, _, cid in instances])
                out_msg = self._bridge.cv2_to_imgmsg(overlay, encoding='rgb8')
                out_msg.header = msg.header
                self._pub.publish(out_msg)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(f'Segmentation failed: {exc}')

        self.get_logger().debug(
            f'Finished processing image with timestamp {msg.header.stamp.sec}.{msg.header.stamp.nanosec}'
        )


def _color_for_class_id(class_id: int) -> np.ndarray:
    cached = _COLOR_CACHE.get(class_id)
    if cached is not None:
        return cached
    # Floor of 64 keeps colors from being too dark to see on dark backgrounds.
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

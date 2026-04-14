#!/usr/bin/env python3
# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from opennav_sam3_msgs.srv import ChangePrompt
from PIL import Image as PILImage
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from transformers import Sam3Model, Sam3Processor

import matplotlib

"""ROS 2 node that runs SAM3 text-prompted instance segmentation on an image topic."""
class Sam3InferenceNode(Node):
    def __init__(self):
        super().__init__('sam3_inference')

        self.declare_parameter('model_name', 'facebook/sam3')
        self.declare_parameter('prompt', 'object')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('score_threshold', 0.5)
        self.declare_parameter('mask_threshold', 0.5)
        self.declare_parameter('queue_depth', 5)
        self.declare_parameter('start_enabled', True)

        self._model_name = self.get_parameter('model_name').value
        self._prompt = self.get_parameter('prompt').value
        self._score_threshold = self.get_parameter('score_threshold').value
        self._mask_threshold = self.get_parameter('mask_threshold').value
        queue_depth = self.get_parameter('queue_depth').value
        self._enabled = self.get_parameter('start_enabled').value

        device_param = self.get_parameter('device').value
        self._device = device_param or ('cuda' if torch.cuda.is_available() else 'cpu')

        self._bridge = CvBridge()

        self.get_logger().info(f'Loading {self._model_name} on {self._device}')
        self._model = Sam3Model.from_pretrained(self._model_name).to(self._device)
        # self._model.eval()
        self._processor = Sam3Processor.from_pretrained(self._model_name)
        self.get_logger().info(f'Loaded {self._model_name} on {self._device}')

        self._pub = self.create_publisher(Image, '~/segmentation', queue_depth)
        self._sub = self.create_subscription(Image, '~/image', self._on_image, queue_depth)
        self._change_prompt_srv = self.create_service(
            ChangePrompt, '~/change_prompt', self._on_change_prompt
        )
        self._enable_srv = self.create_service(
            SetBool, '~/enable', self._on_enable
        )

        # Allow `score_threshold` and `mask_threshold` to be changed live via
        # the built-in parameter service (e.g. `ros2 param set ...`).
        self.add_on_set_parameters_callback(self._on_param_update)

        self.get_logger().info(f'SAM3 model {self._model_name} ready to process, starting enabled: {self._enabled}')


    def _on_param_update(self, params):
        for p in params:
            if p.name == 'score_threshold':
                self._score_threshold = p.value
                self.get_logger().info(f'Setting score_threshold to {self._score_threshold}')
            elif p.name == 'mask_threshold':
                self._mask_threshold = p.value
                self.get_logger().info(f'Setting mask_threshold to {self._mask_threshold}')
        return SetParametersResult(successful=True)

    def _on_change_prompt(self, request, response):
        new_prompt = request.prompt.strip()
        if not new_prompt:
            self.get_logger().warn('Rejected empty prompt')
            response.success = False
            return response
        self._prompt = new_prompt
        self.get_logger().info(f'Prompt updated to "{self._prompt}"')
        response.success = True
        return response

    def _on_enable(self, request, response):
        self._enabled = request.data
        state = 'enabled' if self._enabled else 'disabled'
        self.get_logger().info(f'Inference {state}')
        response.success = True
        response.message = f'Inference {state}'
        return response

    def _on_image(self, msg: Image):
        print (f'Received image with timestamp {msg.header.stamp.sec}.{msg.header.stamp.nanosec} and frame_id "{msg.header.frame_id}"')
        if not self._enabled:
            return
        try:
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            pil_image = PILImage.fromarray(cv_image)

            # We can actually prompt with any/all:
            # - text: Text to loop for in the image
            # - points: Lists of points and labels
            # - boxes: Bounding boxes as localization priors
            # - Masks: Feed previous masks to refine
            inputs = self._processor(
                images=pil_image,
                text=self._prompt,
                return_tensors='pt',
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self._score_threshold,
                mask_threshold=self._mask_threshold,
                target_sizes=inputs.get('original_sizes').tolist(),
            )[0]
            # Results contain:
            # - masks: Binary masks resized to original image size
            # - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
            # - scores: Confidence scores
            # Only masks are consumed here; boxes/scores are available for downstream
            # users who want to publish detections (e.g. vision_msgs/Detection2DArray).

            masks = results['masks']
            self.get_logger().debug(f'Found {len(masks)} objects for prompt "{self._prompt}"')

            overlay = _overlay_masks(pil_image, masks)
            out_msg = self._bridge.cv2_to_imgmsg(np.array(overlay), encoding='rgb8')
            out_msg.header = msg.header
            self._pub.publish(out_msg)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(f'Segmentation failed: {exc}')
        
        print(f'Finished processing image with timestamp {msg.header.stamp.sec}.{msg.header.stamp.nanosec} at timestamp {self.get_clock().now().to_msg().sec}.{self.get_clock().now().to_msg().nanosec}')


def _overlay_masks(image: PILImage.Image, masks) -> PILImage.Image:
    rgba = image.convert('RGBA')
    if len(masks) == 0:
        return rgba.convert('RGB')
    masks_np = 255 * masks.cpu().numpy().astype(np.uint8)
    n = masks_np.shape[0]
    cmap = matplotlib.colormaps.get_cmap('rainbow').resampled(n)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n)]
    for mask, color in zip(masks_np, colors):
        mask_img = PILImage.fromarray(mask)
        overlay = PILImage.new('RGBA', rgba.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        rgba = PILImage.alpha_composite(rgba, overlay)
    return rgba.convert('RGB')


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

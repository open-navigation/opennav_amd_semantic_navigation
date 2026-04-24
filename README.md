# Open Navigation x AMD Semantic SAM3 Navigation on AMD Ryzen AI Max

This repository hosts a demonstration of using edge semantic segmentation for autonomous navigation with [Nav2](https://docs.nav2.org/), running on the [AMD Ryzen AI Max](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html) using the [AMD Robotics SDK](https://www.amd.com/en/developer/robotics.html) and Meta's [SAM3](https://ai.meta.com/sam3) foundation model for text-prompted instance segmentation. It turns a camera stream into a live, named segmentation of the robot's environment that Nav2 can reason about for **terrain- and situation-aware navigation**, detection of **small or otherwise hard-to-see obstacles**, and handling of **dynamic obstacles** that traditional geometry-only pipelines miss.

TODO ^ other use-cases missing

This demonstrates an alternative to the Nvidia Jetson to perform state-of-the-art semantic segmentation workflows using AMD's Ryzen AI Max "Strix Halo" - which is also capable to perform workloads on the edge like Detection, Segmentation, VLMs, VLAs, LLMs, and more (with powerful AMD Ryzen CPU cores to boot).

**⚠️ Need ROS 2, Nav2, or deployment help? Contact [Open Navigation](https://www.opennav.org/)! ⚠️**

TODO video of the navigation + mask (concise)

SAM3 is a state-of-the-art _text-promptable_ foundation model. It accepts text prompts regarding what to segment from the image ("person", "pallet", "wet floor sign", "curb", "mud", ...) and it returns per-instance masks for each. This opens up a different class of perception than classical fixed-class detectors or segmentation algorithms: the same node can be repurposed at runtime for different environments, obstacle classes, or mission objectives without retraining. Combined with Nav2's costmap plugin ecosystem, this enables navigation decisions that are driven by what the robot is actually looking at, not just generic obstacles found by depth cameras/lidars or what fits a preset number of labels.

## Package Structure

This repository is layed out as the following:

*  `opennav_sam3_inference`: the ROS 2 node that hosts SAM3, subscribes to a camera topic, and publishes per-instance masks as an overlay image.
*   `opennav_sam3_msgs`: message and service definitions (e.g. `ChangePrompt.srv`) used to reconfigure the segmentation node at runtime.
* `semantic_segmentation_layer`: A symlink to the semantic segmentation layer used to project the 2D segmetnation mask into the costmap frame and set relative environment costs for terrain-aware navigation and detecting small or otherwise hard-to-see obstacles

This also contains a handy `Dockerfile` containing the full ROCm, AMD PyTorch, `transformers`, & ROS 2 Jazzy stack to make it easy to use. This is setup to run on ROCm 7.2.1, but the `BASE_IMAGE` can be replaced based on your system. ROCm 7.0.0 is also common.

## Demonstrations

TODO

Explain the task and why this is necessary to have more awareness about the terrain and difficult obstacles.

Videos:

  * SAM3 in the environment mask

  * Robot navigating terrains and avoiding small or dfficult obstacles using this

Images

  * Infographic of the pipeline


Note: This demonstration does not showcase using the masks for dynamic obstacle segmentation for tracking or enhanced localization performance by removing dynamic obstacle measurements. Nor does it show you how you can use the masks in the behavior tree to segment context about the environment (crowdedness, confined vs open space, etc) to change behavoral characteristics. I just thought that they'd be good extensions that are easy to do building off of this 😉


## SAM3 on Ryzen AI Max

The segmentation node wraps Hugging Face's `Sam3Model` / `Sam3Processor` which is downloaded and optimized on first-boot and stored for later fast start up. Expect about ~10 minutes to load the first time, afterwards under 30 seconds. 

TODO performance metrics in bold, if low, mention a compariable one with the Jetson instead. Add qualfications that this is server class algorithm that its impressive we can run on the edge at all.

The SAM3 node publishes three outputs:

| Topic | Type | Description |
| --- | --- | --- |
| `~/label_mask` | `sensor_msgs/Image` (`mono16`) | Pixel value = `class_id` of the top-scoring class at that pixel; `0` means no detection. |
| `~/label_info` | `vision_msgs/LabelInfo` | Maps `class_id` to `class_name` for use of `label_mask` downstream. |
| `~/segmentation_mask` | `sensor_msgs/Image` (`rgb8`) | Per-class-tinted overlay of the source frame for validation. Colors are deterministic per `class_id` |

Two services let you steer the node without restarting:

| Service | Type | Description |
| --- | --- | --- |
| `~/change_prompt` | `opennav_sam3_msgs/srv/ChangePrompt` | Replace the prompts & class_ids live (e.g. `[("person", 1), ("pallet", 2), ("puddle", 3)]`). |
| `~/enable` | `std_srvs/srv/SetBool` | Pause / resume inference. |


All parameters are declared in [`opennav_sam3_inference/config/sam3_inference.yaml`](opennav_sam3_inference/config/sam3_inference.yaml) and can be overridden via the `params_file` launch argument.

The following parameters are also provided:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `string` | `facebook/sam3` | Hugging Face repo ID for the SAM3 checkpoint. Must be one your `HF_TOKEN` has access to. |
| `prompts` | `string[]` | `["object"]` | Text prompts to segment. Must be the same length as `class_ids`, with no duplicate entries. Swappable at runtime via the `~/change_prompt` service. |
| `class_ids` | `uint16[]` | `[1]` | Parallel array of class IDs for `prompts`. Each ID is written into the `~/label_mask` output at pixels belonging to that class and echoed in `~/label_info`. Must be in `[1, 65535]` (0 is reserved for "no detection") and unique. |
| `device` | `string` | `cuda` | Torch device string. `cuda` is correct on ROCm (PyTorch reuses the CUDA device namespace for HIP). Falls back to `cpu` if no accelerator is available. |
| `score_threshold` | `double` | `0.5` | Minimum confidence score for a detected instance to be emitted. Live-editable, validated to `[0.0, 1.0]`. Echoed in `~/label_info.threshold`. |
| `mask_threshold` | `double` | `0.5` | Per-pixel threshold applied when binarizing the predicted mask. Live-editable, validated to `[0.0, 1.0]`. |
| `queue_depth` | `int` | `5` | History depth for the `~/segmentation` and `~/label_mask` publishers. The `~/image` subscription uses a fixed `depth=2`, `BEST_EFFORT` profile so stale frames are dropped — real-time perception wants latest-frame-wins. |
| `start_enabled` | `bool` | `true` | If `false`, the node loads and compiles the model but returns early from the image callback until `~/enable` is called with `data: true`. |
| `compile_cache_path` | `string` | `/cache/sam3_compiled.pt` | Where to persist the compiled-state checkpoint between runs. Mount a host directory at `/cache` to keep it across container removals. |

## Build and Run

TODO probably want to fix this up so its not in a workspace at all.

It is recommended to use the Dockerfile to deploy the SAM3 node as it uses an AMD provided base image from the [Ryzers](https://github.com/AMDResearch/Ryzers) project which works with a respective ROCm version to setup compatible versions of key dependencies like PyTorch. This makes it easy to use without fighting with dependencies. This base image can also be used for VLMs, YOLO, LLMs, OpenCV and more.

This assumes your `HF_TOKEN` is set as an environmental variable (from your `~/.bashrc` for instance)

```bash
cd ~/ros2_ws/src/opennav_sam3_inference
docker build -t opennav_sam3_inference .

docker run --rm -it \
    --network host \
    --ipc host \
    --shm-size 16G \
    --device=/dev/kfd --device=/dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/sam3:/cache \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    --cap-add=SYS_PTRACE \
    -e HF_TOKEN=$HF_TOKEN \
    --name sam3 \
    opennav_sam3_inference
```

It is **key** that you mount the hugging face and sam3 caches so that the downloaded weights and optimized compiled models persist between docker images! Else, each run will download and optimize the model. 

The entrypoint will automatically launch the node using the provided launch file and configuration. Override this with `bash` at the end of the command to open a terminal in the docker image. There you could manually run via:

```bash
    ros2 launch opennav_sam3_inference sam3_inference.launch.py \
        image_topic:=/my_camera/image_raw \
        segmentation_topic:=/sam3/segmentation
```

## Related Projects

*   [opennav_amd_demonstrations](https://github.com/open-navigation/opennav_amd_demonstrations) — companion project demonstrating indoor 2D, urban 3D, and outdoor GPS-based navigation on Ryzen AI with the Honeybee reference platform.
*   [Nav2](https://github.com/ros-navigation/navigation2) — the ROS 2 navigation stack this work plugs into.

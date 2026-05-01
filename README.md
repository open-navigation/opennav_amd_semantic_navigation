# Semantic SAM3 Navigation on AMD Ryzen AI Max+

This repository hosts a demonstration of using generalized and edge compute segmentation for autonomous navigation with [Nav2](https://docs.nav2.org/), running on the [AMD Ryzen AI Max+](https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-personal-ai-supercomputing-guide.html) and Meta's [SAM3](https://ai.meta.com/sam3) foundation model for text-prompted semantic segmentation. It turns a camera stream into a live, locally-run named segmentation of the robot's environment that Nav2 can reason about for **terrain-aware navigation**, detection of **small or otherwise hard-to-see obstacles**, **changing situational context**, and handling of **dynamic obstacles**, and much more.

This enables applications to:

  * Make trade-offs in planning and control about navigating via certain surfaces over others. For example, prefer sidewalks over grass (while strictly avoiding street), prefer wide open areas over confined aisles, avoid spills or puddles where possible, etc.

  * Detect small or far away objects on the ground difficult to pick up on depth cameras or lidars, in a generalized way without having to know every possible object you may encounter

  * Make decisions about the nature of the situation the robot is currently in to adjust its behavior or algorithms. For example if in confined vs open space, in an area approaching another robot/human/vehicle, or in a situation with lots of commotion.

  * Find common dynamic objects without training on each specific type to remove from the static scene for dynamic tracking and/or localization improvements.

  * The list goes on! If you can imagine it, it can be integrated into a behavior tree, navigation algorithm, or costmap layer. Doubly so for application-specific tasks.

TODO ^ other use-cases missing

This demonstrates state-of-the-art foundation model workflows using AMD's Ryzen AI Max+ "Strix Halo" - which is also capable to perform workloads on the edge like Detection, Segmentation, VLMs, VLAs, LLMs, and more with 32 powerful x86 CPU cores to boot. The newest generations of AI-enabled processors are absolutely amazing for robotics (NPU, GPU, FPGA, 32x x86 cores) workloads without needing to purchase a robotics-specific SOM.

**⚠️ Need ROS 2, Nav2, or deployment help? Contact [Open Navigation](https://www.opennav.org/)! ⚠️**

TODO video of the navigation + mask (concise)

SAM3 is a state-of-the-art _text-promptable_ foundation model. It accepts text prompts regarding what to segment from the image ("person", "pallet", "wet floor sign", "curb", "mud", "ceiling", ...) and it returns masks for each. Gone are the days of fixed detectors or segmentation algorithm classes: the same model can be used at run-time to find various environments, objects, surfaces, and more without retraining (and may be dynamically changed at run-time too!). Combining this with Nav2's costmap, behavior tree, and/or algorithm plugins, SAM3 is extremely powerful and empowers intelligent applications to be developed understanding the world more fully. This enables the robot to make decisions about navigation or behaviors driven not by obstacles but by rich semantic context.

## Package Structure

This repository is layed out as the following:

*  `opennav_sam3_inference`: the ROS 2 node optimized for Strix Halo that hosts SAM3, subscribes to a camera topic, and publishes segmentation masks as an overlay image and labeled masks for use in downstream applications.
*   `opennav_sam3_msgs`: message and service definitions (e.g. `ChangePrompt.srv`) used to reconfigure the segmentation node at runtime.
* `semantic_segmentation_layer`: A symlink to the semantic segmentation layer used to project the 2D segmentation mask into the costmap frame and set relative environment costs for terrain-aware navigation and detecting small or otherwise hard-to-see obstacles

TODO behavior tree nodes to use it? Crowded/confined/etc.
TODO preprocessing for removing dynamic obstacles for localtzation improvements

This also contains a handy `Dockerfile` containing the full ROCm, AMD PyTorch, `transformers`, & ROS 2 Jazzy stack to make it easy to use. This is setup to run on ROCm 7.2.1, but the `BASE_IMAGE` can be replaced based on your system. ROCm 7.0.0 is also common.

## Real-World Tech Demonstrations

TODO

Explain the task and why this is necessary to have more awareness about the terrain and difficult obstacles. Other TODOs?

Videos:

  * SAM3 in the environment mask

  * Robot navigating terrains and avoiding small or dfficult obstacles using this

Images

  * Infographic of the pipeline


Note: This demonstration does not showcase using the masks for dynamic obstacle segmentation for tracking or enhanced localization performance by removing dynamic obstacle measurements. Nor does it show you how you can use the masks in the behavior tree to segment context about the environment (crowdedness, confined vs open space, etc) to change behavoral characteristics. I just thought that they'd be good extensions that are easy to do building off of this 😉

## TODO 

section explaining the code used, arhiticture, how to use for your application


## SAM3 on Ryzen AI Max+

The segmentation node wraps Hugging Face's `Sam3Model` / `Sam3Processor` which is downloaded and optimized on first-boot and stored for later fast start up. Expect about ~10 minutes to load the first time, afterwards under 60 seconds. 

TODO performance metrics in bold, if low, mention a compariable one with the Jetson instead. Add qualfications that this is server class algorithm that its impressive we can run on the edge at all.
TODO mention proportionate to the number of prompts used, so minimize grouping any togetehr that are used together

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

### Build and Run

TODO probably want to fix this up so its not in a workspace at all.

#### Get Access to SAM3

If you have not already, you must create a hugging face account and do the following:

* Accept license at https://huggingface.co/facebook/sam3
* Get token from https://huggingface.co/settings/tokens 
* Export your `HF_TOKEN` to your environment (probably add to `~/.bashrc`)

#### Build and Run Node

It is recommended to use the Dockerfile to deploy the SAM3 node as it uses an AMD provided base image from the [Ryzers](https://github.com/AMDResearch/Ryzers) project which works with a respective ROCm version to setup compatible versions of key dependencies like PyTorch. This makes it easy to use without fighting with dependencies. This base image can also be used for VLMs, YOLO, LLMs, OpenCV and more.

This assumes your `HF_TOKEN` is set as an environmental variable (from your `~/.bashrc` for instance)

```bash
cd ~/ros2_ws/src/opennav_sam3_inference
docker build -t opennav_sam3_inference .

docker run --rm -it \
    --network host \
    --ipc host \
    --privileged \
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

The entrypoint raises `net.core.rmem_max` / `net.core.rmem_default` at startup so DDS can keep up with raw camera images at 10+ Hz. Without larger UDP receive buffers, fragmented image packets are dropped and the subscriber sees a sparse, bursty stream. If you prefer, set these persistently on the host in `/etc/sysctl.d/` instead on your host.

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

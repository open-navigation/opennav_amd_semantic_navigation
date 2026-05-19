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

SAM3 is a state-of-the-art _text-promptable_ foundation model. It accepts text prompts regarding what to segment from the image ("person", "pallet", "wet floor sign", "curb", "mud", "ceiling", ...) and it returns masks for each class and ID of each object within a class. Gone are the days of fixed detectors or segmentation algorithm classes: the same model can be used at run-time to find various environments, objects, surfaces, and more without retraining (and may be dynamically changed at run-time too!). Combining this with Nav2's costmap, behavior tree, and/or algorithm plugins, SAM3 is extremely powerful and empowers intelligent applications to be developed understanding the world more fully. This enables the robot to make decisions about navigation or behaviors driven not by obstacles but by rich semantic context.

## Package Structure

This repository is layed out as the following:

*  `opennav_sam3_inference`: the ROS 2 node optimized for Strix Halo that hosts SAM3, subscribes to a camera topic, and publishes segmentation masks as an overlay image and labeled masks for use in downstream applications.
*   `opennav_sam3_msgs`: message and service definitions (e.g. `ChangePrompt.srv`) used to reconfigure the segmentation node at runtime.
* `semantic_segmentation_layer`: A symlink to the semantic segmentation layer used to project the 2D segmentation mask into the costmap frame and set relative environment costs for terrain-aware navigation and detecting small or otherwise hard-to-see obstacles

TODO behavior tree nodes to use it? Crowded/confined/etc.
TODO preprocessing for removing dynamic obstacles for localtzation improvements

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

The segmentation node wraps SAM3 model that could either be downloaded from Hugging Face or from community through the provided setup script. In addition, some artifacts for optimization are also leveraged for better peformance.

<!-- TODO performance metrics in bold, if low, mention a compariable one with the Jetson instead. Add qualfications that this is server class algorithm that its impressive we can run on the edge at all.
TODO mention proportionate to the number of prompts used, so minimize grouping any togetehr that are used together -->

TODO: Fill the table

| Prompts | New detect every frame | New Detect every 1s, track between |
| ------- | ------------------ | --------------------- |
| 1       |                    |                       |
| 2       |                    |                       |
| 4       |                    |                       |

The SAM3 node publishes three outputs:

| Topic | Type | Description |
| --- | --- | --- |
| `~/label_mask` | `sensor_msgs/Image` (`mono8`) | Pixel value = `class_id` of the top-scoring class at that pixel; `0` means no detection. |
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
| `checkpoint` | `string` | `path/to/sam3/model` | Path containing SAM3 model weights `model.safetensors` file |
| `onnx_dir` | `string` | `path/to/onnx/files` | Build artifacts with optimizations consisting ONNX files |
| `prompts` | `string[]` | `["object"]` | Text prompts to segment. Must be the same length as `class_ids`, with no duplicate entries. Swappable at runtime via the `~/change_prompt` service. |
| `class_ids` | `uint16[]` | `[1]` | Parallel array of class IDs for `prompts`. Each ID is written into the `~/label_mask` output at pixels belonging to that class and echoed in `~/label_info`. Must be in `[1, 255]` (0 is reserved for "no detection") and unique. |
| `device` | `string` | `cuda` | Torch device string. `cuda` is correct on ROCm (PyTorch reuses the CUDA device namespace for HIP). Falls back to `cpu` if no accelerator is available. |
| `score_threshold` | `double` | `0.5` | Minimum confidence score for a detected instance to be emitted. Live-editable, validated to `[0.0, 1.0]`. Echoed in `~/label_info.threshold`. |
| `mask_threshold` | `double` | `0.5` | Per-pixel threshold applied when binarizing the predicted mask. Live-editable, validated to `[0.0, 1.0]`. |
| `queue_depth` | `int` | `5` | History depth for the `~/segmentation` and `~/label_mask` publishers. The `~/image` subscription uses a fixed `depth=2`, `BEST_EFFORT` profile so stale frames are dropped — real-time perception wants latest-frame-wins. |
| `start_enabled` | `bool` | `true` | If `false`, the node loads and compiles the model but returns early from the image callback until `~/enable` is called with `data: true`. |
| `max_objects_per_prompt` | `int` | `5` | Cap on simultaneously tracked objects per prompt. Excess (lowest score) are evicted via session.remove_object so the tracker stops propagating them. |
| `redetect_every` | `int` | `1` | Does the full SAM3 detection after the given number of frames to ensure new objects/people are detected over time. All the intermediate frames does tracker propagation on just the previous detections |

#### Get Access to SAM3

If you have not already, you must create a hugging face account and do the following:

* Accept license at https://huggingface.co/facebook/sam3
* Get token from https://huggingface.co/settings/tokens 

#### Install Dependencies and Setup Model

Setup a recent version of `conda` / `miniforge`. Check out ([miniforge repo](https://github.com/conda-forge/miniforge)) for installation steps. Then, run the `setup.sh` script to set up the Python, ROCm, and other important dependencies, including model weights:

```bash
cd <workspace>/src/opennav_amd_samantic_sam3_navigation/opennav_sam3_inference/
./setup.sh
```

Without any modifiers, the script will install ROCm, migraphx, setup Python dependencies in `opennav-sam3-inference` conda environment, and downloads the model weights to `/mode/sam3/` folder in the same directory used to run the script.

Note: The setup script provides two options to either obtain SAM3 weights from the official repository or from a community mirror, either of them works fine.

Then the build model artifacts for a specific resolution/pipeline which takes a few minutes

```bash
conda activate opennav-sam3-inference

# Text-prompt (~18 min @504px)
python export/build.py --pipeline text --imgsz 504

# Both resolutions (~45 min total)
python export/build.py --pipeline text --imgsz 504 1008
```

You should see the output files in `onnx_files_*` folder.

#### Build and Run Node

Configure the node parameters in `sam3_inference.yaml` to ensure absolute paths to model weights and build artifacts are correctly set. Feel free to adjust other parameters as well following the parameters table above.

> Always do this when this package needs to be built

Build just this package in the conda environment used for the setup script.

```bash
conda activate sam3-tracker # Or the a different environment name used in setup.sh script

colcon build --packages-select opennav_sam3_inference

conda deactivate 
```

Now you can source the workspace as usual and launch the inference node

```bash
ros2 launch opennav_sam3_inference sam3_inference.launch.py image_topic:=/my_camera/image_raw
```

Segmentation and label mask topics are set to `~/segmentation_mask` and `~/label_mask` respectively by default, but can be remapped as needed.

## Related Projects

*   [opennav_amd_demonstrations](https://github.com/open-navigation/opennav_amd_demonstrations) — companion project demonstrating indoor 2D, urban 3D, and outdoor GPS-based navigation on Ryzen AI with the Honeybee reference platform.
*   [Nav2](https://github.com/ros-navigation/navigation2) — the ROS 2 navigation stack this work plugs into.

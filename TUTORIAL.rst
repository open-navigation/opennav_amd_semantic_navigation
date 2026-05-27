.. _sam3_navigation_on_amd_strix_halo:

Navigating with Semantic Segmentation (SAM3, AMD X100 Strix Halo)
*****************************************************************

- `Overview`_
- `Why SAM3 + Why Strix Halo`_
- `Requirements`_
- `Architecture Overview`_
- `Tutorial Steps`_
- `Swapping Prompts at Runtime`_
- `Real-World Demonstrations`_

Overview
========

This tutorial walks through running a live, text-promptable semantic segmentation costmap for `Nav2 <https://docs.nav2.org/>`_ on the edge, using Meta's `SAM3 <https://ai.meta.com/sam3>`_ foundation model on an `AMD X100 Strix Halo <https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-personal-ai-supercomputing-guide.html>`_. By the end you will have:

- The SAM3 inference node running on the X100's GPU and publishing high resolution per-pixel class masks at 5-10 Hz.
- A Nav2 stack configured to consume those masks through the `semantic_segmentation_layer <https://github.com/kiwicampus/semantic_segmentation_layer>`_ costmap plugin to set costs based on the terrain and/or obstacle types.
- The ability to *change segmentation prompts at runtime* - swap from ``["floor", "wall"]`` to ``["sidewalk", "grass, trees, or benches"]`` to ``["pallet", "puddle"]`` without restarting anything.

SAM3 is *text-promptable*: you tell it what to find ("person", "pallet", "wet floor sign", "curb", "mud", "ceiling", ...) and it returns masks for each class.
There is no per-class training step.
The same model handles indoor floors, outdoor terrain, bike lanes, and arbitrary obstacles, so the costmap can be re-aimed at a new environment by editing a list of strings.

TODO VIDEO

Why SAM3 + Why Strix Halo
=========================

Traditional perception for navigation tells you *where* obstacles are (from a depth camera or lidar), but not *what* they are or how the robot should treat them.
It also struggles with thin or low-profile objects (cables, curbs, debris, spills) and with anything beyond the effective range of the depth sensor.
Semantic segmentation fills that gap by labeling every pixel in the color image, even where depth is noisy or missing.

What's new with SAM3 is the *text prompt*.
Older segmentation networks had a fixed class list baked in at training time.
If you wanted to add a new object, you collected data and retrained.
SAM3 instead takes arbitrary string(s) describing what to segment which can also be entire sentences or phrases.
The "class list" is whatever you typed in the launch parameter, and you can change it at runtime via the ``~/change_prompt`` service.
That is a key property that makes a single model useful across warehouses, sidewalks, farms, and construction sites without retraining possible.

Running a server-class foundation model at 5-10 Hz on a robot is the other half of the story.
The AMD X100 Strix Halo (Ryzen AI Max+ 395) pairs an NPU and a Radeon GPU with 32 x86 cores and unified memory in a single package, so SAM3 fits and runs on the edge without an external accelerator.
That same compute is available for everything else a modern robot might want to run locally (i.e. detectors, VLMs, VLAs, LLMs, planners) without needing another GPU-forwar device.
The X100 / Strix Halo has as much CPU power as a high-end AMD laptop and as much GPU as a Jetson, married together to enable this.

Requirements
============

This tutorial assumes you already have:

- AMD X100 Strix Halo running **Ubuntu**. We use the `GMKtec EVO-X2 AI Mini PC <https://www.gmktec.com/products/amd-ryzen%E2%84%A2-ai-max-395-evo-x2-ai-mini-pc>`_
- ROS 2 Jazzy or newer installed
- A robot platform with a depth camera that produces an RGB image and a registered, aligned pointcloud. We use an Orbbec Gemini 355 and Intel RealSense families, but larger FOV and disparities are beneficial.
- ``conda`` / ``miniforge`` for the Python environment SAM3 runs in. Install instructions: `miniforge <https://github.com/conda-forge/miniforge>`_.
- A `Hugging Face <https://huggingface.co>`_ account if you intend to download the SAM3 weights from the official Meta repository.

TODO robot picture

Architecture Overview
=====================

The pipeline takes in synchronized and aligned RGB image and pointcloud from the camera, runs SAM3 interence to produce a label mask of classes.
Then, the ``label_mask`` is resized to a lower resolution with the pointcloud to pass onto the costmap layer for processing.
It is not required to resize the label mask, but it is a common practice to reduce the computational load on the costmap layer.
There is little benefit to using a 504x504 or 1008x1008 mask in the costmap layer with ~2-5cm resolution cells for a local 3-10m horizon.
In the range or below ~400x200 is sufficient.
Finally, planning and control uses the semantically annotated costmap cells and their associated costs to impact navigation behavior to prevent navigating over certain spaces, prefer some surfaces over others, or treat particular objects uniquely.

TODO diagram (already have) 

If your sensor does not produce registered RGB + depth, ``sensor_processing_pipeline.launch.py`` can decimate and align them for you.
The SAM3 node itself only requires a color image topic as input; the costmap layer is where the registered pointcloud comes in.

The SAM3 node subscribes to the color image topic, runs detection and an optional tracker between full detections, and then publishes:

- ``~/label_mask`` (``sensor_msgs/Image``, ``mono8``) pixel value = class ID, 0 means "no detection".
- ``~/label_info`` (``vision_msgs/LabelInfo``) class ID to class name mapping for downstream consumers.
- ``~/segmentation_mask`` (``sensor_msgs/Image``, ``rgb8``) a per-class-tinted overlay of the source frame, for debugging and demos.

The semantic costmap layer takes those three pieces: the mask, the labels, and the registered pointcloud.
It projects each labeled mask pixel into 3D using the pointcloud and accumulating observations per costmap tile to set its class type.
Internally, the ``semantic_segmentation_layer`` maintains a *tile map* to store these observations over a sliding window used to populate the costmap.

Tutorial Steps
==============

0 - Clone the repository
------------------------

.. code-block:: bash

   cd <your_ros2_ws>/src
   git clone --recursive https://github.com/open-navigation/opennav_amd_semantic_navigation.git

If you already cloned without ``--recursive`` (to get the costmap layer submodule):

.. code-block:: bash

   cd opennav_amd_semantic_navigation
   git submodule update --init --recursive

1 - Get the SAM3 weights
------------------------

SAM3's weights (~3.3 GB) are not redistributed with the repo and must be obtained from Hugging Face.
If you haven't already, make an account, then go to https://huggingface.co/facebook/sam3 and accept the license.
It may take a few minutes before your account permissions propagate and you can download.
Create a ``HF_TOKEN`` at https://huggingface.co/settings/tokens and login with ``hf auth login``.
Its generally advised to export this token as an environment variable to use in the future.
Finally:

   .. code-block:: bash

      hf download facebook/sam3 model.safetensors --local-dir model/sam3

2 - Install ROCm and Python dependencies
----------------------------------------

The ``opennav_sam3_inference`` package ships a one-shot setup script that installs the pinned ROCm 7.2 stack, a patched MIGraphX 2.15, an ``opennav-sam3-inference`` conda environment with ROCm-nightly PyTorch + ``onnxruntime-migraphx``, and (optionally) the model weights
That sounds like alot, but basically it sets up the right versions of AMD's software, AI optimization libraries, and PyTorch so that everyone plays nicely and there is no dependency hell.
You can review the setup script easily yourself to see exactly what happens.

.. code-block:: bash

   cd <your_ros2_ws>/src/opennav_amd_semantic_navigation/opennav_sam3_inference
   ./setup.sh


3 - Build the SAM3 inference artifacts
--------------------------------------

The SAM3 model has to be compiled into ONNX/MIGraphX artifacts for the X100's GPU.
This runs once per resolution and takes ~18 minutes for the default 504-pixel input.
You may use the full 1008 resolution, but this takes more inference time with little to no improvement in the quality of masks.

.. code-block:: bash

   conda activate opennav-sam3-inference
   cd <your_ros2_ws>/src/opennav_amd_semantic_navigation/opennav_sam3_inference

   # Single resolution (~18 min @ 504 px)
   python scripts/export/build.py --pipeline text --imgsz 504 # 1008 also optional

You will see an ``onnx_files_504/`` directory in the working directory containing the build artifacts.

Move ``model/sam3/`` and the ``onnx_files_504/`` directory to a stable location on the machine (for example ``/opt/opennav/sam3/`` or ``~/opennav-sam3-models/``).

4 - Configuration
-----------------

Edit ``opennav_sam3_inference/config/sam3_inference.yaml`` so the node knows where the weights and artifacts live:

.. code-block:: yaml

   sam3_inference:
     ros__parameters:
       checkpoint: /absolute/path/to/model/sam3              # contains the model
       onnx_dir:   /absolute/path/to/onnx_files_504          # contains the optimized ONNX files
       prompts:    ["floor", "wall"]
       class_ids:  [1, 2]
       device:     cuda                                      # correct on ROCm - PyTorch reuses CUDA names for HIP
       score_threshold:        0.5
       max_objects_per_prompt: 5
       redetect_interval_ms:   0.0
       queue_depth:            5
       start_enabled:          true

A few notes on the parameters you will tune most often:

- ``prompts`` / ``class_ids`` Each prompt becomes a class with the matching ID written into ``~/label_mask`` at every detected pixel. IDs must be in ``[1, 255]`` and unique (0 is reserved for "no detection").
- ``score_threshold`` / ``mask_threshold`` - confidence thresholds for whole detections and for per-pixel binarization.
- ``redetect_interval_ms`` - run full SAM3 detection every N milliseconds; track-only on the rest. ``0`` redetects every frame; larger values cut compute at the cost of catching new objects more slowly. See `Performance and Tuning`_.
- ``max_objects_per_prompt`` - cap on simultaneously tracked instances per prompt. Excess (lowest score) are evicted so the tracker stops propagating them.

Services are exposed to enable/disable or change the prompts and class IDs at run-time.


A few prompt-design lessons worth keeping in mind:

Prompts can be long phrases that group multiple physical objects under one ID - for example ``"car, bus, plane, or motorcycle"`` is *one* prompt.
Fewer prompts run faster as SAM3 will do processing over each prompt given. Try to consolidate as much as possible or use liberal use of the prompt change service.

5 - Build the ROS 2 workspace
-----------------------------

Build the SAM3 inference package inside the conda environment so the SAM3 node picks up the right Python.
After built, you do not need to activate the conda enviornment again - launching the node will automatically handle the environment correctly!

.. code-block:: bash

   conda activate opennav-sam3-inference
   source /opt/ros/jazzy/setup.bash
   cd <your_ros2_ws>
   colcon build --packages-select \
       opennav_sam3_inference
   conda deactivate

You can then proceed to build the rest of the package in or out of the conda environment.
In general, I think folks would prefer not to.

.. code-block:: bash

   cd <your_ros2_ws>
   colcon build --packages-skip \
       opennav_sam3_inference

Once built, source the workspace.

.. code-block:: bash

   source <your_ros2_ws>/install/setup.bash


6 - Launch the SAM3 inference node
----------------------------------

.. code-block:: bash

   ros2 launch opennav_sam3_inference sam3_inference.launch.py \
       image_topic:=<your image topic>

Within a few seconds you should see the three topics published.
It useful to visualize the segmentation mask topic to visualize the detection overlay to see it working!

TODO detection overlay gif

How that we have that working, we can focus on the costmap integration to use this information.

7 - Configure Nav2 to use the layer
-----------------------------------

The ``opennav_sam3_nav_demo`` package ships a Nav2 parameter file pre-wired for the SAM3 inference node for demonstration purposes.
The interesting parts are the costmap ``plugins`` lists and the ``semantic_segmentation_layer`` blocks.

.. code-block:: yaml

   local_costmap:
     local_costmap:
       ros__parameters:
         plugins: ["semantic_segmentation_layer", "inflation_layer"]
         semantic_segmentation_layer:
           plugin: "semantic_segmentation_layer::SemanticSegmentationLayer"
           enabled: True
           observation_sources: camera
           camera:
             segmentation_topic: "/sam3_inference/resized/label_mask"
             labels_topic:       "/sam3_inference/label_info"
             pointcloud_topic:   "/sensors/camera_0/points_registered"
             observation_persistence: 0.0
             expected_update_rate:    0.0
             visualize_tile_map:      True
             use_cost_selection:      True
             max_obstacle_distance:   3.5
             min_obstacle_distance:   0.3
             tile_map_decay_time:     1.5
             class_types: ["traversable", "avoid"]
             traversable:
               classes: ["floor"]
               base_cost: 15
               max_cost:  30
               mark_confidence:     0
               samples_to_max_cost: 20
               dominant_priority:   False
             avoid:
               classes: ["wall", "large static objects like furniture, columns, carts, boxes, or trash cans", "person, dog, or cat"]
               base_cost: 150
               max_cost:  254
               mark_confidence:     0
               samples_to_max_cost: 20
               dominant_priority:   False

The global costmap uses the same layer with a slightly longer ``tile_map_decay_time`` (3.0 s) so the map persists over the larger 40×40 m window.
The demo also keeps ``visualize_tile_map: True`` so you can see the underlying tile observations in RViz.

We say that floor is some non-zero low cost for demonstration purposes, so you can see that its correctly detected and functioning properly.
The obstacle classes are then set high as potential collisions to force the planner and controller to route around them.

This demonstration intentionally disables any lidar or depth based costmap layers such as the obstacle or voxel layers to showcase vision, semantic ONLY navigation.
This is probably not optimal for a production application where semantics are useful augmentations but still want to properly avoid collisions with detectable obstacles.

8 - Launch Nav2
---------------

With the SAM3 inference node from step 6 still running in one terminal, start Nav2 in another:

.. code-block:: bash

   ros2 launch opennav_sam3_nav_demo nav2.launch.py

Open RViz, set the fixed frame, and send a Nav2Goal. With the default prompt set, you should see:

- The robot's floor tiles colored low-cost (traversable).
- Walls colored high-cost (avoid), with the planner routing around them.

TODO hallway video


Real-World Demonstrations
=========================

The same setup, with three different prompt configurations, applied to three classes of environment.

Indoor Terrain
--------------

Prompts: ``["wall or large static objects like furniture, columns, carts, boxes, or trash cans", "floor", "person, dog, or cat"]``. Costs: walls and static objects are lethal, floor is low-cost (preferred), people / animals are high non-lethal cost so the planner gives them a wide berth but does not hard-fail.

.. todo::

   Indoor video #1 - the existing office hallway demo (already captured per
   ``README.md:67``). Embed as a YouTube iframe via ``.. raw:: html``.

.. todo::

   Indoor video #2 - Polymath office shoot (``README.md:68``). Should show the same
   prompts generalizing to a different indoor space without any reconfiguration.

.. todo::

   Indoor video #3 - multiple hallways in one continuous clip (``README.md:70``) to
   make the "the same model handles all of these without retraining" point visually.

.. note::

   This kind of single-camera sweep also exposes a useful side-effect: as the robot
   rotates and translates, the costmap layer accumulates semantic observations of
   walls, floor, and other classes across the entire space. With a longer
   ``tile_map_decay_time``, this effectively *annotates the static map during mapping*
   with semantic information - a building block for global semantic maps.

Outdoor Terrain
---------------

Two prompt configurations for two outdoor cases.

**Sidewalks vs. non-sidewalks.** Prompts: ``["sidewalk", "grass, trees, or benches"]``, with the second class as lethal cost. The planner stays on the sidewalk and treats grass/trees/benches as obstacles.

.. todo::

   Outdoor video #1 - Precita Park, San Francisco (``README.md:55``).

.. todo::

   Outdoor video #2 - Main Street Linear Park / A-7 Corsair II static aircraft
   display, Alameda (``README.md:56``, droneable).

**Bike lanes.** Prompts: ``["bike lane", "curb, street, plants, or grass"]``, with the second class as lethal cost. Same idea, different surface class.

.. todo::

   Bike-lane video #1 - Alameda bike lane down to Humble Sea (``README.md:60``,
   droneable).

.. todo::

   Bike-lane video #2 - left/right lane on a well-marked road in Alameda
   (``README.md:61``, droneable).

.. todo::

   Trail / plaza video - Glen Canyon, Point Reyes, Stern Grove, or the 4-square
   plaza in Alameda (``README.md:78-81``). Should show the system handling
   irregular natural surfaces rather than engineered paths.

Difficult Small Obstacles
-------------------------

The previous demos showcase terrain. The other side of foundation-model perception is that you can ask SAM3 about specific things - cables on the floor, a puddle, a piece of debris, a low-profile pallet - without training a detector for each one.

.. image:: docs/examples.gif
    :width: 90%
    :align: center
    :alt: SAM3 detecting cables, spills, debris, pallets, and other low-profile obstacles across various test images

Example prompt sets we have used:

.. code-block:: text

   ["cable or wire on the floor"]
   ["puddle or spill on the floor"]
   ["pallet or empty pallet jack"]
   ["wet floor sign"]

Each of these can be added or swapped at runtime via ``~/change_prompt``. None of them required collecting data or retraining anything.

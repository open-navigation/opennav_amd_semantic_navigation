.. _sam3_navigation_on_amd_strix_halo:

SAM3 Semantic Navigation on AMD X100 Strix Halo
***********************************************

- `Overview`_
- `Why SAM3 + Why Strix Halo`_
- `Requirements`_
- `Architecture Overview`_
- `Tutorial Steps`_
- `Swapping Prompts at Runtime`_
- `Real-World Demonstrations`_
- `Performance and Tuning`_
- `Conclusion`_
- `Related Projects`_

Overview
========

This tutorial walks through running a live, text-promptable semantic segmentation costmap for `Nav2 <https://docs.nav2.org/>`_ on the edge, using Meta's `SAM3 <https://ai.meta.com/sam3>`_ foundation model on an `AMD X100 Strix Halo <https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-personal-ai-supercomputing-guide.html>`_. By the end you will have:

- The SAM3 inference node running on the X100's GPU and publishing per-pixel class masks at 5–10 Hz.
- A Nav2 stack configured to consume those masks through the `semantic_segmentation_layer <https://github.com/kiwicampus/semantic_segmentation_layer>`_ costmap plugin.
- The ability to *change segmentation prompts at runtime* — swap from ``["floor", "wall"]`` to ``["sidewalk", "grass, trees, or benches"]`` to ``["pallet", "puddle"]`` without restarting anything.

SAM3 is *text-promptable*: you tell it what to find ("person", "pallet", "wet floor sign", "curb", "mud", "ceiling", ...) and it returns masks for each class. There is no per-class training step. The same model handles indoor floors, outdoor terrain, bike lanes, and arbitrary obstacles, so the costmap can be re-aimed at a new environment by editing a list of strings.

.. todo::

   Hero video (~30–45 s): robot navigating with the live ``~/segmentation_mask`` overlay
   composited in a corner. Should make it visually obvious that (a) the mask is updating
   in real time and (b) the costmap is responding to it. Embed via the ``.. raw:: html``
   YouTube iframe pattern used elsewhere on docs.nav2.org.

.. image:: docs/examples.gif
    :width: 90%
    :align: center
    :alt: SAM3 segmenting small, low, and unusual obstacles across many environments

.. note::

   Need ROS 2, Nav2, or deployment help? Contact `Open Navigation <https://www.opennav.org/>`_.

Why SAM3 + Why Strix Halo
=========================

Traditional perception for navigation tells you *where* obstacles are (from a depth camera or lidar), but not *what* they are or how the robot should treat them. It also struggles with thin or low-profile objects (cables, curbs, debris, spills) and with anything beyond the effective range of the depth sensor. Semantic segmentation fills that gap by labeling every pixel in the color image, even where depth is noisy or missing.

What's new with SAM3 is the *text prompt*. Older segmentation networks had a fixed class list baked in at training time — if you wanted to add a new object, you collected data and retrained. SAM3 takes a string. The "class list" is whatever you typed in the launch parameter, and you can change it at runtime via the ``~/change_prompt`` service. That is the property that makes a single model useful across warehouses, sidewalks, farms, and construction sites without retraining a thing.

Running a server-class foundation model at 5–10 Hz on a robot is the other half of the story. The AMD X100 Strix Halo (Ryzen AI Max+ 395) pairs an NPU and a Radeon GPU (``gfx1151``) with 32 x86 cores and unified memory in a single package, so SAM3 fits and runs on the edge without an external accelerator. That same compute is available for everything else a modern robot might want to run locally — detectors, VLMs, VLAs, LLMs, planners — without leaving the platform.

.. todo::

   Hardware photo: a shot of the AMD X100 Strix Halo dev kit (or the Ryzen AI Max+ 395
   mini-PC) ideally mounted on the demo robot, so readers can calibrate what
   "Strix Halo on a robot" looks like physically.

Requirements
============

This tutorial assumes you already have:

- An **AMD X100 Strix Halo** (Ryzen AI Max+ 395, ``gfx1151``) running **Ubuntu 24.04 Noble**.
- **ROS 2 Jazzy** installed and sourced.
- A robot platform with a **depth camera that produces an RGB image and a registered, aligned pointcloud** (Orbbec Gemini 335/355 and Intel RealSense families work well). The semantic costmap layer needs the color frame and the pointcloud to share intrinsics so the 2D mask can be projected into 3D — see `Architecture Overview`_.
- ``conda`` / ``miniforge`` for the Python environment SAM3 runs in. Install instructions: `miniforge <https://github.com/conda-forge/miniforge>`_.
- A `Hugging Face <https://huggingface.co>`_ account if you intend to download the SAM3 weights from the official Meta repository. A community mirror is also available — both are described in step 1.

.. code-block:: bash

   source /opt/ros/jazzy/setup.bash

.. todo::

   Sensor-mount photo: depth camera mounted on the demo robot, ideally taken from
   roughly the camera's height so readers can intuit the FOV. Helps motivate the
   "consider multi-camera or wide-disparity stereo for full scene coverage" point
   later on.

Architecture Overview
=====================

The pipeline is intentionally linear, camera → mask → costmap:

.. image:: docs/diagram.png
    :width: 90%
    :align: center
    :alt: Camera image flows into the SAM3 inference node, which publishes a label mask and label info; the semantic_segmentation_layer projects mask pixels via the depth camera's registered pointcloud into the Nav2 costmap

.. todo::

   Data-flow diagram refresh: ``docs/diagram.png`` is reusable, but please review
   that it renders legibly at 90% width on docs.nav2.org. If it looks dense at
   that size, a redraw that also shows text prompts entering the SAM3 node
   (and the ``~/change_prompt`` service) would emphasise the runtime-prompt-swap
   capability that distinguishes this tutorial.

The SAM3 node subscribes to a single color image topic, runs detection (and an optional tracker between full detections), and publishes:

- ``~/label_mask`` (``sensor_msgs/Image``, ``mono8``) — pixel value = class ID, 0 means "no detection".
- ``~/label_info`` (``vision_msgs/LabelInfo``) — class ID → class name mapping for downstream consumers.
- ``~/segmentation_mask`` (``sensor_msgs/Image``, ``rgb8``) — a per-class-tinted overlay of the source frame, for debugging and demos.

The ``sensor_processing_pipeline.launch.py`` include (also part of the inference package) decimates the color image, depth image, and label mask to a common resolution (default 424×240), then runs ``depth_image_proc::RegisterNode`` and ``depth_image_proc::PointCloudXyzNode`` to produce a registered pointcloud (``/sensors/camera_0/points_registered``) aligned to the mask. The semantic costmap layer takes those three pieces — the mask, the labels, and the registered pointcloud — and projects each labeled mask pixel into 3D, accumulating observations per costmap tile.

.. note::

   The pointcloud must be aligned to the color image (and therefore to the mask). If
   your camera driver provides a registered pointcloud you can decimate directly, you
   can bypass the resize/register/pointcloud nodes in the include file — see the
   comments inside ``sensor_processing_pipeline.launch.py``.

Tutorial Steps
==============

0 — Clone the repository
------------------------

The repo carries the ``semantic_segmentation_layer`` plugin as a git submodule, so use ``--recursive``:

.. code-block:: bash

   cd <your_ros2_ws>/src
   git clone --recursive https://github.com/open-navigation/opennav_amd_semantic_navigation.git

If you already cloned without ``--recursive``:

.. code-block:: bash

   cd opennav_amd_semantic_navigation
   git submodule update --init --recursive

1 — Get the SAM3 weights
------------------------

SAM3's weights (~3.3 GB) are not redistributed with the repo. You have two options. Either is fine; pick one.

**Option A — Official Meta release (requires accepting the license)**

#. Create or log into a Hugging Face account.
#. Accept the license at https://huggingface.co/facebook/sam3.
#. Generate an access token at https://huggingface.co/settings/tokens and log in with ``hf auth login``.
#. Download:

   .. code-block:: bash

      hf download facebook/sam3 model.safetensors --local-dir model/sam3

**Option B — Community mirror (no account, same weights)**

.. code-block:: bash

   hf download 1038lab/sam3 sam3.safetensors --local-dir model/sam3
   mv model/sam3/sam3.safetensors model/sam3/model.safetensors

In both cases you should end up with ``model/sam3/model.safetensors`` (plus the tokenizer/config files that already ship in the repo). The ``setup.sh`` script in the next step will offer to do Option B for you if you prefer to skip ahead.

2 — Install ROCm and Python dependencies
----------------------------------------

The ``opennav_sam3_inference`` package ships a one-shot setup script that installs the pinned ROCm 7.2 stack, a patched MIGraphX 2.15, an ``opennav-sam3-inference`` conda environment with ROCm-nightly PyTorch + ``onnxruntime-migraphx``, and (optionally) the model weights from Option B above.

.. code-block:: bash

   cd <your_ros2_ws>/src/opennav_amd_semantic_navigation/opennav_sam3_inference
   ./setup.sh

The script auto-detects the GPU and will warn if it does not see ``gfx1151``. It is safe to re-run; each step skips if its outputs are already present.

.. note::

   If you re-run after a partial install and want to skip the ROCm APT or MIGraphX steps,
   use ``./setup.sh --skip-apt --skip-migraphx``. To pin the conda environment name, use
   ``./setup.sh --env my-env-name``.

3 — Build the SAM3 inference artifacts
--------------------------------------

The SAM3 model has to be compiled into ONNX/MIGraphX artifacts for the X100's GPU. This runs once per resolution and takes ~18 minutes for the default 504-pixel input.

.. code-block:: bash

   conda activate opennav-sam3-inference
   cd <your_ros2_ws>/src/opennav_amd_semantic_navigation/opennav_sam3_inference

   # Single resolution (~18 min @ 504 px)
   python scripts/export/build.py --pipeline text --imgsz 504

You will see an ``onnx_files_504/`` directory in the working directory containing the build artifacts.

.. note::

   For both 504 and 1008 resolutions (~45 min total) run
   ``python scripts/export/build.py --pipeline text --imgsz 504 1008``.
   Higher resolution produces sharper masks at a ~2× cost; 504 is the right starting point.

Move ``model/sam3/`` and the ``onnx_files_504/`` directory to a stable location on the machine (anywhere you'd like — for example ``/opt/opennav/sam3/`` or ``~/opennav-sam3-models/``). You will reference them by absolute path in step 5.

4 — Build the ROS 2 workspace
-----------------------------

Build the packages inside the conda environment so the SAM3 node picks up the right Python:

.. code-block:: bash

   conda activate opennav-sam3-inference
   cd <your_ros2_ws>
   colcon build --packages-select \
       opennav_sam3_msgs \
       opennav_sam3_inference \
       semantic_segmentation_layer \
       opennav_sam3_nav_demo
   conda deactivate

Once built, source the workspace from a fresh shell. The launch file injects the right ROCm and MIGraphX library paths via ``LD_LIBRARY_PATH`` and ``LD_PRELOAD``, so you do not need the conda environment active to *run* — only to build.

.. code-block:: bash

   source /opt/ros/jazzy/setup.bash
   source <your_ros2_ws>/install/setup.bash

5 — Configure paths
-------------------

Edit ``opennav_sam3_inference/config/sam3_inference.yaml`` so the node knows where the weights and artifacts live:

.. code-block:: yaml

   sam3_inference:
     ros__parameters:
       checkpoint: /absolute/path/to/model/sam3              # contains model.safetensors
       onnx_dir:   /absolute/path/to/onnx_files_504          # contains the build artifacts
       prompts:    ["floor", "wall"]
       class_ids:  [1, 2]
       device:     cuda                                      # correct on ROCm — PyTorch reuses CUDA names for HIP
       score_threshold:        0.5
       max_objects_per_prompt: 5
       redetect_interval_ms:   0.0
       queue_depth:            5
       start_enabled:          true

A few notes on the parameters you will tune most often:

- ``prompts`` / ``class_ids`` — parallel arrays. Each prompt becomes a class with the matching ID written into ``~/label_mask`` at every detected pixel. IDs must be in ``[1, 255]`` and unique (0 is reserved for "no detection"). Prompts can be long phrases that group multiple physical objects under one ID — for example ``"car, bus, plane, or motorcycle"`` is *one* prompt that lights up four kinds of vehicle as the same class.
- ``score_threshold`` / ``mask_threshold`` — confidence thresholds for whole detections and for per-pixel binarization. Live-editable via parameter updates.
- ``redetect_every`` — run full SAM3 detection every N frames; track-only on the rest. ``1`` redetects every frame; larger values cut compute at the cost of catching new objects more slowly. See `Performance and Tuning`_.
- ``max_objects_per_prompt`` — cap on simultaneously tracked instances per prompt. Excess (lowest score) are evicted so the tracker stops propagating them.

6 — Launch the SAM3 inference node
----------------------------------

.. code-block:: bash

   ros2 launch opennav_sam3_inference sam3_inference.launch.py \
       image_topic:=/sensors/camera_0/color/image

Within a few seconds you should see three topics published:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Topic
     - Type
     - Description
   * - ``~/label_mask``
     - ``sensor_msgs/Image`` (``mono8``)
     - Pixel value = ``class_id`` of the top-scoring class at that pixel; ``0`` means no detection.
   * - ``~/label_info``
     - ``vision_msgs/LabelInfo``
     - Maps ``class_id`` to ``class_name`` for use of ``label_mask`` downstream.
   * - ``~/segmentation_mask``
     - ``sensor_msgs/Image`` (``rgb8``)
     - Per-class-tinted overlay of the source frame, for validation. Colors are deterministic per ``class_id``.

And two services let you steer the node without restarting it:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Service
     - Type
     - Description
   * - ``~/change_prompt``
     - ``opennav_sam3_msgs/srv/ChangePrompt``
     - Replace the prompts and class IDs live (e.g. ``[("person", 1), ("pallet", 2), ("puddle", 3)]``).
   * - ``~/enable``
     - ``std_srvs/srv/SetBool``
     - Pause / resume inference.

Visualize ``~/segmentation_mask`` in RViz to confirm the prompts are doing what you expect before you turn on Nav2.

.. todo::

   RViz overlay screenshot: the ``~/segmentation_mask`` topic visualized side-by-side
   with the raw camera image, with the ``prompts: ["floor", "wall"]`` configuration
   active. Should make it obvious which pixels belong to which class.

7 — Understand the semantic_segmentation_layer cost model
---------------------------------------------------------

Before you turn on Nav2, it's worth understanding what the costmap layer is going to do with the SAM3 mask, because the layer's parameters are where you actually express *robot behavior*.

**Inputs to the layer.** Per camera observation source:

- A segmentation mask topic (``~/label_mask``) — each pixel is a ``class_id``.
- A label info topic (``~/label_info``) — the ``class_id`` → name table.
- A registered pointcloud topic (``/sensors/camera_0/points_registered``) — same resolution as the mask, with each pixel's 3D position.

**What it does.** For each pixel that has both a class ID and a valid 3D point, the layer projects the point into the costmap frame and drops the class as an "observation" into whichever costmap tile it lands in. Tiles accumulate observations over time and decay them.

**Class types.** Rather than mapping a class ID directly to a cost, the layer groups classes into named *class types*, each with its own cost curve. From [opennav_sam3_nav_demo/config/nav2_semantic_params.yaml](opennav_sam3_nav_demo/config/nav2_semantic_params.yaml):

.. code-block:: yaml

   class_types: ["traversable", "avoid"]
   traversable:
     classes: ["floor"]
     base_cost: 15                # cost when a single observation has arrived
     max_cost: 30                 # cost once samples_to_max_cost have accumulated
     mark_confidence: 0           # average confidence threshold to apply max_cost
     samples_to_max_cost: 20      # number of observations needed to apply max_cost
     dominant_priority: False     # if True, this type wins outright and clears others
   avoid:
     classes: ["wall"]
     base_cost: 150
     max_cost: 200
     mark_confidence: 0
     samples_to_max_cost: 20
     dominant_priority: False

A tile that has seen one ``floor`` observation gets cost ``15``. After 20 ``floor`` observations within the decay window, it ramps to ``30``. The ``avoid`` curve is the same shape but at much higher costs — one ``wall`` observation reads as ``150``, fully-confirmed wall reads as ``200`` (effectively lethal).

**Resolving conflicts.** When a tile has observations of both ``traversable`` and ``avoid`` classes (it happens — a "wall" pixel sitting on top of a "floor" tile due to small angular errors), ``use_cost_selection: True`` keeps the *highest-cost* class. That means a single ``avoid`` observation outranks a long history of ``traversable`` observations — which is what you want for safety.

**Forgetting.** ``tile_map_decay_time`` (1.5 s local, 3.0 s global in the demo config) controls how long observations live before being aged out. Short decay times keep the costmap responsive to dynamic scenes; long decay times build up a more stable map but lag behind change.

Conceptually:

.. code-block:: text

   Camera frame                     Costmap tile
   ┌─────────────────┐              ┌──────────┐
   │ pixel (u,v)=floor│  ──proj──>  │ +floor   │  ──>  base_cost=15  (1 obs)
   │ pixel (u,v)=wall │  ──proj──>  │ +wall    │  ──>  base_cost=150 (1 obs)
   │  ...             │             │  ...     │
   └─────────────────┘              │ +floor×20│  ──>  max_cost=30   (20 obs, no wall)
                                    │ +wall×1  │  ──>  cost=150      (wall wins via
                                    └──────────┘       use_cost_selection)

That's the whole model. Everything else in the layer's parameter block is bookkeeping (which sources, which topics, which ranges).

8 — Configure Nav2 to use the layer
-----------------------------------

The ``opennav_sam3_nav_demo`` package ships a complete Nav2 parameter file pre-wired for the SAM3 inference node. The interesting parts are the costmap ``plugins`` lists and the ``semantic_segmentation_layer`` blocks. From [opennav_sam3_nav_demo/config/nav2_semantic_params.yaml](opennav_sam3_nav_demo/config/nav2_semantic_params.yaml):

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
               classes: ["wall"]
               base_cost: 150
               max_cost:  200
               mark_confidence:     0
               samples_to_max_cost: 20
               dominant_priority:   False

The global costmap uses the same layer with a slightly longer ``tile_map_decay_time`` (3.0 s) so the map persists over the larger 40×40 m window. The demo also keeps ``visualize_tile_map: True`` so you can see the underlying tile observations in RViz.

.. warning::

   The demo configuration intentionally **disables the depth-based voxel layer** so the
   semantic contribution is visually obvious. For a production deployment you should
   keep depth-based costmap layers (voxel or obstacle) as a backup — semantic
   segmentation is powerful but it is not a replacement for direct geometric
   measurement, and a model that misclassifies a pixel should not be the only
   barrier between your robot and a person. The disabled voxel layer is left
   commented out in ``nav2_semantic_params.yaml`` as a reference for re-enabling it.

9 — Launch Nav2 and drive
-------------------------

With the SAM3 inference node from step 6 still running in one terminal, start Nav2 in another:

.. code-block:: bash

   ros2 launch opennav_sam3_nav_demo nav2.launch.py

Open RViz, set the fixed frame, and send a Nav2Goal. With the default ``["floor", "wall"]`` prompt set, you should see:

- The robot's floor tiles colored low-cost (traversable).
- Walls colored high-cost (avoid), with the planner routing around them.
- If you enabled ``visualize_tile_map``, an extra pointcloud topic showing the raw tile observations colored by confidence — useful for diagnosing why a particular tile got the cost it did.

.. todo::

   RViz costmap screenshot: global + local costmap rendered with ``visualize_tile_map: True``
   enabled, showing floor (low cost) and wall (high cost) tiles clearly, ideally with a
   planned path threading down a hallway. Capture in the same office/hallway environment
   used in the Indoor Terrain demonstration below for visual continuity.

Swapping Prompts at Runtime
===========================

The reason to use a foundation model instead of a fixed-class network is that you can re-aim it. With the SAM3 node still running, change what it's looking for without restarting anything:

.. code-block:: bash

   ros2 service call /sam3_inference/change_prompt opennav_sam3_msgs/srv/ChangePrompt \
       "{prompts: ['sidewalk', 'grass, trees, or benches'], class_ids: [1, 2]}"

The ``~/segmentation_mask`` overlay should update within one or two frames to reflect the new classes. The label mask, label info, and downstream costmap layer all pick up the new class IDs automatically — though if you change the *meaning* of a class ID (e.g. ID ``2`` was ``wall`` and is now ``grass``), you should also update the layer's ``traversable.classes`` / ``avoid.classes`` lists in the Nav2 config, otherwise the costmap will continue to treat ID ``2`` under whatever name it was bound to.

A few prompt-design lessons worth keeping in mind:

- **Long prompts are fine.** ``"person, child, dog, or bicycle"`` is one class. SAM3 happily groups several physical things under a single ID, which is often what you want for navigation.
- **Fewer prompts run faster.** See the table in `Performance and Tuning`_. Two prompts at 504 px hit ~7 Hz; four prompts roughly halve that. Use prompt grouping rather than multiplying prompts when you can.
- **Be specific about what you don't want.** "curb, street, plants, or grass" works well as the "stay out of this" prompt for sidewalk navigation, because the contrast with "sidewalk" is sharp.

.. todo::

   Live-prompt-swap clip (~10 s GIF or short video): the segmentation overlay before
   and after the ``change_prompt`` service call, with the new classes immediately
   visible. Pair the visual with the actual ``ros2 service call`` shown on screen
   if possible.

Real-World Demonstrations
=========================

The same setup, with three different prompt configurations, applied to three classes of environment.

Indoor Terrain
--------------

Prompts: ``["wall or large static objects like furniture, columns, carts, boxes, or trash cans", "floor", "person, dog, or cat"]``. Costs: walls and static objects are lethal, floor is low-cost (preferred), people / animals are high non-lethal cost so the planner gives them a wide berth but does not hard-fail.

.. todo::

   Indoor video #1 — the existing office hallway demo (already captured per
   ``README.md:67``). Embed as a YouTube iframe via ``.. raw:: html``.

.. todo::

   Indoor video #2 — Polymath office shoot (``README.md:68``). Should show the same
   prompts generalizing to a different indoor space without any reconfiguration.

.. todo::

   Indoor video #3 — multiple hallways in one continuous clip (``README.md:70``) to
   make the "the same model handles all of these without retraining" point visually.

.. note::

   This kind of single-camera sweep also exposes a useful side-effect: as the robot
   rotates and translates, the costmap layer accumulates semantic observations of
   walls, floor, and other classes across the entire space. With a longer
   ``tile_map_decay_time``, this effectively *annotates the static map during mapping*
   with semantic information — a building block for global semantic maps.

Outdoor Terrain
---------------

Two prompt configurations for two outdoor cases.

**Sidewalks vs. non-sidewalks.** Prompts: ``["sidewalk", "grass, trees, or benches"]``, with the second class as lethal cost. The planner stays on the sidewalk and treats grass/trees/benches as obstacles.

.. todo::

   Outdoor video #1 — Precita Park, San Francisco (``README.md:55``).

.. todo::

   Outdoor video #2 — Main Street Linear Park / A-7 Corsair II static aircraft
   display, Alameda (``README.md:56``, droneable).

**Bike lanes.** Prompts: ``["bike lane", "curb, street, plants, or grass"]``, with the second class as lethal cost. Same idea, different surface class.

.. todo::

   Bike-lane video #1 — Alameda bike lane down to Humble Sea (``README.md:60``,
   droneable).

.. todo::

   Bike-lane video #2 — left/right lane on a well-marked road in Alameda
   (``README.md:61``, droneable).

.. todo::

   Trail / plaza video — Glen Canyon, Point Reyes, Stern Grove, or the 4-square
   plaza in Alameda (``README.md:78-81``). Should show the system handling
   irregular natural surfaces rather than engineered paths.

Difficult Small Obstacles
-------------------------

The previous demos showcase terrain. The other side of foundation-model perception is that you can ask SAM3 about specific things — cables on the floor, a puddle, a piece of debris, a low-profile pallet — without training a detector for each one.

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

Performance and Tuning
======================

Measured on AMD X100 Strix Halo with the 504 px artifact, single-camera input:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Prompts
     - New detect every frame
     - Detect every 1 s, track in-between
   * - 1
     - 154.6 ms (6.46 Hz)
     - 103.8 ms (9.63 Hz)
   * - 2
     - 183.3 ms (5.45 Hz)
     - 136.0 ms (7.35 Hz)
   * - 4
     - 305.6 ms (3.27 Hz)
     - 211.3 ms (4.73 Hz)

A few practical levers if you need to move along this curve:

- **redetect_every.** ``1`` runs full detection every frame and is the most reactive setting. Larger values run cheaper tracker-only updates between full detections. The trade-off is responsiveness to *new* objects entering the frame: with ``redetect_every: 30`` and a 6 Hz inference rate, you might wait up to 5 s to pick up a person who just walked in. Tune to the dynamism of your environment and the field of view of your camera.
- **Number of prompts.** Cost grows approximately linearly with number of prompts. Group physical classes under a single phrase ("person, child, dog, or bicycle") instead of multiplying prompts whenever the grouped classes share a behavioral treatment.
- **imgsz.** 504 is the default and is a good speed/quality balance. The 1008 pipeline (built with ``python scripts/export/build.py --pipeline text --imgsz 504 1008``) produces sharper masks for far-field or fine-grained classes at roughly 2× cost.
- **score_threshold / mask_threshold.** Both live-editable. Raising them suppresses low-confidence detections (useful for noisy outdoor scenes); lowering them surfaces more borderline observations (useful for picking up small or distant objects). Defaults of ``0.5`` / ``0.5`` are a reasonable starting point.
- **max_objects_per_prompt.** Bounds the tracker's memory; lowest-score excess instances are evicted. Increase if you genuinely need to track many simultaneous instances of one class.

Conclusion
==========

You now have a real-time, text-promptable semantic costmap running on the edge: SAM3 on the X100 Strix Halo's GPU producing per-pixel class masks, the ``semantic_segmentation_layer`` projecting those masks into Nav2's costmap, and a service interface for changing what the robot is looking for without restarting.

There are several directions this opens up:

- **Behavior trees.** ``~/label_info`` plus class IDs from ``~/label_mask`` are enough to write BT conditions like "if a 'person' class is currently detected, switch to a slower controller." The fact that prompts are runtime-editable means a BT can also *change what's being detected* depending on the task.
- **Dynamic-obstacle extraction.** A "person, dog, or cat" prompt produces a stream of masks that can be used to *remove* dynamic agents from a localization pipeline (e.g. to keep AMCL or a scan-matcher from latching onto walking people), or to feed a dedicated tracker.
- **Wider FOV.** A single Orbbec or RealSense will run this fine, but for outdoor terrain navigation in particular you may want a wide-disparity stereo rig or multiple synchronized depth cameras to fully capture the semantic richness of the scene.
- **Beyond navigation.** The X100 has plenty of headroom — the same compute that runs SAM3 at 5–10 Hz can co-host VLMs, VLAs, and LLMs for higher-level decision making.

.. note::

   Need ROS 2, Nav2, or deployment help? Contact
   `Open Navigation <https://www.opennav.org/>`_.

Related Projects
================

- `opennav_amd_demonstrations <https://github.com/open-navigation/opennav_amd_demonstrations>`_ — companion project demonstrating indoor 2D, urban 3D, outdoor GPS-based navigation, and now semantic segmentation navigation on Ryzen AI / X100 computers with the Honeybee reference platform.

  .. todo::

     Confirm the exact section/anchor on opennav_amd_demonstrations to deep-link to
     for readers who want to see the Honeybee end-to-end demo using SAM3, then
     replace the bare repo link with a section link.

- `Nav2 <https://github.com/ros-navigation/navigation2>`_ — the ROS 2 navigation stack this work plugs into.
- `semantic_segmentation_layer <https://github.com/kiwicampus/semantic_segmentation_layer>`_ — the costmap plugin used in step 8.
- `Ryzers <https://github.com/AMDResearch/Ryzers>`_ — pre-configured and optimized Docker images for AI on AMD GPUs.
- :ref:`navigation2_with_semantic_segmentation` — companion Nav2 tutorial covering semantic segmentation in simulation with a fixed-class ONNX model, useful background reading for the costmap concepts re-explained in step 7.

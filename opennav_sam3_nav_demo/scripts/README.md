## To record the rosbag

Run this in the directory you wish to save the rosbag

```bash
ros2 run opennav_sam3_nav_demo record_semantic_data.sh
```

## To replay the rosbag

Run this preferably in the same path as the rosbag or provide a full path as an argument

```bash
ros2 run opennav_sam3_nav_demo replay_semantic_data.sh <path/to/rosbag>
```

Before replaying, start SAM3 node and Nav2 in two different terminals with `use_sim_time` parameter set to `true`.

### SAM3 node

```bash
ros2 launch opennav_sam3_inference sam3_inference.launch.py use_sim_time:=true
```

### Nav2

```bash
ros2 launch opennav_sam3_nav_demo nav2.launch.py use_sim_time:=true
```
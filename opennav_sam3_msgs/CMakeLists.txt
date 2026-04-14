cmake_minimum_required(VERSION 3.10)
project(opennav_sam3_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ChangePrompt.srv"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

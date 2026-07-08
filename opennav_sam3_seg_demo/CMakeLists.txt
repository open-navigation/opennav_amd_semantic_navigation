cmake_minimum_required(VERSION 3.8)
project(opennav_sam3_seg_demo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Find dependencies

set(QT_COMPONENTS
  Core
  Gui
  Widgets
)
find_package(Qt5 REQUIRED COMPONENTS ${QT_COMPONENTS})

find_package(ament_cmake)

# List panel dependencies here
set(CHANGE_PROMPT_PANEL_DEPS
  rclcpp
  ament_index_cpp
  qt_gui_cpp
  pluginlib
)
set(CHANGE_PROMPT_PANEL_TARGETS "")

foreach(pkg IN ITEMS ${CHANGE_PROMPT_PANEL_DEPS})
  find_package(${pkg} REQUIRED)
  list(APPEND CHANGE_PROMPT_PANEL_TARGETS ${pkg}::${pkg})
endforeach()

# List panel ROS interfaces here
set(MSG_DEPS
  opennav_sam3_msgs
)
set(MSG_TARGETS "")

foreach(pkg IN ITEMS ${MSG_DEPS})
  find_package(${pkg} REQUIRED)
  list(APPEND MSG_TARGETS ${${pkg}_TARGETS})
endforeach()

set(UI_FILES
  ui/change_prompt_panel.ui
)

set(${PROJECT_NAME}_HEADERS
  include/change_prompt_panel.hpp
)

qt5_wrap_ui(UI_HEADERS ${UI_FILES})
qt5_wrap_cpp(MOCS ${${PROJECT_NAME}_HEADERS})

# Panel executable
set(EXECUTABLE_NAME change_prompt_panel)

add_executable(${EXECUTABLE_NAME}
  src/change_prompt_panel.cpp
  "${UI_HEADERS}"
  "${MOCS}"
)

set(QT_TARGETS ${QT_COMPONENTS})
list(TRANSFORM QT_TARGETS PREPEND "Qt5::")

target_link_libraries(${EXECUTABLE_NAME}
  "${QT_TARGETS}"
  "${CHANGE_PROMPT_PANEL_TARGETS}"
  "${MSG_TARGETS}"
)

target_include_directories(${EXECUTABLE_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/icons"
  DESTINATION "share/${PROJECT_NAME}"
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/launch"
  DESTINATION "share/${PROJECT_NAME}"
)

install(
  TARGETS ${EXECUTABLE_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

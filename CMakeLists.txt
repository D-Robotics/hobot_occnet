cmake_minimum_required(VERSION 3.8)
project(hobot_occnet)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# setting variables
set(SYS_ROOT ${CMAKE_SYSROOT})
message("=> SYS_ROOT is " ${SYS_ROOT})

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
include_directories(${SYS_ROOT}/usr/include)
include_directories(${SYS_ROOT}/usr/hobot/include)
link_directories(
  ${SYS_ROOT}/usr/lib/hbbpu
  ${SYS_ROOT}/usr/hobot/lib
  ${SYS_ROOT}/usr/lib
)

# add executable
add_executable(hobot_occnet_node src/main.cpp src/occnet_node.cpp src/occnet_infer.cpp src/img_convert_utils.cpp)
target_compile_features(hobot_occnet_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_include_directories(hobot_occnet_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(hobot_occnet_node
  dnn
)
ament_target_dependencies(hobot_occnet_node
  rclcpp
  OpenCV
  sensor_msgs
)

# install
install(TARGETS hobot_occnet_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

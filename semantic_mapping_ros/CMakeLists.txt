cmake_minimum_required(VERSION 3.15)

project(semantic_mapping_ros)

set(CMAKE_CXX_STANDARD 17)
# Add -o3 to compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

## Find system libraries
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  std_msgs
  sensor_msgs
  visualization_msgs
  std_msgs 
  message_generation
  message_runtime
  cv_bridge
  tf_conversions
  semantic_mapping
)

add_service_files(
  FILES
  FilePath.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs message_runtime message_generation cv_bridge tf_conversions semantic_mapping
  DEPENDS
)

###########
# INCLUDE #
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

#############
# LIBRARIES #
#############

add_library(semantic_mapping_ros
  src/VoxelHashMapRos.cpp
)

target_link_libraries(semantic_mapping_ros
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  Eigen3::Eigen
)

add_executable(voxel_map
  src/VoxelHashMapRos_node.cpp
  src/VoxelHashMapRos.cpp
)

target_link_libraries(voxel_map
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  Eigen3::Eigen
)

##########
# EXPORT #
##########
install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

cmake_minimum_required(VERSION 3.15)

project(semantic_mapping)

set(CMAKE_CXX_STANDARD 17)
# Add -o3 to compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

## Find system libraries
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

find_package(catkin REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  LIBRARIES
  CATKIN_DEPENDS
  DEPENDS EIGEN3 Boost
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

add_executable(map_evaluator
  src/MapEvaluator.cpp
)
target_link_libraries(map_evaluator  
  ${catkin_LIBRARIES} 
  ${Boost_LIBRARIES}
  Eigen3::Eigen
)

##########
# EXPORT #
##########
# install(DIRECTORY include/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
# )

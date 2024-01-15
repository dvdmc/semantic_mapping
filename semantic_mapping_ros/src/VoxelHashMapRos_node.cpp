#include "ros/ros.h"

#include "semantic_mapping_ros/VoxelHashMapRos.hpp"

int main(int argc, char **argv) {
    ros::init(argc, argv, "voxel_hash_map_node");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    ROS_INFO("Starting voxel_hash_map_node...");
    semantic_mapping::VoxelHashMapNode voxel_hash_map_node(nh, nh_private);

    while (ros::ok()) {
        ros::spinOnce();
    }
    return 0;
}

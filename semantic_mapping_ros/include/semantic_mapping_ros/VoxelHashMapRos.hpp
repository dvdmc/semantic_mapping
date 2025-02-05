#ifndef SEMANTIC_MAPPING_MAP_ROS_H_
#define SEMANTIC_MAPPING_MAP_ROS_H_

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/String.h>
#include <std_srvs/Empty.h>
#include <tf/transform_listener.h>

#include <semantic_mapping/VoxelHashMap.hpp>

#include <semantic_mapping_ros/FilePath.h>

namespace semantic_mapping {

// Define ENUM with the DL method
enum class DlMethod {
    MCD,  // The node expects a point cloud with _num_mc_samples_ probability
          // vectors and combine them using Monte Carlo integration.
    DET   // The node expects a point cloud with one probability vector per 3D
          // point.
};

class VoxelHashMapNode {
   public:
    VoxelHashMapNode(ros::NodeHandle &nh, ros::NodeHandle &nh_private);
    ~VoxelHashMapNode();

    // Accessor
    inline std::shared_ptr<semantic_mapping::VoxelHashMap> getVoxelHashMapSharedPtr() {
        // TODO: Not thread safe at the moment. Could be a problem if
        // a pointcloud is received at the same time as the map is being
        // accessed for planning.
        return voxel_hash_map_;
    };

    inline std::shared_ptr<semantic_mapping::VoxelIntegrator> getVoxelIntegratorSharedPtr() {
        return voxel_integrator_;
    };

    inline float getVoxelSize() { return p_resolution_; };

    inline bool getExperimentSave() { return p_save_experiment_; };
    inline int getExperimentSaveEachNUpdates() { return p_save_each_n_updates_; };
    inline std::string getExperimentSavePath() { return p_save_path_; };
    inline int getSeqNumber() { return seq_number_; };
    
   private:
    // Members
    std::shared_ptr<VoxelHashMap> voxel_hash_map_;
    // The planner is abstracted in a base class to allow for different planners
    std::shared_ptr<VoxelIntegrator> voxel_integrator_;

    // Control
    bool did_voxel_map_update_;

    // ROS related members
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    std::string map_frame_;
    ros::Time last_update_time_;

    // Subscribers
    ros::Subscriber sub_pcd_;

    // Publishers
    ros::Publisher pub_voxel_markers_;
    ros::Publisher pub_background_markers_;

    // Services
    ros::ServiceServer srv_save_map_;
    ros::ServiceServer srv_save_pgm_;
    ros::ServiceServer srv_save_map_request_;
    ros::ServiceServer srv_open_map_;
    ros::ServiceServer srv_evaluate_map_;
    ros::ServiceClient srv_client_sensor_;

    // Timers
    ros::Timer vis_timer_;

    // TF
    tf::TransformListener tf_listener_;

    // Configuration

    // Sensor related

    int p_n_classes_;
    int p_n_samples_mc_;
    float p_depth_threshold_;
    DlMethod p_dl_method_;

    // Map related

    float p_resolution_;

    // Fusion method
    FusionMethod p_fusion_method_;
    UncertaintyType p_uncertainty_method_;
    float p_beta_;

    // Visualization
    bool p_visualize_semantics_;
    double p_vis_sem_freq_, p_vis_top_height_;
    std::vector<std::vector<uint8_t>> label_to_rgb_;
    // Initialized from semantic utils from the number of classes.

    // Experiment parameters

    bool p_save_experiment_;
    std::string p_save_directory_path_;
    std::string p_experiment_map_name_;
    std::string p_variant_name_; // Used to differentiate between different
                                 // experiments with the same method.
    std::string p_save_path_;
    int p_save_each_n_updates_;
    bool p_save_last_only_;

    int seq_number_;
    int n_updates_since_last_save_;

    // Methods

    void rosInit();
    void obtainMCClassProb(
        const sensor_msgs::PointCloud2ConstIterator<float> &iter_class_prob,
        VoxelInfo &processed_measurement);
    void obtainDetClassProb(
        const sensor_msgs::PointCloud2ConstIterator<float> &iter_class_prob,
        VoxelInfo &processed_measurement);

    // Callbacks

    void pcdCallback(sensor_msgs::PointCloud2ConstPtr const &msg);

    // Services
    bool saveMapRequestSrvCallback(
        std_srvs::Empty::Request &req,
        std_srvs::Empty::Response &res);
    bool savePGMMapSrvCallback(
        semantic_mapping_ros::FilePath::Request &req,
        semantic_mapping_ros::FilePath::Response &res);
        
    bool saveMapSrvCallback(semantic_mapping_ros::FilePath::Request &req,
                            semantic_mapping_ros::FilePath::Response &res);
    bool openMapSrvCallback(semantic_mapping_ros::FilePath::Request &req,
                            semantic_mapping_ros::FilePath::Response &res);
    bool evaluateMapSrvCallback(std_srvs::Empty::Request &req,
                                std_srvs::Empty::Response &res);

    // Timers

    void publishVoxelMarkers(const ros::TimerEvent &event);
    void plannerCallback(const ros::TimerEvent &event);

    bool tryGetTransform(const std::string &frame_id, const std::string &origin,
                         const ros::Time &stamp, Eigen::Affine3f &T_Map_Cam,
                         tf::StampedTransform &transform);

    bool saveVoxelMap();
};

}  // namespace semantic_mapping

#endif
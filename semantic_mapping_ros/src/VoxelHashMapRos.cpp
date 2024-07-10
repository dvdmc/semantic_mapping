#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <geometry_msgs/Pose.h>
#include <std_srvs/SetBool.h>
#include <tf/LinearMath/Vector3.h>
#include <tf/tf.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/MarkerArray.h>

#include <semantic_mapping/SemanticUtils.hpp>

#include <semantic_mapping_ros/VoxelHashMapRos.hpp>

namespace semantic_mapping {

namespace fs = std::filesystem;

// Maps created for ROS parameters to enums
std::unordered_map<std::string, FusionMethod> fusion_method_map = {
    {"KF", FusionMethod::KF},
    {"W_BAY", FusionMethod::W_BAY},
    {"BAY", FusionMethod::BAY},
    {"SUM", FusionMethod::SUM},
    {"SUM_PROBS", FusionMethod::SUM_PROBS}};

std::unordered_map<std::string, UncertaintyType> uncertainty_method_map = {
    {"UNCERTAINTY", UncertaintyType::UNCERTAINTY},
    {"CONFIDENCE", UncertaintyType::CONFIDENCE}};

std::unordered_map<std::string, DlMethod> dl_method_map = {
    {"DET", DlMethod::DET}, {"MCD", DlMethod::MCD}};

VoxelHashMapNode::VoxelHashMapNode(ros::NodeHandle &nh,
                                   ros::NodeHandle &nh_private)
    : nh_(nh), nh_private_(nh_private) {
    did_voxel_map_update_ = false;
    rosInit();
    // Create voxel hash map
    voxel_hash_map_ =
        std::make_shared<VoxelHashMap>(p_resolution_, p_n_classes_);
    // Configure the integrator to process measurements coming from the sensor
    // into the map
    voxel_integrator_ = std::make_shared<VoxelIntegrator>(
        p_n_classes_, p_fusion_method_, p_uncertainty_method_, p_beta_);

    // Assign integrator to voxel hash map
    voxel_hash_map_->setIntegrator(voxel_integrator_);
}

VoxelHashMapNode::~VoxelHashMapNode() {}

void VoxelHashMapNode::rosInit() {
    if (!nh_private_.getParam("depth_threshold", p_depth_threshold_)) {
        ROS_WARN("depth_threshold not set, using default: 7");
        p_depth_threshold_ = 7;
    }
    if (!nh_private_.getParam("n_classes", p_n_classes_)) {
        ROS_WARN("n_classes not set, using default: %d", 21);
        p_n_classes_ = 21;
    }
    if (!nh_private_.getParam("resolution", p_resolution_)) {
        ROS_WARN("resolution not set, using default: 0.1");
        p_resolution_ = 0.1;
    }

    std::string dl_method, fusion_method, uncertainty_method, controller_type;
    // Exp parameters related to p(x|z_1...t) = p(x|z_1...t-1) * p(z_t|x)
    if (!nh_private_.getParam("method/dl_method", dl_method)) {
        ROS_WARN("dl_method not, using default: MCD");
        dl_method = "MCD";
    }
    if (!nh_private_.getParam("method/fusion_method", fusion_method)) {
        ROS_WARN("fusion_method not set, using default: W_BAY");
        fusion_method = "W_BAY";
    }
    if (!nh_private_.getParam("method/uncertainty_method",
                              uncertainty_method)) {
        // If we are using uncerts as in MCD or confidence based method
        ROS_WARN("uncertainty_method not set, using default: UNCERTAINTY");
        uncertainty_method = "UNCERTAINTY";
    }
    if (!nh_private_.getParam("method/beta", p_beta_)) {
        // Beta for regularization
        ROS_WARN("beta not set, using default: 0.3");
        p_beta_ = 0.3;
    }
    if (!nh_private_.getParam("map_frame", map_frame_)) {
        ROS_WARN("map_frame not set, using default: map");
        map_frame_ = "map";
    }
    if (!nh_private_.getParam("save_experiment", p_save_experiment_)) {
        ROS_WARN("save_experiment not set, using default: false");
        p_save_experiment_ = false;
    }
    if (!nh_private_.getParam("save_each_n_updates", p_save_each_n_updates_)) {
        ROS_WARN("save_each_n_updates not set, using default: 10");
        p_save_each_n_updates_ = 10;
    }
    if (!nh_private_.getParam("variant", p_variant_name_)) {
        ROS_ERROR("variant not set, using default: \"\"");
        p_variant_name_ = "";
        return;
    }
    if (!nh_private_.getParam("experiment_save_path", p_save_directory_path_)) {
        ROS_WARN(
            "experiment_save_path not set, using default: "
            "/tmp/semantic_mapping/");
        p_save_directory_path_ = "/tmp/semantic_mapping/";
    }
    if (!nh_private_.getParam("experiment_map_name", p_experiment_map_name_)) {
        ROS_WARN(
            "experiment_save_path not set, using default: "
            "/tmp/semantic_mapping/");
        p_experiment_map_name_ = "unnammed";
    }
    if (!nh_private_.getParam("visualize_semantics", p_visualize_semantics_)) {
        ROS_WARN("visualize_semantics not set, using default: true");
        p_visualize_semantics_ = true;
    }
    if (!nh_private_.getParam("visualize_semantics_frequency",
                              p_vis_sem_freq_)) {
        ROS_WARN("visualize_semantics_frequency not set, using default: 1.0");
        p_vis_sem_freq_ = 1.0;
    }
    if (!nh_private_.getParam("visualization_top_height", p_vis_top_height_)) {
        ROS_WARN("visualization_top_height not set, using default: 3.0");
        p_vis_top_height_ = 3.0;
    }

    // Parameters processing

    label_to_rgb_ = getLabelMap(p_n_classes_);

    p_dl_method_ = dl_method_map[dl_method];
    p_fusion_method_ = fusion_method_map[fusion_method];
    p_uncertainty_method_ = uncertainty_method_map[uncertainty_method];

    if (p_dl_method_ == DlMethod::MCD) {
        // If we are using MCD we need to sample from the network
        if (!nh_private_.getParam("method/n_samples_mc", p_n_samples_mc_)) {
            ROS_WARN("n_samples_mc not set, using default: 10");
            p_n_samples_mc_ = 10;
        }
    }

    n_updates_since_last_save_ = 0;
    seq_number_ = 0;
    // Print config
    ROS_INFO("-----VoxelHashMapNode configuration-----");
    ROS_INFO("variant name: %s", p_variant_name_.c_str());
    ROS_INFO("dl_method: %s", dl_method.c_str());
    ROS_INFO("fusion_method: %s", fusion_method.c_str());
    ROS_INFO("uncertainty_method: %s", uncertainty_method.c_str());
    ROS_INFO("beta: %f", p_beta_);
    ROS_INFO("map_frame: %s", map_frame_.c_str());

    // Subscribers
    sub_pcd_ =
        nh_.subscribe("point_cloud", 1, &VoxelHashMapNode::pcdCallback, this);

    // Publishers
    if (p_visualize_semantics_) {
        pub_voxel_markers_ =
            nh_.advertise<visualization_msgs::MarkerArray>("voxel_markers", 1);
        pub_background_markers_ =
            nh_.advertise<visualization_msgs::MarkerArray>("background_markers",
                                                           1);

        // Timer to publish voxel markers
        vis_timer_ =
            nh_.createTimer(ros::Duration(1.0 / p_vis_sem_freq_),
                            &VoxelHashMapNode::publishVoxelMarkers, this);

        // Send one delete message to clear the markers
        visualization_msgs::MarkerArray delete_msg;
        visualization_msgs::Marker delete_marker;
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        delete_msg.markers.push_back(delete_marker);

        pub_voxel_markers_.publish(delete_msg);
    }

    // Services
    srv_save_map_request_ = nh_.advertiseService(
        "save_map_request", &VoxelHashMapNode::saveMapRequestSrvCallback, this);
    // The difference between these two is that the first one does not specify
    // the path and instead requests the saveVoxelMap() function to be called.
    srv_save_map_ = nh_.advertiseService(
        "save_map", &VoxelHashMapNode::saveMapSrvCallback, this);
    srv_open_map_ = nh_.advertiseService(
        "open_map", &VoxelHashMapNode::openMapSrvCallback, this);
    srv_evaluate_map_ = nh_.advertiseService(
        "evaluate_map", &VoxelHashMapNode::evaluateMapSrvCallback, this);

    // Generate save paths
    if (p_save_experiment_) {
        // Get timestamp in the correct format
        std::time_t time = std::time({});
        char timeString[std::size("yyyy-mm-dd_hh-mm-ss")];
        std::strftime(std::data(timeString), std::size(timeString),
                      "%F_%H-%M-%S", std::gmtime(&time));
        // The save path will be
        // *p_save_path*/dl_method_fusion_method_uncertainty_method_beta/*timestamp*/
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << p_beta_;
        std::string beta_str = stream.str();

        if (p_variant_name_ == "") {
            p_save_path_ = p_save_directory_path_ + "/" + p_experiment_map_name_ +
                        "/" + dl_method + "_" +
                        fusion_method + "_" + uncertainty_method + "_" +
                        beta_str + "/" + std::string(timeString) + "/";
        } else {
            p_save_path_ = p_save_directory_path_ + "/" + p_experiment_map_name_ +
                        "/" + p_variant_name_ + "_" + dl_method + "_" +
                        fusion_method + "_" + uncertainty_method + "_" +
                        beta_str + "/" + std::string(timeString) + "/";
        }

        ROS_INFO("Save path: %s", p_save_path_.c_str());
        // Try to create the directory if it doesn't exist
        if (!fs::create_directories(p_save_path_)) {
            ROS_WARN("Could not create directory %s", p_save_path_.c_str());
        }
        // Save the configuration used in the experiment path
        std::ofstream config_file(p_save_path_ + "config.txt");
        config_file << "variant_name: " << p_variant_name_ << std::endl;
        config_file << "dl_method: " << dl_method << std::endl;
        config_file << "fusion_method: " << fusion_method << std::endl;
        config_file << "uncertainty_method: " << uncertainty_method
                    << std::endl;
        config_file << "beta: " << p_beta_ << std::endl;
        config_file << "saved_each_n_updates: " << p_save_each_n_updates_
                    << std::endl;
        config_file << "depth_threshold: " << p_depth_threshold_ << std::endl;
        config_file << "n_classes: " << p_n_classes_ << std::endl;
        config_file << "resolution: " << p_resolution_ << std::endl;
        if (p_dl_method_ == DlMethod::MCD) {
            config_file << "n_samples_mc: " << p_n_samples_mc_ << std::endl;
        }
        config_file.close();
    }

    if (p_save_experiment_) {
        ROS_INFO("Save path: %s", p_save_path_.c_str());
    }
}

void VoxelHashMapNode::pcdCallback(
    sensor_msgs::PointCloud2ConstPtr const &msg) {
    auto start = std::chrono::high_resolution_clock::now();
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
    sensor_msgs::PointCloud2ConstIterator<uint32_t> iter_rgba(*msg, "rgb");
    sensor_msgs::PointCloud2ConstIterator<float> iter_gt_class(*msg,
                                                               "gt_class");
    sensor_msgs::PointCloud2ConstIterator<float> iter_class_prob(*msg, "prob");

    // Try to get transform from point cloud frame to map frame or return
    Eigen::Affine3f T_Map_Cam;
    tf::StampedTransform transform;

    if (!tryGetTransform(msg->header.frame_id, map_frame_, msg->header.stamp,
                         T_Map_Cam, transform)) {
        ROS_ERROR("Could not get transform from map to camera in pcd callback");
        return;
    }
    last_update_time_ = msg->header.stamp;
    Eigen::Vector3f cam_pos = T_Map_Cam.translation();

    // ROS_INFO("Received point cloud with %d points", msg->width *
    // msg->height); Loop all iterators
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_rgba,
                                   ++iter_gt_class, ++iter_class_prob) {
        Eigen::Vector3f point(*iter_x, *iter_y, *iter_z);
        // Only integrate points in a certain depth range
        if (point(0) > p_depth_threshold_ || point(0) < 0.001f) {
            continue;
        }

        // Transform point to map frame
        Eigen::Vector3f point_map = T_Map_Cam * point;
        point(0) = point_map.x();
        point(1) = point_map.y();
        point(2) = point_map.z();

        // Get color information
        uint32_t rgba = *iter_rgba;
        uint8_t r = (rgba >> 16) & 0x0000ff;
        uint8_t g = (rgba >> 8) & 0x0000ff;
        uint8_t b = rgba & 0x0000ff;
        uint8_t a = (rgba >> 24) & 0x0000ff;

        // Get ground truth class
        uint32_t gt_class = uint32_t(*iter_gt_class);

        // Get class probabilities
        VoxelInfo processed_measurement;

        // Manage fusion coming from sensor

        // Sampling based DL method
        if (p_dl_method_ == DlMethod::MCD) {
            obtainMCClassProb(iter_class_prob, processed_measurement);
            processed_measurement.addGtObservation(gt_class);

        } else if (p_dl_method_ == DlMethod::DET) {
            obtainDetClassProb(iter_class_prob, processed_measurement);
            processed_measurement.addGtObservation(gt_class);

        } else {
            ROS_ERROR("Unknown DL method");
        }
        // ROS_ERROR("Class probs");
        // for (int i = 0; i < p_n_classes_; i++) {
        //     ROS_ERROR("%f", processed_measurement.getClassProbability(i));
        // }
        voxel_hash_map_->integrateVoxel(point, processed_measurement);
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    // ROS_INFO("Time to fuse map: %f", elapsed.count());
    // ROS_INFO("Map size: %ld", voxel_hash_map_->size());
    did_voxel_map_update_ = true;
    // Save map for evaluation
    if (p_save_experiment_ && voxel_hash_map_->size() > 0) {
        n_updates_since_last_save_++;
        seq_number_++;
        if (n_updates_since_last_save_ >= p_save_each_n_updates_) {
            bool did_save = saveVoxelMap();
            n_updates_since_last_save_ = 0;
            if (!did_save) {
                ROS_ERROR("Could not save voxel map");
            }
        }
        // ROS_INFO("Status: %d", seq_number_);
    }
}

void VoxelHashMapNode::obtainMCClassProb(
    const sensor_msgs::PointCloud2ConstIterator<float> &iter_class_prob,
    VoxelInfo &processed_measurement) {
    // The sample fusion is at pixel level so we can fuse when we process all
    // samples from a pixel The structure is as follows: class_1 of sample_1,
    // ..., class_n of sample_1, class_1 of sample_2, ..., class_n of sample_2,
    // ...
    std::vector<VoxelInfo> class_prob_mc(p_n_samples_mc_);
    for (int sample = 0; sample < p_n_samples_mc_; sample++) {
        Eigen::ArrayXf class_prob(p_n_classes_, 1);
        for (int n_class = 0; n_class < p_n_classes_; n_class++) {
            class_prob(n_class, 0) =
                iter_class_prob[sample * p_n_classes_ + n_class];
        }
        VoxelInfo voxel_info(class_prob);
        // We join all the samples for one pixel
        class_prob_mc[sample] = voxel_info;
    }
    // Fuse the samples with MC for the same pixel/point
    processed_measurement = voxel_integrator_->fuseVoxelsMC(class_prob_mc);
}

void VoxelHashMapNode::obtainDetClassProb(
    const sensor_msgs::PointCloud2ConstIterator<float> &iter_class_prob,
    VoxelInfo &processed_measurement) {
    Eigen::ArrayXf class_prob(p_n_classes_, 1);
    for (int n_class = 0; n_class < p_n_classes_; n_class++) {
        class_prob(n_class, 0) = iter_class_prob[n_class];
    }

    VoxelInfo voxel_info(class_prob);
    processed_measurement = voxel_info;
}

inline bool VoxelHashMapNode::tryGetTransform(const std::string &frame_id,
                                              const std::string &origin,
                                              const ros::Time &stamp,
                                              Eigen::Affine3f &T_Map_Cam,
                                              tf::StampedTransform &transform) {
    try {
        tf_listener_.lookupTransform(origin, frame_id, stamp, transform);
    } catch (tf::TransformException ex) {
        ROS_ERROR("%s", ex.what());
        return false;
    }
    Eigen::Affine3d dT_Map_Cam;
    tf::transformTFToEigen(transform, dT_Map_Cam);
    T_Map_Cam = dT_Map_Cam.cast<float>();
    return true;
}

void VoxelHashMapNode::publishVoxelMarkers(const ros::TimerEvent &event) {
    if (!p_visualize_semantics_ || voxel_hash_map_->size() == 0 ||
        !did_voxel_map_update_) {
        return;
    }
    did_voxel_map_update_ = false;

    // Configure markers
    visualization_msgs::MarkerArray marker_array, marker_background_array;
    visualization_msgs::Marker marker_probabilities, marker_background,
        marker_uncertainties, marker_entropies, marker_gt;

    marker_probabilities.header.frame_id = marker_background.header.frame_id =
        marker_uncertainties.header.frame_id =
            marker_entropies.header.frame_id = marker_gt.header.frame_id =
                map_frame_;

    marker_probabilities.header.stamp = marker_background.header.stamp =
        marker_uncertainties.header.stamp = marker_entropies.header.stamp =
            marker_gt.header.stamp = ros::Time::now();

    marker_probabilities.id = 0;
    marker_background.id = 0;
    marker_uncertainties.id = 1;
    marker_entropies.id = 2;
    marker_gt.id = 3;

    marker_probabilities.type = marker_background.type =
        marker_uncertainties.type = marker_entropies.type = marker_gt.type =
            visualization_msgs::Marker::CUBE_LIST;

    marker_probabilities.action = marker_background.action =
        marker_uncertainties.action = marker_entropies.action =
            marker_gt.action = visualization_msgs::Marker::MODIFY;

    marker_probabilities.scale.x = marker_probabilities.scale.y =
        marker_probabilities.scale.z = marker_background.scale.x =
            marker_background.scale.y = marker_background.scale.z =
                marker_uncertainties.scale.x = marker_uncertainties.scale.y =
                    marker_uncertainties.scale.z = marker_entropies.scale.x =
                        marker_entropies.scale.y = marker_entropies.scale.z =
                            marker_gt.scale.x = marker_gt.scale.y =
                                marker_gt.scale.z =
                                    voxel_hash_map_->getVoxelSize();

    marker_probabilities.color.a = marker_background.color.a =
        marker_uncertainties.color.a = marker_entropies.color.a =
            marker_gt.color.a = 1.0;

    marker_probabilities.pose.orientation.w =
        marker_background.pose.orientation.w =
            marker_uncertainties.pose.orientation.w =
                marker_entropies.pose.orientation.w =
                    marker_gt.pose.orientation.w = 1.0;

    marker_probabilities.ns = "probabilities";
    marker_background.ns = "background";
    marker_uncertainties.ns = "uncertainties";
    marker_entropies.ns = "entropies";
    marker_gt.ns = "ground_truth";

    auto hash_map = voxel_hash_map_->getVoxelHashMapData();

    std::vector<Eigen::Vector3f> voxel_centers;
    for (auto it = hash_map.begin(); it != hash_map.end(); it++) {
        geometry_msgs::Point point;
        Eigen::Vector3f v_point = voxel_hash_map_->keyToPoint(it->first);
        point.x = v_point(0);
        point.y = v_point(1);
        point.z = v_point(2);
        voxel_centers.push_back(v_point);
    }

    std::vector<float> voxel_uncertanties;
    float max_uncertainty = 0.0;
    for (auto it = hash_map.begin(); it != hash_map.end(); it++) {
        float voxel_uncertainty = it->second.getTraceUncertainties();
        if (voxel_uncertainty > max_uncertainty) {
            max_uncertainty = voxel_uncertainty;
        }
    }
    float max_entropy = -log(1.0 / (float)p_n_classes_);

    for (auto it = hash_map.begin(); it != hash_map.end(); it++) {
        // ROS_INFO("Voxel prob: %f", voxel_prob[i]);
        geometry_msgs::Point point;
        Eigen::Vector3f v_point = voxel_hash_map_->keyToPoint(it->first);
        point.x = v_point(0);
        point.y = v_point(1);
        point.z = v_point(2);

        if (point.z > p_vis_top_height_)  // Cut off the ceiling of the room for
                                          // better visualization
        {
            continue;
        }
        std_msgs::ColorRGBA color;
        std::vector<uint8_t> rgb;
        int voxel_class;
        float class_probability;
        it->second.getMostProbClassAndProb(voxel_class, class_probability);
        // Only display class if prob is above threshold
        if (class_probability > 0) {
            // Class marker
            rgb = label_to_rgb_[voxel_class];
            color.r = rgb[0] / 255.0;
            color.g = rgb[1] / 255.0;
            color.b = rgb[2] / 255.0;
            color.a = 1.0;
            if (voxel_class == 0) {
                marker_background.points.push_back(point);
                marker_background.colors.push_back(color);
            } else {
                marker_probabilities.points.push_back(point);
                marker_probabilities.colors.push_back(color);
            }
        }

        // Gt marker. Only display if class is not unknown
        int gt_class = it->second.getGtClass();
        if (gt_class != 0) {
            marker_gt.points.push_back(point);
            rgb = label_to_rgb_[gt_class];
            color.r = rgb[0] / 255.0;
            color.g = rgb[1] / 255.0;
            color.b = rgb[2] / 255.0;
            color.a = 1.0;

            marker_gt.colors.push_back(color);
        }

        // Include epistemic uncertainty
        color.r = it->second.getTraceUncertainties() / max_uncertainty;
        color.g = it->second.getTraceUncertainties() / max_uncertainty;
        color.b = it->second.getTraceUncertainties() / max_uncertainty;
        color.a = 1.0;
        marker_uncertainties.points.push_back(point);
        marker_uncertainties.colors.push_back(color);

        // Include aleatoric uncertainty
        color.r = it->second.computeEntropy() / max_entropy;
        color.g = it->second.computeEntropy() / max_entropy;
        color.b = it->second.computeEntropy() / max_entropy;
        color.a = 1.0;
        marker_entropies.points.push_back(point);
        marker_entropies.colors.push_back(color);
    }

    if (marker_probabilities.points.size() > 0) {
        marker_array.markers.push_back(marker_probabilities);
    }
    if (marker_background.points.size() > 0) {
        marker_background_array.markers.push_back(marker_background);
    }
    if (marker_uncertainties.points.size() > 0) {
        marker_array.markers.push_back(marker_uncertainties);
    }
    if (marker_entropies.points.size() > 0) {
        marker_array.markers.push_back(marker_entropies);
    }
    if (marker_gt.points.size() > 0) {
        marker_array.markers.push_back(marker_gt);
    }

    pub_voxel_markers_.publish(marker_array);
    pub_background_markers_.publish(marker_background_array);
}

bool VoxelHashMapNode::saveVoxelMap() {
    // Add the seq_num to the file path
    std::stringstream ss;
    ss << p_save_path_ << std::setw(6) << std::setfill('0') << seq_number_
       << ".semantic";
    voxel_hash_map_->serializeVoxelHashMap(ss.str());

    return true;
}

bool VoxelHashMapNode::saveMapRequestSrvCallback(
    std_srvs::Empty::Request &req, std_srvs::Empty::Response &res) {
    saveVoxelMap();
    return true;
}

bool VoxelHashMapNode::saveMapSrvCallback(
    semantic_mapping_ros::FilePath::Request &req,
    semantic_mapping_ros::FilePath::Response &res) {
    std::string file_path = req.file_path;
    ROS_INFO("Saving map to %s", file_path.c_str());
    voxel_hash_map_->serializeVoxelHashMap(file_path);
    return true;
}

bool VoxelHashMapNode::openMapSrvCallback(
    semantic_mapping_ros::FilePath::Request &req,
    semantic_mapping_ros::FilePath::Response &res) {
    std::string file_path = req.file_path;
    ROS_INFO("Loading map from %s", file_path.c_str());
    voxel_hash_map_->deserializeVoxelHashMap(file_path);
    std::cout << voxel_hash_map_ << std::endl;
    return true;
}

bool VoxelHashMapNode::evaluateMapSrvCallback(std_srvs::Empty::Request &req,
                                              std_srvs::Empty::Response &res) {
    ROS_INFO("Evaluating map");
    voxel_hash_map_->evaluateVoxelMap();
    return true;
}

}  // namespace semantic_mapping
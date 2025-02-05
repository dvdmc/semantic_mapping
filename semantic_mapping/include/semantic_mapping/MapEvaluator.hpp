#ifndef SEMANTIC_MAPPING_MAP_EVALUATOR_H_
#define SEMANTIC_MAPPING_MAP_EVALUATOR_H_

#include <nlohmann/json.hpp>

#include <core_tools/limits.h>
#include <semantic_mapping/VoxelHashMap.hpp>
#include <semantic_mapping/VoxelInfo.hpp>
#include <semantic_mapping/metrics.hpp>

using json = nlohmann::json;

namespace semantic_mapping {

class MapEvaluator {
   public:
    MapEvaluator(float _resolution, int _num_classes)
        : resolution(_resolution),
          num_classes(_num_classes),
          map(_resolution, _num_classes),
          gt_map(_resolution, _num_classes),
          is_volumetric_data(false),
          use_gt_map(false) {
        // std::vector<Eigen::Vector2f> corners{{-1.0, 5.0}, {15.0, 5.0}, {15.0,
        // -12}, {3.82, -12}, {3.82, -6.34}, {-1, -6.34}}; Eigen::Vector2f
        // heigh_limits{-1.0, 2.0}; map_volume =
        // std::make_unique<core_tools::Polygonal3DVolume>(corners,
        // heigh_limits);
    }
    ~MapEvaluator() {}

    SemanticVolumetricMapMetrics* getVolumetricMetricsPtr() {
        return static_cast<SemanticVolumetricMapMetrics*>(metrics);
    }

    SemanticVolumetricMapMetrics* getGtVolumetricMetricsPtr() {
        return static_cast<SemanticVolumetricMapMetrics*>(gt_metrics);
    }

    void setExperimentFolder(std::string folder_name) {
        experiment_folder = folder_name;
    }

    void setTotalVoxels(int _total_voxels) { total_voxels = _total_voxels; }

    void setTotalVoxelsFromGt() { total_voxels = gt_map.size(); }

    void setGtMap(std::string path) { gt_map_path = path; }

    void setMetrics(SemanticMapMetrics* _metrics) { metrics = _metrics; }

    void setGtMetrics(SemanticMapMetrics* _metrics) { gt_metrics = _metrics; }

    void openMap(std::string filename) {
        map.deserializeVoxelHashMap(filename);
        map_filename = filename;
        if (is_volumetric_data) {
            // The volumetric metrics are stored in a json file with the same
            // name as the map
            // ({map_name}.semantic) but with the extension .json and the prefix
            // volumetric_data_
            std::string id_map =
                map_filename.substr(map_filename.find_last_of("/") + 1);
            id_map = id_map.substr(0, id_map.find_last_of("."));
            std::string volumetric_filename =
                experiment_folder + "/volumetric_data_" + id_map + ".json";
            std::ifstream volumetric_file(volumetric_filename);
            volumetric_file >> volumetric_data;
        }
    }

    void openGtMap() {
        gt_map.deserializeVoxelHashMap(gt_map_path);
        if (is_volumetric_data) {
            // The volumetric metrics are stored in a json file with the same
            // name as the map
            // ({map_name}.semantic) but with the extension .json and the prefix
            // volumetric_data_
            std::string path_to_gt =
                gt_map_path.substr(0, gt_map_path.find_last_of("/"));
            std::string volumetric_filename =
                path_to_gt + "/volumetric_data_gt.json";
            std::ifstream volumetric_file(volumetric_filename);
            volumetric_file >> gt_volumetric_data;
        }
        use_gt_map = true;
    }

    void printConsistencyTest() {
        SemanticVolumetricMapMetrics* volumetric_metrics =
            getVolumetricMetricsPtr();
        SemanticVolumetricMapMetrics* gt_volumetric_metrics =
            getGtVolumetricMetricsPtr();
        int filtered_map_size = 0;
        auto map_data = map.getVoxelHashMapData();
        for (auto it = map_data.begin(); it != map_data.end(); ++it) {
            if (map_volume->contains(map.keyToPoint(it->first))) {
                filtered_map_size++;
            }
        }
        int filtered_gt_map_size = 0;
        auto gt_map_data = gt_map.getVoxelHashMapData();
        for (auto it = gt_map_data.begin(); it != gt_map_data.end(); ++it) {
            if (map_volume->contains(gt_map.keyToPoint(it->first))) {
                filtered_gt_map_size++;
            }
        }
        std::cout << "Hash map: - Obtained: " << map.size()
                  << " - Filtered: " << filtered_map_size
                  << " - GT: " << gt_map.size()
                  << " - Filtered: " << filtered_gt_map_size << std::endl
                  << "Volumetric map: - Occ/Free/Invalid: "
                  << volumetric_metrics->occupied << "/"
                  << volumetric_metrics->free << "/"
                  << volumetric_metrics->invalid << std::endl
                  << "Volumetric GT:  - Occ/Free/Invalid: "
                  << gt_volumetric_metrics->occupied << "/"
                  << gt_volumetric_metrics->free << "/"
                  << gt_volumetric_metrics->invalid << std::endl
                  << "Discrepancy:" << std::endl
                  << "- Map Voxels/Occ/Diff: " << filtered_map_size << "/"
                  << volumetric_metrics->occupied << "/"
                  << (int)filtered_map_size - volumetric_metrics->occupied
                  << std::endl
                  << "- GT Voxels/Occ/Diff: " << filtered_gt_map_size << "/"
                  << gt_volumetric_metrics->occupied << "/"
                  << (int)filtered_gt_map_size - gt_volumetric_metrics->occupied
                  << std::endl;
    }

    void evaluateMap() {
        metrics->num_classes = num_classes;
        metrics->resolution = resolution;
        if (use_gt_map) {
            gt_metrics->num_classes = num_classes;
            gt_metrics->resolution = resolution;
        }
        evaluateMapSemantics();
        if (is_volumetric_data) {
            addVolumetricMetrics();
        }
        evaluateMapEntropy();
        saveMetricsJson(map_filename);
    }

    void evaluateMapSemantics() {
        std::vector<std::vector<int>> confusion_matrix;
        std::vector<int> pred_labels;
        std::vector<int> true_labels;
        std::vector<std::vector<float>> confidences;
        // Compute values that depend only on the current map
        evaluateMapPredictions(confusion_matrix, pred_labels, true_labels,
                               confidences);

        if (is_volumetric_data) {
            SemanticVolumetricMapMetrics* volumetric_metrics =
                getVolumetricMetricsPtr();
            volumetric_metrics->pred_labels = pred_labels;
            volumetric_metrics->true_labels = true_labels;
            volumetric_metrics->confidences = confidences;
        } else {
            metrics->pred_labels = pred_labels;
            metrics->true_labels = true_labels;
            metrics->confidences = confidences;
        }
    }

    void evaluateMapPredictions(std::vector<std::vector<int>>& confusion_matrix,
                                std::vector<int>& pred_label,
                                std::vector<int>& true_label,
                                std::vector<std::vector<float>>& confidences) {
        // Get voxel hash map
        auto voxel_hash_map_data = map.getVoxelHashMapData();

        // Initialize structures with the size of the map and num classes
        confusion_matrix = std::vector<std::vector<int>>(
            map.size(), std::vector<int>(num_classes, 0));
        pred_label = std::vector<int>(map.size(), -1);
        true_label = std::vector<int>(map.size(), -1);
        confidences = std::vector<std::vector<float>>(
            map.size(), std::vector<float>(num_classes, 1.0f / num_classes));

        int idx = 0;
        for (auto it = voxel_hash_map_data.begin();
             it != voxel_hash_map_data.end(); it++) {
            if (map_volume != nullptr &&
                !map_volume->contains(map.keyToPoint(it->first))) {
                // Resize the vector removing an element at the end
                pred_label.pop_back();
                true_label.pop_back();
                confidences.pop_back();
                continue;
            }
            int class_id = it->second.getMostProbableClass();
            if (class_id == -1) {
                class_id = 0;
            }

            int gt_class_id;
            if (!use_gt_map) {
                it->second.getGtClass();
            } else {
                gt_class_id =
                    gt_map.getVoxelPtr(map.keyToPoint(it->first))->getGtClass();
            }

            confusion_matrix[idx][gt_class_id]++;

            pred_label[idx] = class_id;
            true_label[idx] = gt_class_id;
            for (int i = 0; i < num_classes; i++) {
                confidences[idx][i] = it->second.getClassProbability(i);
            }
            idx++;
        }
    }

    // TODO: Below is probably wrong due to mismatch between volumetric and
    // actual map
    void evaluateMapEntropy() {
        int map_voxels = 0;
        float entropy = 0;
        auto map_data = map.getVoxelHashMapData();
        for (auto it = map_data.begin(); it != map_data.end(); ++it) {
            if (map_volume != nullptr &&
                map_volume->contains(map.keyToPoint(it->first))) {
                map_voxels++;
                entropy += it->second.computeEntropy();
            } else {
                entropy += it->second.computeEntropy();
            }
        }

        int unknown_voxels;
        if(!use_gt_map)
        {
            // For known voxels, the entropy is the same as the map entropy
            // For unknown voxels, the entropy is the entropy of a uniform
            // distribution of C classes h_v = - sum_C (1/C) * log2(1/C) = log2(C).
            // For all unknown voxels, the entropy is V * h_v
            // Total entropy = map entropy + unknown entropy
            if (is_volumetric_data) {
                SemanticVolumetricMapMetrics* volumetric_metrics =
                    getVolumetricMetricsPtr();
                // Reduce also the entropy of free and unknown voxels
                unknown_voxels = total_voxels - (volumetric_metrics->occupied +
                                                volumetric_metrics->free +
                                                volumetric_metrics->invalid);
            } else {
                // This only works if we know the total SURFACE voxels
                // It does not account for volume!
                unknown_voxels = total_voxels - map_voxels;
            }
        } else {
            int gt_map_voxels = 0;
            auto gt_map_data = gt_map.getVoxelHashMapData();
            for (auto it = gt_map_data.begin(); it != gt_map_data.end(); ++it) {
                if (map_volume != nullptr &&
                    map_volume->contains(map.keyToPoint(it->first))) {
                    gt_map_voxels++;
                } else {
                    gt_map_voxels++;
                }
            }
            unknown_voxels = gt_map_voxels - map_voxels;
        }

        float unknown_entropy =
            -(unknown_voxels)*std::log(1.0 / metrics->num_classes);

        float map_entropy = unknown_entropy + entropy;
        if (is_volumetric_data) {
            SemanticVolumetricMapMetrics* volumetric_metrics =
                static_cast<SemanticVolumetricMapMetrics*>(metrics);
            volumetric_metrics->entropy = map_entropy;
        } else {
            metrics->entropy = map_entropy;
        }
    }

        void addVolumetricMetrics() {
        SemanticVolumetricMapMetrics* volumetric_metrics =
            getVolumetricMetricsPtr();
        volumetric_metrics->frontiers = volumetric_data["frontiers"];
        volumetric_metrics->occupied = volumetric_data["occupied"];
        volumetric_metrics->free = volumetric_data["free"];
        volumetric_metrics->invalid = volumetric_data["invalid"];

        if (!use_gt_map) {
            volumetric_metrics->coverage =
                (float)(volumetric_metrics->occupied +
                        volumetric_metrics->free +
                        volumetric_metrics->invalid) /
                (float)total_voxels;
        } else {
            SemanticVolumetricMapMetrics* gt_volumetric_metrics =
                getGtVolumetricMetricsPtr();
            gt_volumetric_metrics->frontiers = gt_volumetric_data["frontiers"];
            gt_volumetric_metrics->occupied = gt_volumetric_data["occupied"];
            gt_volumetric_metrics->free = gt_volumetric_data["free"];
            gt_volumetric_metrics->invalid = gt_volumetric_data["invalid"];
            // Below is wrong since invalid might be different. We only care
            // about occupied voxels actually volumetric_metrics->coverage =
            // (float)volumetric_voxels_count/volumetric_gt_voxels_count;

            // This is the most simple version that aggregates the map.
            // volumetric_metrics->coverage = (float)map.size() / gt_map.size();

            // TODO: The more precise version checks if the voxels in the gt
            // exist Get voxel hash map
            auto voxel_hash_map_data = map.getVoxelHashMapData();
            auto gt_voxel_hash_map_data = gt_map.getVoxelHashMapData();

            int gt_voxels_in_volume = 0;
            int covered_voxels = 0;

            for (auto item = gt_voxel_hash_map_data.begin();
                 item != gt_voxel_hash_map_data.end(); item++) {
                if (map_volume->contains(gt_map.keyToPoint(item->first))) {
                    gt_voxels_in_volume++;
                    if (voxel_hash_map_data.find(item->first) !=
                        voxel_hash_map_data.end()) {
                        covered_voxels++;
                    }
                }
            }

            volumetric_metrics->coverage =
                (float)covered_voxels / gt_voxels_in_volume;
        }
    }

    void saveMetricsJson(std::string filename) {
        // Save the confusion matrix in a text file for further analysis in the
        // map_filename with the same name but .txt instead of .semantic
        std::string metrics_filename = map_filename;
        metrics_filename.replace(metrics_filename.end() - 8,
                                 metrics_filename.end(), "json");
        std::ofstream metrics_file;
        metrics_file.open(metrics_filename);
        json j;
        if (is_volumetric_data) {
            j = *getVolumetricMetricsPtr();
        } else {
            j = *metrics;
        }
        metrics_file << j.dump(4) << std::endl;
        metrics_file.close();
    }

    // TODO: Change to pointers
    int num_classes;
    float resolution;
    VoxelHashMap map;
    bool use_gt_map;
    VoxelHashMap gt_map;
    json volumetric_data;
    json gt_volumetric_data;
    SemanticMapMetrics* metrics;
    SemanticMapMetrics* gt_metrics;
    std::string map_filename;
    std::string experiment_folder;
    std::string gt_map_path;
    int total_voxels;
    bool is_volumetric_data;
    std::unique_ptr<core_tools::Polygonal3DVolume> map_volume;
};

}  // namespace semantic_mapping
#endif  // SEMANTIC_MAPPING_MAP_EVALUATOR_H_
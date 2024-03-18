#ifndef SEMANTIC_MAPPING_MAP_EVALUATOR_H_
#define SEMANTIC_MAPPING_MAP_EVALUATOR_H_

#include <nlohmann/json.hpp>

#include <semantic_mapping/VoxelHashMap.hpp>
#include <semantic_mapping/VoxelInfo.hpp>
#include <semantic_mapping/metrics.hpp>

using json = nlohmann::json;

namespace semantic_mapping {

class MapEvaluator {
   public:
    MapEvaluator(float _resolution, int _num_classes)
        : map(_resolution, _num_classes), is_volumetric_data(false) {}
    ~MapEvaluator() {}

    SemanticVolumetricMapMetrics* getVolumetricMetricsPtr() {
        return static_cast<SemanticVolumetricMapMetrics*>(metrics);
    }

    void setExperimentFolder(std::string folder_name) {
        experiment_folder = folder_name;
    }

    void setTotalVoxels(int _total_voxels) { total_voxels = _total_voxels; }

    void setMetrics(SemanticMapMetrics *_metrics) { metrics = _metrics; }

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

            // Check consistency in the volumetric data: the number of surface
            // voxels should be the same as the number of occupied voxels
            int occupied_voxels = volumetric_data["occupied"];
            int hash_map_occupied = map.size();
            std::cout << "Occupied voxels: map = " << hash_map_occupied
                      << " volumetric = " << occupied_voxels
                      << " discrepancy = "
                      << hash_map_occupied - occupied_voxels << std::endl;
        }
    }

    void addVolumetricMetrics() {
        SemanticVolumetricMapMetrics* volumetric_metrics = getVolumetricMetricsPtr();
        volumetric_metrics->frontiers = volumetric_data["frontiers"];
        volumetric_metrics->occupied = volumetric_data["occupied"];
        volumetric_metrics->free = volumetric_data["free"];
        volumetric_metrics->invalid = volumetric_data["invalid"];
        int volumetric_voxels_count =volumetric_metrics->occupied +
                                     volumetric_metrics->free +
                                     volumetric_metrics->invalid;
        std::cout << "Volumetric metrics. Consistency test: " << volumetric_voxels_count << " <= " << total_voxels << "?" << std::endl;
        volumetric_metrics->coverage = (float)(volumetric_metrics->occupied +
                                              volumetric_metrics->free +
                                              volumetric_metrics->invalid) / (float)total_voxels;
    }

    void evaluateMap() {
        metrics->num_classes = map.getNumClasses();
        metrics->resolution = map.getResolution();
        map.setVerbose(true);
        evaluateMapSemantics();
        if (is_volumetric_data) {
            addVolumetricMetrics();
        }
        evaluateMapEntropy(total_voxels);
        saveMetricsJson(map_filename);
    }

    void saveMetricsJson(std::string filename) {
        // Save the confusion matrix in a text file for further analysis in the
        // map_filename with the same name but .txt instead of .semantic
        std::string metrics_filename = map_filename;
        metrics_filename.replace(metrics_filename.end() - 8,
                                 metrics_filename.end(), "txt");
        std::ofstream metrics_file;
        metrics_file.open(metrics_filename);
        json j;
        if(is_volumetric_data)
        {
            j = *getVolumetricMetricsPtr();
        } else {
            j = *metrics;
        }
        metrics_file << j.dump(4) << std::endl;
        metrics_file.close();
    }

    void evaluateMapSemantics() {
        std::vector<std::vector<int>> confusion_matrix;
        std::cout << "Start evaluating map semantics" << std::endl;
        // Compute values that depend only on the current map
        map.evaluateVoxelMapConfMatrix(confusion_matrix);
        std::cout << "End evaluating map semantics" << std::endl;
        if(is_volumetric_data)
        {
            // Add the volumetric metrics
            SemanticVolumetricMapMetrics* volumetric_metrics = getVolumetricMetricsPtr();
            volumetric_metrics->confusion_matrix = confusion_matrix;
        } else {
            metrics->confusion_matrix = confusion_matrix;
        }
    }

    void evaluateMapEntropy(int total_voxels) {
        std::cout << "Start evaluating map entropy" << std::endl;
        int map_voxels = map.size();
        float entropy = map.getVoxelMapEntropy();
        // For known voxels, the entropy is the same as the map entropy
        // For unknown voxels, the entropy is the entropy of a uniform
        // distribution of C classes h_v = - sum_C (1/C) * log2(1/C) = log2(C). 
        // For all unknown voxels, the entropy is V * h_v
        // Total entropy = map entropy + unknown entropy
        int unknown_voxels;
        if (is_volumetric_data) {
            SemanticVolumetricMapMetrics* volumetric_metrics = getVolumetricMetricsPtr();
            // Reduce also the entropy of free and unknown voxels
            unknown_voxels = total_voxels - (volumetric_metrics->occupied + volumetric_metrics->free + volumetric_metrics->invalid);
        } else 
        {
            unknown_voxels = total_voxels - map_voxels;
        }
        std::cout << "Total voxels: " << total_voxels << " map voxels: " << map_voxels << " unknown voxels: " << unknown_voxels << std::endl;
        float unknown_entropy = -(unknown_voxels)*std::log(1.0 / metrics->num_classes);
        std::cout << "Unknown entropy: " << unknown_entropy << std::endl;
        float map_entropy = unknown_entropy + entropy;
        if(is_volumetric_data)
        {
            SemanticVolumetricMapMetrics* volumetric_metrics = static_cast<SemanticVolumetricMapMetrics*>(metrics);
            volumetric_metrics->entropy = map_entropy;
        } else {
            metrics->entropy = map_entropy;
        }
        std::cout << "End evaluating map entropy" << std::endl;
    }

    VoxelHashMap map;
    json volumetric_data;
    SemanticMapMetrics* metrics;
    std::string map_filename;
    int total_voxels;
    std::string experiment_folder;
    bool is_volumetric_data;
};

}  // namespace semantic_mapping
#endif  // SEMANTIC_MAPPING_MAP_EVALUATOR_H_
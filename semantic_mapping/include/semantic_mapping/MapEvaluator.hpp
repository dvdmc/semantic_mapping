#ifndef SEMANTIC_MAPPING_MAP_EVALUATOR_H_
#define SEMANTIC_MAPPING_MAP_EVALUATOR_H_

#include <semantic_mapping/VoxelHashMap.hpp>
#include <semantic_mapping/VoxelInfo.hpp>

namespace semantic_mapping {

class MapEvaluator {
   public:
    MapEvaluator(float _resolution, int _num_classes) : map(_resolution, _num_classes) {}
    ~MapEvaluator() {}

    void setExperimentFolder(std::string folder_name) {
        experiment_folder = folder_name;
    }

    void openMap(std::string filename) {
        map.deserializeVoxelHashMap(filename);
        map_filename = filename;
    }

    void evaluateMap() {
        // std::unordered_map<uint64_t, VoxelInfo> gt_map_hash_map =
        // gt_map.getVoxelHashMap();
        std::unordered_map<uint64_t, VoxelInfo> map_hash_map =
            map.getVoxelHashMap();

        Eigen::MatrixXi confusion_matrix;

        std::cout << "Num classes: " << map.getNumClasses() << std::endl;
        // Compute values that depend only on the current map
        map.evaluateVoxelMap(confusion_matrix);

        // Save the confusion matrix in a text file for further analysis in the
        // map_filename with the same name but .txt instead of .semantic
        std::string confusion_matrix_filename = map_filename;
        confusion_matrix_filename.replace(confusion_matrix_filename.end() - 8,
                                          confusion_matrix_filename.end(),
                                          "txt");
        std::ofstream confusion_matrix_file;
        confusion_matrix_file.open(confusion_matrix_filename);
        confusion_matrix_file << confusion_matrix << std::endl;
        confusion_matrix_file.close();
    }

    VoxelHashMap map;

    std::string map_filename;
    std::string experiment_folder;
};

}  // namespace semantic_mapping
#endif  // SEMANTIC_MAPPING_MAP_EVALUATOR_H_
#include <dirent.h>
#include <boost/program_options.hpp>

#include <semantic_mapping/MapEvaluator.hpp>
#include <semantic_mapping/metrics.hpp>

namespace po = boost::program_options;

// Main
int main(int argc, char **argv) {
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")(
        "num_classes", po::value<int>()->required(),
        "the number of classes in the map required for deserialization")(
        "resolution", po::value<float>()->required(),
        "the resolution of the map required for deserialization")(
        "exp_path", po::value<std::string>()->required(),
        "the path to the experiment folder")(
        "volumetric", po::value<bool>()->default_value(false),
        "if the evaluation contains volumetric information")(
        "total_voxels", po::value<float>(),
        "an upper bound for the total number of voxels in the map")(
        "gt_map", po::value<std::string>(),
        "the path to the GT map containing the whole scene coverage")(
        "map_limits", po::value<std::string>(), "the path to the map limits");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    int num_classes;
    if (vm.count("num_classes")) {
        num_classes = vm["num_classes"].as<int>();
        // std::cout << "Number of classes was set to " << num_classes << ".\n";
    } else {
        // std::cout << "Number of classes was not set.\n";
        return 1;
    }

    float resolution;
    if (vm.count("resolution")) {
        resolution = vm["resolution"].as<float>();
        // std::cout << "Resolution was set to " << resolution << ".\n";
    } else {
        // std::cout << "Resolution was not set.\n";
        return 1;
    }

    std::string experiment_folder;
    if (vm.count("exp_path")) {
        experiment_folder = vm["exp_path"].as<std::string>();
        // std::cout << "Experiment path was set to " << experiment_folder
        //   << ".\n";
    } else {
        // std::cout << "Experiment path was not set.\n";
        return 1;
    }

    bool volumetric;
    if (vm.count("volumetric")) {
        volumetric = vm["volumetric"].as<bool>();
        // std::cout << "Volumetric was set to " << volumetric << ".\n";
    } else {
        // std::cout << "Volumetric was not set.\n";
        volumetric = false;
    }

    int total_voxels;
    if (volumetric) {
        if (vm.count("total_voxels")) {
            total_voxels = vm["total_voxels"].as<float>();
            // std::cout << "Total voxels was set to " << total_voxels << ".\n";
        } else {
            // std::cout << "Total voxels was not set.\n";
            total_voxels = -1;
        }
    }

    std::string gt_file_path;
    bool eval_with_gt = false;
    if (vm.count("gt_map")) {
        gt_file_path = vm["gt_map"].as<std::string>();
        // std::cout << "GT path was set to " << gt_file_path << ".\n";
        eval_with_gt = true;
    } else {
        // std::cout << "GT path path was not set.\n";
    }

    std::string map_limits_path;
    bool eval_with_map_limits = false;
    if (vm.count("map_limits")) {
        map_limits_path = vm["map_limits"].as<std::string>();
        // std::cout << "Map limits was set to " << map_limits_path << ".\n";
        eval_with_map_limits = true;
    } else {
        // std::cout << "Map limits path was not set.\n";
    }

    // Read the experiment folder
    semantic_mapping::MapEvaluator map_evaluator(resolution, num_classes);
    map_evaluator.setExperimentFolder(experiment_folder);
    map_evaluator.is_volumetric_data = volumetric;

    // Read the maps
    // Check if there is gt
    if (eval_with_gt) {
        std::cout << "Evaluating with GT" << std::endl;
        map_evaluator.setGtMap(gt_file_path);
        map_evaluator.openGtMap();
    }
    std::vector<std::string> map_filenames;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(experiment_folder.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename.find(".semantic") != std::string::npos) {
                std::cout << "Adding " << filename << std::endl;
                map_filenames.push_back(experiment_folder + "/" + filename);
            }
        }
        closedir(dir);
    } else {
        std::cout << "Could not open the experiment folder" << std::endl;
        return 0;
    }
    // Sort the maps
    std::sort(map_filenames.begin(), map_filenames.end());

    // Evaluate all maps
    for (int i = 0; i < map_filenames.size(); i++) {
        std::cout << "Evaluating map " << i << std::endl;
        std::unique_ptr<semantic_mapping::SemanticMapMetrics> metrics,
            gt_metrics;
        if (volumetric) {
            metrics =
                std::unique_ptr<semantic_mapping::SemanticVolumetricMapMetrics>(
                    new semantic_mapping::SemanticVolumetricMapMetrics());
            gt_metrics =
                std::unique_ptr<semantic_mapping::SemanticVolumetricMapMetrics>(
                    new semantic_mapping::SemanticVolumetricMapMetrics());
        } else {
            metrics = std::unique_ptr<semantic_mapping::SemanticMapMetrics>(
                new semantic_mapping::SemanticMapMetrics());
            gt_metrics = std::unique_ptr<semantic_mapping::SemanticMapMetrics>(
                new semantic_mapping::SemanticMapMetrics());
        }

        map_evaluator.openMap(map_filenames[i]);
        map_evaluator.setMetrics(metrics.get());
        if (eval_with_gt) {
            map_evaluator.setTotalVoxelsFromGt();
            map_evaluator.setGtMetrics(gt_metrics.get());
            map_evaluator.evaluateMap();

        } else {
            map_evaluator.setTotalVoxels(total_voxels);
            map_evaluator.evaluateMap();
        }
    }

    return 0;
}
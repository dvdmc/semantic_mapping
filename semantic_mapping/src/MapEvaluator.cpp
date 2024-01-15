#include <dirent.h>
#include <boost/program_options.hpp>

#include<semantic_mapping/MapEvaluator.hpp>

namespace po = boost::program_options;

// Main
int main(int argc, char** argv)
{
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("num_classes", po::value<int>(), "the number of classes in the map required for deserialization")
        ("resolution", po::value<double>(), "the resolution of the map required for deserialization")
        ("exp_path", po::value<std::string>(), "the path to the experiment folder")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    int num_classes;
    if (vm.count("num_classes")) {
        std::cout << "Number of classes was set to " 
                  << vm["num_classes"].as<int>() << ".\n";
        num_classes = vm["num_classes"].as<int>();
    } else {
        std::cout << "Number of classes was not set.\n";
        return 1;
    }

    float resolution;
    if (vm.count("resolution")) {
        std::cout << "Resolution was set to " 
                  << vm["resolution"].as<float>() << ".\n";
        resolution = vm["resolution"].as<float>();
    } else {
        std::cout << "Resolution was not set.\n";
        return 1;
    }

    if (vm.count("exp_path")) {
        std::cout << "Experiment path was set to " 
                  << vm["exp_path"].as<std::string>() << ".\n";
    } else {
        std::cout << "Experiment path was not set.\n";
        return 1;
    }

    // The evaluation work as follows: 
    // 1. Read the experiment folder
    // 3. There are maps captured in sequence, each one is called ######.semantic
    // 4. The maps in the sequence are evaluated saving the confusion matrix by the MapEvaluator class
    
    // Read the experiment folder
    std::string experiment_folder = argv[1];

    semantic_mapping::MapEvaluator map_evaluator(resolution, num_classes);
    if(resolution != map_evaluator.map.getResolution() || num_classes != map_evaluator.map.getNumClasses())
    {
        std::cout << "The map resolution or number of classes does not match the specified" << std::endl;
        return 0;
    }
    map_evaluator.setExperimentFolder(experiment_folder);

    // Read the maps
    std::vector<std::string> map_filenames;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (experiment_folder.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename.find(".semantic") != std::string::npos && filename != "gt.semantic")
            {
                map_filenames.push_back(experiment_folder + "/" + filename);
            }
        }
        closedir (dir);
    } else {
        std::cout << "Could not open the experiment folder" << std::endl;
        return 0;
    }
    // Sort the maps
    std::sort(map_filenames.begin(), map_filenames.end());

    // Evaluate all maps
    for (int i = 0; i < map_filenames.size(); i++)
    {
        std::cout << "Evaluating map " << i << std::endl;
        map_evaluator.openMap(map_filenames[i]);
        map_evaluator.evaluateMap();
    }
    

    return 0;
}
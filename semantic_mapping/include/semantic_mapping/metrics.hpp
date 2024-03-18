/***********************************************************
 *
 * @file: metrics.hpp
 * @breif: Metrics used in the map evaluator
 * @author: David Morilla-Cabello
 * @update: TODO
 * @version: 1.0
 *
 * Copyright (c) 2023， David Morilla-Cabello
 * All rights reserved.
 * --------------------------------------------------------
 *
 **********************************************************/
#ifndef SEMANTIC_MAPPING_METRICS_H_
#define SEMANTIC_MAPPING_METRICS_H_

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace semantic_mapping {

/**
 * @brief Metrics used in the map evaluator
 */
struct SemanticMapMetrics
{
    int num_classes;
    float resolution;
    std::vector<std::vector<int>> confusion_matrix;
    float entropy;

    virtual ~SemanticMapMetrics() = default;
};

void to_json(json& j, const SemanticMapMetrics& p){
    j = json{{"num_classes", p.num_classes},
             {"resolution", p.resolution},
             {"confusion_matrix", p.confusion_matrix},
             {"entropy", p.entropy}};
}

void from_json(const json& j, SemanticMapMetrics& p){
    j.at("num_classes").get_to(p.num_classes);
    j.at("resolution").get_to(p.resolution);
    j.at("confusion_matrix").get_to(p.confusion_matrix);
    j.at("entropy").get_to(p.entropy);
}

/**
 * @brief Metrics used in the map evaluator
 */
struct SemanticVolumetricMapMetrics : public SemanticMapMetrics
{
    int num_classes;
    float resolution;
    std::vector<std::vector<int>> confusion_matrix;
    float entropy;
    int frontiers;
    int occupied;
    int free;
    int invalid;
    float coverage;

    virtual ~SemanticVolumetricMapMetrics() = default;
};

void to_json(json& j, const SemanticVolumetricMapMetrics& p){
    j = json{{"num_classes", p.num_classes},
             {"resolution", p.resolution},
             {"confusion_matrix", p.confusion_matrix},
             {"entropy", p.entropy},
             {"frontiers", p.frontiers},
             {"occupied", p.occupied},
             {"free", p.free},
             {"invalid", p.invalid},
             {"coverage", p.coverage}};
}

void from_json(const json& j, SemanticVolumetricMapMetrics& p){
    j.at("num_classes").get_to(p.num_classes);
    j.at("resolution").get_to(p.resolution);
    j.at("confusion_matrix").get_to(p.confusion_matrix);
    j.at("entropy").get_to(p.entropy);
    j.at("frontiers").get_to(p.frontiers);
    j.at("occupied").get_to(p.occupied);
    j.at("free").get_to(p.free);
    j.at("invalid").get_to(p.invalid);
    j.at("coverage").get_to(p.coverage);
}

}  // namespace semantic_mapping

#endif  // SEMANTIC_MAPPING_METRICS_H_

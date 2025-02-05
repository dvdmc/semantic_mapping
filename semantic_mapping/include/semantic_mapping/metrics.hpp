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
    float entropy;
    std::vector<std::vector<float>> confusion_matrix;
    std::vector<std::vector<float>> confidences;
    std::vector<int> pred_labels;
    std::vector<int> true_labels;

    virtual ~SemanticMapMetrics() = default;
};

void to_json(json& j, const SemanticMapMetrics& p){
    j = json{{"num_classes", p.num_classes},
             {"resolution", p.resolution},
             {"pred_labels", p.pred_labels},
             {"entropy", p.entropy},
             {"confusion_matrix", p.confusion_matrix},
             {"confidences", p.confidences},
             {"true_labels", p.true_labels}};
}

void from_json(const json& j, SemanticMapMetrics& p){
    j.at("num_classes").get_to(p.num_classes);
    j.at("resolution").get_to(p.resolution);
    j.at("pred_labels").get_to(p.pred_labels);
    j.at("entropy").get_to(p.entropy);
    j.at("confusion_matrix").get_to(p.confusion_matrix);
    j.at("confidences").get_to(p.confidences);
    j.at("true_labels").get_to(p.true_labels);
}

/**
 * @brief Metrics used in the map evaluator
 */
struct SemanticVolumetricMapMetrics : public SemanticMapMetrics
{
    int num_classes;
    float resolution;
    float entropy;
    int frontiers;
    int occupied;
    int free;
    int invalid;
    float coverage;
    std::vector<std::vector<float>> confusion_matrix;
    std::vector<std::vector<float>> confidences;
    std::vector<int> pred_labels;
    std::vector<int> true_labels;

    virtual ~SemanticVolumetricMapMetrics() = default;
};

void to_json(json& j, const SemanticVolumetricMapMetrics& p){
    j = json{{"num_classes", p.num_classes},
             {"resolution", p.resolution},
             {"entropy", p.entropy},
             {"frontiers", p.frontiers},
             {"occupied", p.occupied},
             {"free", p.free},
             {"invalid", p.invalid},
             {"coverage", p.coverage},
             {"confusion_matrix", p.confusion_matrix},
             {"confidences", p.confidences},
             {"pred_labels", p.pred_labels},
             {"true_labels", p.true_labels}};
}

void from_json(const json& j, SemanticVolumetricMapMetrics& p){
    j.at("num_classes").get_to(p.num_classes);
    j.at("resolution").get_to(p.resolution);
    j.at("entropy").get_to(p.entropy);
    j.at("frontiers").get_to(p.frontiers);
    j.at("occupied").get_to(p.occupied);
    j.at("free").get_to(p.free);
    j.at("invalid").get_to(p.invalid);
    j.at("coverage").get_to(p.coverage);
    j.at("confusion_matrix").get_to(p.confusion_matrix);
    j.at("confidences").get_to(p.confidences);
    j.at("pred_labels").get_to(p.pred_labels);
    j.at("true_labels").get_to(p.true_labels);
}

}  // namespace semantic_mapping

#endif  // SEMANTIC_MAPPING_METRICS_H_

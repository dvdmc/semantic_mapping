#ifndef SEMANTIC_MAPPING_VOXEL_INFO_H_
#define SEMANTIC_MAPPING_VOXEL_INFO_H_

#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include <cereal/archives/binary.hpp>
#include <cereal/types/eigen.hpp>
#include <cereal/types/vector.hpp>

namespace semantic_mapping {

#ifndef INITIAL_UNCERTAINTY
#define INITIAL_UNCERTAINTY 0.001f
#endif

class VoxelInfo {
   public:
    // Default constructor, needed for
    // the unordered_map and serialization
    // In the future we should try to use a template?
    VoxelInfo() {}

    // Basic constructor with only the number of classes
    VoxelInfo(int n_classes)
        : probabilities(Eigen::ArrayXf::Ones(n_classes, 1) / n_classes),
          samples_count(0),
          gt_class_count(Eigen::ArrayXi::Zero(n_classes, 1)),
          uncertainties(Eigen::MatrixXf::Ones(n_classes, 1) *
                        INITIAL_UNCERTAINTY) {}

    // Constructor with the class probabilities
    VoxelInfo(Eigen::ArrayXf _class_ps)
        : probabilities(_class_ps),
          samples_count(0),
          gt_class_count(Eigen::ArrayXi::Zero(_class_ps.rows(), 1)),
          uncertainties(Eigen::MatrixXf::Ones(_class_ps.rows(), 1) *
                        INITIAL_UNCERTAINTY) {}

    // Constructor with all the data
    VoxelInfo(Eigen::ArrayXf _class_ps, Eigen::MatrixXf _uncertainties,
              int _samples_count, Eigen::ArrayXi _gt_class_count)
        : probabilities(_class_ps),
          uncertainties(_uncertainties),
          samples_count(_samples_count),
          gt_class_count(_gt_class_count) {}

   private:
    // Class members
    Eigen::ArrayXf probabilities;  // Class probability distribution. This may
                                   // be probabilities OR log probabilities.
                                   // CURRENTLY: probabilities
    Eigen::MatrixXf
        uncertainties;  // Class uncertainty might be a covariance or other
                        // kind. It is left as a matrix to allow covariances.
                        // CURRENTLY: covariance diagonal
    int samples_count;  // This value is only incremented in MC and maintained
                        // in other fusions
    Eigen::ArrayXi
        gt_class_count;  // This vector is incremented with the addGtObservation
                         // method and maintained in other fusions

   public:
    // Setters
    void setNumClasses(const int &n_classes) {
        probabilities = Eigen::ArrayXf::Ones(n_classes, 1) / n_classes;
        gt_class_count = Eigen::ArrayXi::Zero(n_classes, 1);
        uncertainties =
            Eigen::MatrixXf::Ones(n_classes, 1) * INITIAL_UNCERTAINTY;
    }

    // These setters don't check consistency of the n_classes, TODO: make more
    // robust
    void setProbabilities(const Eigen::ArrayXf &_probabilities) {
        assert(
            probabilities.size() ==
            _probabilities.size());  // ERROR: Class probabilities size mismatch
        probabilities = _probabilities;
    }
    void setUncertainties(const Eigen::MatrixXf &_uncertainties) {
        assert(
            uncertainties.size() ==
            _uncertainties.size());  // ERROR: Class uncertainties size mismatch
        uncertainties = _uncertainties;
    }
    void setSamplesCount(const int &_samples_count) {
        samples_count = _samples_count;
    }
    void setGtClassCount(const Eigen::ArrayXi &_gt_class_count) {
        assert(gt_class_count.size() ==
               _gt_class_count
                   .size());  // ERROR: Class gt_class_count size mismatch
        gt_class_count = _gt_class_count;
    }
    void addGtObservation(const int &gt_class) {
        gt_class_count(gt_class, 0) += 1;
    }

    // Getters
    int getNumClasses() const { return probabilities.rows(); }
    const Eigen::ArrayXf &getProbabilities() const { return probabilities; }
    const Eigen::MatrixXf &getUncertainties() const { return uncertainties; }
    const Eigen::ArrayXi &getGtClassCount() const { return gt_class_count; }
    int getSamplesCount() const { return samples_count; }

    float getClassProbability(const int &class_id) const {
        return probabilities(class_id, 0);
    }
    float getClassUncertainty(const int &class_id) const {
        return uncertainties(class_id, 0);
    }
    float getTraceUncertainties() const { return uncertainties.sum(); }

    void getMostProbClassAndProb(int &class_id, float &prob) {
        int max_class = 0;
        float max_prob = 0;
        for (int i = 0; i < probabilities.size(); i++) {
            if (probabilities(i, 0) > max_prob) {
                max_prob = probabilities(i, 0);
                max_class = i;
            }
        }
        class_id = max_class;
        prob = max_prob;
    }
    int getMostProbableClass() const {
        int max_class = 0;
        float max_prob = 0;
        for (int i = 0; i < probabilities.size(); i++) {
            if (probabilities(i, 0) > max_prob) {
                max_prob = probabilities(i, 0);
                max_class = i;
            }
        }
        return max_class;
    }
    int getGtClass() const {
        int max_class = 0;
        int max_count = 0;
        for (int i = 0; i < gt_class_count.size(); i++) {
            if (gt_class_count(i, 0) > max_count) {
                max_count = gt_class_count(i, 0);
                max_class = i;
            }
        }
        return max_class;
    }

    float computeEntropy() {
        float entropy = 0.0f;
        for (int i = 0; i < probabilities.rows(); i++) {
            if (probabilities(i) > 0.0f)
                entropy -= probabilities(i) * std::log(probabilities(i));
        }
        return entropy;
    }

    // Operators
    friend std::ostream &operator<<(std::ostream &os, const VoxelInfo &voxel) {
        os << "VoxelInfo: " << std::endl;
        os << "Probabilities: " << voxel.probabilities.transpose() << std::endl;
        os << "Uncertainties: " << voxel.uncertainties.transpose() << std::endl;
        os << "GT Class count: " << voxel.gt_class_count.transpose()
           << std::endl;
        os << "Samples count: " << voxel.samples_count << std::endl;
        return os;
    }

    // Serialization
    template <class Archive>
    void serialize(Archive &archive) {
        archive(probabilities, uncertainties, gt_class_count, samples_count);
    }
};

}  // namespace semantic_mapping

#endif  // SEMANTIC_MAPPING_VOXEL_INFO_H
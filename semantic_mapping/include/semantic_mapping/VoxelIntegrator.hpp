#ifndef SEMANTIC_MAPPING_VOXEL_INTEGRATOR_H_
#define SEMANTIC_MAPPING_VOXEL_INTEGRATOR_H_

#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include <semantic_mapping/VoxelInfo.hpp>

/// @brief Voxel integrator
namespace semantic_mapping {
enum class FusionMethod {
    KF,
    BAY,
    W_BAY,
    SUM,
    SUM_PROBS,
    GAUSSIAN,
    W_DIRCHLET,
    DIRICHLET
};

enum class UncertaintyType {
    UNCERTAINTY,
    CONFIDENCE,
};

class VoxelIntegrator {
   private:
    int num_classes_;
    FusionMethod fusion_method_;
    UncertaintyType uncertainty_type_;
    float beta_;
    Eigen::ArrayXf class_weights_;
    std::function<void(const VoxelInfo &, const VoxelInfo &, VoxelInfo &)>
        fusion_function_;

   public:
    UncertaintyType getUncertaintyType() const { return uncertainty_type_; }
    
    VoxelIntegrator(int num_classes, FusionMethod fusion_method,
                    UncertaintyType uncertainty_type, float beta)
        : num_classes_(num_classes),
          fusion_method_(fusion_method),
          uncertainty_type_(uncertainty_type),
          beta_(beta) {
        class_weights_ = Eigen::ArrayXf::Ones(num_classes_, 1);
        // Set beta per class if different than 0.0
        // Entropy: {0.0478, 0.0874, 0.1845, 0.1289, 0.1519, 0.1659, 0.0806, 0.1468, 0.0922, 0.2434, 0.1832, 0.2033, 0.1433, 0.1083, 0.1926, 0.1457, 0.1807, 0.1739, 0.2346, 0.0894, 0.1287}
        if (beta_ == 1.0) {
            Eigen::ArrayXf weights(num_classes_, 1);
            // if (num_classes_ == 20)
            // {
            //     weights << 0.0478, 0.0874, 0.1845, 0.1289, 0.1519, 0.1659, 0.0806, // Entropy
            //                 0.1468, 0.0922, 0.2434, 0.1832, 0.2033, 0.1433, 0.1083, 
            //                 0.1926, 0.1457, 0.1807, 0.1739, 0.2346, 0.0894, 0.1287;
            //     // weights << 0.1063, 0.5525, 0.4314, 0.4305, 0.5077, 0.4005, 0.2744, // ECE
            //     //             0.3664, 0.3246, 0.5643, 0.3654, 0.5190, 0.3370, 0.3909, 
            //     //             0.4434, 0.4497, 0.4953, 0.4543, 0.5643, 0.4091, 0.5187;
            // } else {
            //     weights = Eigen::ArrayXf::Ones(num_classes_, 1);
            // }
            class_weights_ = 1-weights.array();
        }
        if (fusion_method_ == FusionMethod::KF) {
            fusion_function_ = std::bind(
                &VoxelIntegrator::fusionKF, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3);
        } else if (fusion_method_ == FusionMethod::BAY ||
                   fusion_method_ == FusionMethod::GAUSSIAN) {
            fusion_function_ =
                std::bind(&VoxelIntegrator::fusionDeterministicBayesian, this,
                          std::placeholders::_1, std::placeholders::_2,
                          std::placeholders::_3);
        } else if (fusion_method_ == FusionMethod::W_BAY) {
            fusion_function_ =
                std::bind(&VoxelIntegrator::fusionWeightedBayesian, this,
                          std::placeholders::_1, std::placeholders::_2,
                          std::placeholders::_3);
        } else if (fusion_method_ == FusionMethod::SUM) {
            fusion_function_ = std::bind(
                &VoxelIntegrator::fusionSum, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3);
        } else if (fusion_method_ == FusionMethod::SUM_PROBS) {
            fusion_function_ =
                std::bind(&VoxelIntegrator::fusionSumProbabilities, this,
                          std::placeholders::_1, std::placeholders::_2,
                          std::placeholders::_3);

        } else {
            std::cout << "ERROR: Fusion type not supported" << std::endl;
        }
    }
    ~VoxelIntegrator(){};

    void fusionKF(const VoxelInfo &current, const VoxelInfo &input,
                  VoxelInfo &output) {
        // Fusion using KF element-wise. Assuming off diagonals 0!
        // this->probabilities x input.probabilities z
        Eigen::ArrayXf a_uncertainties =
            current.getUncertainties().array();  // P
        Eigen::ArrayXf a_input_uncertainties =
            input.getUncertainties().array();  // Q
        Eigen::ArrayXf K =
            a_uncertainties * (a_uncertainties + a_input_uncertainties)
                                  .inverse();  // K = P * (P + Q)^-1
        output.setProbabilities(
            current.getProbabilities() +
            K * (input.getProbabilities() -
                 current.getProbabilities()));  // x = x + K * (z - x)
        output.setUncertainties(
            ((1 - K) * a_uncertainties + K * a_input_uncertainties)
                .matrix());  // P = (I - K) * P + K * Q
    }

    void fusionDeterministicBayesian(const VoxelInfo &current,
                                     const VoxelInfo &input,
                                     VoxelInfo &output) {
        // Fusion bayesian multi-class wo weighting
        // std::cout << "Fusing BAY" << " with alpha " << alpha << std::endl;
        Eigen::ArrayXf uniform =
            Eigen::ArrayXf::Ones(current.getNumClasses(), 1) /
            current.getNumClasses();
        Eigen::ArrayXf input_probabilities = input.getProbabilities();
        if(beta_ == 1.0){
        input_probabilities =
            (1 - class_weights_.array()) * input.getProbabilities() + (class_weights_.array())*uniform;
        } else {
            input_probabilities =
                (1 - beta_) * input.getProbabilities() + (beta_)*uniform;
        }
        Eigen::ArrayXf product =
            (current.getProbabilities() * input_probabilities);
        output.setProbabilities((product / product.sum()).matrix());
    }

    void fusionWeightedBayesian(const VoxelInfo &current,
                                const VoxelInfo &input, VoxelInfo &output) {
        // Fusion weighted bayesian multi-class. Alphas are stored in uncertainty
        Eigen::ArrayXf current_alpha;
        Eigen::ArrayXf input_alpha;
        if (uncertainty_type_ == UncertaintyType::UNCERTAINTY) {
            current_alpha =
                (-(current.getUncertainties().array()+ 1e-9).log()).max(0.0f); // -log(unc)
            input_alpha =
                (-(input.getUncertainties().array()+ 1e-9).log()).max(0.0f);
        } else if (uncertainty_type_ == UncertaintyType::CONFIDENCE) {
            // If no samples have been added yet, use fixed weight different
            // than defined initial for UNCERTAINTY
            if (current.getSamplesCount() == 0) {
                current_alpha = Eigen::ArrayXf::Ones(num_classes_, 1) * 0.001f;
            }
            current_alpha = (current.getUncertainties().array()/100.0f).exp(); // exp(unc)
            input_alpha = (input.getUncertainties().array()/100.0f).exp();
        } else {
            std::cout << "ERROR: Uncertainty type not supported" << std::endl;
        }
        Eigen::ArrayXf max_alpha = current_alpha.max(input_alpha);
        Eigen::ArrayXf class_ps_weighted =
            current.getProbabilities().pow(current_alpha / max_alpha);

        Eigen::ArrayXf input_class_ps_weighted;
        Eigen::ArrayXf uniform =
            Eigen::ArrayXf::Ones(num_classes_, 1) / float(num_classes_);

        if(beta_ == 1.0){
        input_class_ps_weighted =
            ((1 - class_weights_.array()) * input.getProbabilities() + (class_weights_.array()))*uniform;
        } else {
            input_class_ps_weighted =
            ((1 - beta_) * input.getProbabilities() + (beta_)*uniform).pow(input_alpha / max_alpha);;
        }

        // Normalization constants for each class to get a probability
        // distribution
        Eigen::ArrayXf product = class_ps_weighted * input_class_ps_weighted;
        output.setProbabilities(product / product.sum());

        // Epistemic uncertainty
        if (uncertainty_type_ == UncertaintyType::UNCERTAINTY) {
            output.setUncertainties(current.getUncertainties().array().min(
                input.getUncertainties().array()));
        } else if (uncertainty_type_ == UncertaintyType::CONFIDENCE) {
            output.setUncertainties(current.getUncertainties().array().max(
                input.getUncertainties().array()));
        }
    }

    float multiVariateGaussian(const Eigen::ArrayXf &x,
                               const Eigen::ArrayXf &meanVec,
                               const Eigen::MatrixXf &covMat) {
        // Sum small number to avoid log(0) and sqrt(0)
        Eigen::MatrixXf covMatSum =
            covMat +
            Eigen::MatrixXf::Identity(covMat.rows(), covMat.cols()) * 1e-9;

        // avoid magic numbers in your code. Compilers will be able to compute
        // this at compile time:
        const float logSqrt2Pi = 0.5 * std::log(2 * M_PI);
        typedef Eigen::LLT<Eigen::MatrixXf> Chol;
        Chol chol(covMatSum);
        // Handle non positive definite covariance somehow:
        if (chol.info() != Eigen::Success) throw "decomposition failed!";
        const Chol::Traits::MatrixL &L = chol.matrixL();
        float quadform = (L.solve((x - meanVec).matrix())).squaredNorm();
        return std::exp(-x.rows() * logSqrt2Pi - 0.5 * quadform) /
               L.determinant();
    }

    void fusionGaussianBayesian(const VoxelInfo &current,
                                const VoxelInfo &input, VoxelInfo &output) {
        Eigen::ArrayXf product =
            (current.getProbabilities() * input.getProbabilities());
        output.setProbabilities((product / product.sum()).matrix());
    }

    void fusionSum(const VoxelInfo &current, const VoxelInfo &input,
                   VoxelInfo &output) {
        // Get argmax for input
        int max_class = input.getMostProbableClass();
        Eigen::ArrayXf probabilities = current.getProbabilities();
        // If no samples have been added yet, initialize probabilities
        if (current.getSamplesCount() == 0) {
            probabilities(max_class, 0) = 1;
        } else {
            // Fusion sum
            probabilities(max_class, 0)++;
        }

        output.setProbabilities(probabilities);
    }

    void fusionSumProbabilities(const VoxelInfo &current,
                                const VoxelInfo &input, VoxelInfo &output) {
        // Get argmax for input
        int max_class = input.getMostProbableClass();
        Eigen::ArrayXf probabilities = current.getProbabilities();
        // If no samples have been added yet, initialize probabilities
        if (current.getSamplesCount() == 0) {
            probabilities(max_class, 0) = input.getClassProbability(max_class);
        } else {
            // Fusion sum
            probabilities(max_class, 0) += input.getClassProbability(max_class);
        }

        output.setProbabilities(probabilities);
    }

    VoxelInfo fuseVoxel(const VoxelInfo &current, const VoxelInfo &input) {
        VoxelInfo output(num_classes_);

        fusion_function_(current, input, output);

        output.setSamplesCount(current.getSamplesCount() +
                               input.getSamplesCount());
        output.setGtClassCount(current.getGtClassCount() +
                               input.getGtClassCount());
        return output;
    }

    VoxelInfo averageVoxels(const std::vector<VoxelInfo> &input) {
        // Initialize the output voxel
        const size_t n_classes = input[0].getNumClasses();
        VoxelInfo output(n_classes);
        output.setSamplesCount(input.size());
        Eigen::ArrayXf probabilities = Eigen::ArrayXf::Zero(n_classes);
        Eigen::MatrixXf uncertainties = Eigen::MatrixXf::Zero(n_classes, 1);
        Eigen::ArrayXi gt_class_count = Eigen::ArrayXi::Zero(n_classes);
        // Eigen::ArrayXi class_counts = Eigen::ArrayXi::Zero(n_classes);

        // Loop over samples for average, aleatoric term, gt_class addition and
        // increment sample count
        for (const VoxelInfo &voxel : input) {
            probabilities += voxel.getProbabilities();
            uncertainties += voxel.getUncertainties();
            gt_class_count += voxel.getGtClassCount();
            // class_counts[voxel.getMostProbableClass()]++;
        }

        // Average
        probabilities /= input.size();
        uncertainties /= input.size();

        // Check if winning class has enough samples
        // if (class_counts.maxCoeff() < 5)
        // {
        //     return VoxelInfo(n_classes);
        // }

        return output;
    }

    VoxelInfo fuseVoxelsMC(const std::vector<VoxelInfo> &input) {
        // Initialize the output voxel
        VoxelInfo output(num_classes_);
        output.setSamplesCount(input.size());
        Eigen::ArrayXf probabilities = Eigen::ArrayXf::Zero(num_classes_);
        Eigen::MatrixXf uncertainties = Eigen::MatrixXf::Zero(num_classes_, 1);
        Eigen::ArrayXi gt_class_count = Eigen::ArrayXi::Zero(num_classes_);

        // Loop over samples for average, aleatoric term, gt_class addition and
        // increment sample count
        for (const VoxelInfo &voxel : input) {
            // Average
            probabilities += voxel.getProbabilities();
            // GT
            gt_class_count += voxel.getGtClassCount();
        }

        // Average
        probabilities /= input.size();
        output.setProbabilities(probabilities);
        output.setGtClassCount(gt_class_count);

        // Loop over samples for epistemic (having average)
        for (VoxelInfo voxel : input) {
            Eigen::ArrayXf term = voxel.getProbabilities() - probabilities;
            uncertainties +=
                (term * term)
                    .matrix();  // This is only the trace. If we want a
                                // covariance matrix, we need to multiply by the
                                // transpose in matrix form
        }
        uncertainties /= input.size();
        output.setUncertainties(uncertainties);

        if (fusion_method_ == FusionMethod::GAUSSIAN) {
            Eigen::ArrayXf input_probabilities =
                Eigen::ArrayXf::Zero(num_classes_, 1);
            for (int i = 0; i < num_classes_; i++) {
                Eigen::ArrayXf x_v = Eigen::ArrayXf::Zero(num_classes_, 1);
                x_v(i, 0) = 1;
                Eigen::MatrixXf cov = output.getUncertainties().asDiagonal();
                float prob =
                    multiVariateGaussian(x_v, output.getProbabilities(), cov);
                input_probabilities(i, 0) = prob;
            }
            output.setProbabilities(input_probabilities);
        }
        return output;
    }
};

}  // namespace semantic_mapping

#endif  // SEMANTIC_MAPPING_VOXEL_INTEGRATOR_H_
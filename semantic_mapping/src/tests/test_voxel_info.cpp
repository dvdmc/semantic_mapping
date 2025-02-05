#include <gtest/gtest.h>
#include "semantic_mapping/VoxelInfo.hpp"

using namespace semantic_mapping;

class VoxelInfoTest : public ::testing::Test {
protected:
    void SetUp() override {
        n_classes = 3;
        initial_probabilities = Eigen::ArrayXf::Ones(n_classes, 1) / n_classes;
        initial_uncertainties = Eigen::MatrixXf::Ones(n_classes, 1) * INITIAL_UNCERTAINTY;
    }

    int n_classes;
    Eigen::ArrayXf initial_probabilities;
    Eigen::MatrixXf initial_uncertainties;
};

// Test the default constructor
TEST_F(VoxelInfoTest, DefaultConstructor) {
    VoxelInfo voxel;
    EXPECT_EQ(voxel.getSamplesCount(), 0);
}

// Test the constructor with number of classes
TEST_F(VoxelInfoTest, ConstructorWithNumClasses) {
    VoxelInfo voxel(n_classes);

    EXPECT_EQ(voxel.getNumClasses(), n_classes);
    EXPECT_EQ(voxel.getSamplesCount(), 0);

    for (int i = 0; i < n_classes; i++) {
        EXPECT_NEAR(voxel.getClassProbability(i), initial_probabilities(i), 1e-6);
        EXPECT_NEAR(voxel.getClassUncertainty(i), INITIAL_UNCERTAINTY, 1e-6);
    }
}

// Test the constructor with class probabilities
TEST_F(VoxelInfoTest, ConstructorWithProbabilities) {
    VoxelInfo voxel(initial_probabilities);

    EXPECT_EQ(voxel.getNumClasses(), n_classes);
    EXPECT_EQ(voxel.getSamplesCount(), 0);

    for (int i = 0; i < n_classes; i++) {
        EXPECT_NEAR(voxel.getClassProbability(i), initial_probabilities(i), 1e-6);
    }
}

// Test the full constructor
TEST_F(VoxelInfoTest, FullConstructor) {
    int samples_count = 10;
    Eigen::ArrayXi gt_class_count = {2,3,5};
    VoxelInfo voxel(initial_probabilities, initial_uncertainties, samples_count, gt_class_count);

    EXPECT_EQ(voxel.getNumClasses(), n_classes);
    EXPECT_EQ(voxel.getSamplesCount(), samples_count);

    for (int i = 0; i < n_classes; i++) {
        EXPECT_NEAR(voxel.getClassProbability(i), initial_probabilities(i), 1e-6);
        EXPECT_NEAR(voxel.getClassUncertainty(i), initial_uncertainties(i), 1e-6);
        EXPECT_EQ(voxel.getGtClassCount()(i), gt_class_count(i));
    }
}

// Test the setters
TEST_F(VoxelInfoTest, Setters) {
    VoxelInfo voxel(n_classes);

    Eigen::ArrayXf new_probabilities = Eigen::ArrayXf::Random(n_classes, 1);
    Eigen::MatrixXf new_uncertainties = Eigen::MatrixXf::Random(n_classes, 1);
    int new_samples_count = 5;
    Eigen::ArrayXi new_gt_class_count = Eigen::ArrayXi::Random(n_classes, 1);

    voxel.setProbabilities(new_probabilities);
    voxel.setUncertainties(new_uncertainties);
    voxel.setSamplesCount(new_samples_count);
    voxel.setGtClassCount(new_gt_class_count);

    for (int i = 0; i < n_classes; i++) {
        EXPECT_NEAR(voxel.getClassProbability(i), new_probabilities(i), 1e-6);
        EXPECT_NEAR(voxel.getClassUncertainty(i), new_uncertainties(i), 1e-6);
        EXPECT_EQ(voxel.getGtClassCount()(i), new_gt_class_count(i));
    }
    EXPECT_EQ(voxel.getSamplesCount(), new_samples_count);
}

// Test addGtObservation method
TEST_F(VoxelInfoTest, AddGtObservation) {
    VoxelInfo voxel(n_classes);

    int class_id = 1;
    voxel.addGtObservation(class_id);

    EXPECT_EQ(voxel.getGtClassCount()(class_id), 1);
    for (int i = 0; i < n_classes; i++) {
        if (i != class_id) {
            EXPECT_EQ(voxel.getGtClassCount()(i), 0);
        }
    }
}

// Test getMostProbableClass method
TEST_F(VoxelInfoTest, GetMostProbableClass) {
    Eigen::ArrayXf probs(3, 1);
    probs << 0.2, 0.5, 0.3;
    VoxelInfo voxel(probs);
    int most_probable_class;
    float max_prob;
    voxel.getMostProbClassAndProb(most_probable_class, max_prob);
    EXPECT_EQ(most_probable_class, 1);
    EXPECT_NEAR(max_prob, 0.5, 1e-6);
}

// Test getGtClass method
TEST_F(VoxelInfoTest, getGtClass) {
    Eigen::ArrayXf probs(3, 1);
    probs << 0.2, 0.5, 0.3;
    std::vector<int> gt_obs{1, 0, 1, 1, 2};
    VoxelInfo voxel(probs);
    for (int i = 0; i < 3; i++) {
        voxel.addGtObservation(gt_obs[i]);
    }
    EXPECT_EQ(voxel.getGtClass(), 1);
}

// Test computeEntropy method
TEST_F(VoxelInfoTest, ComputeEntropy) {
    Eigen::ArrayXf probs(3, 1);
    probs << 0.5, 0.5, 0.0;
    VoxelInfo voxel(probs);

    float entropy = voxel.computeEntropy();

    // Entropy of [0.5, 0.5, 0.0] is log(2) (since H = -sum(p * log(p)))
    EXPECT_NEAR(entropy, std::log(2), 1e-6);
}

#ifndef SEMANTIC_MAPPING_VOXEL_MAP_H_
#define SEMANTIC_MAPPING_VOXEL_MAP_H_

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <cereal/archives/binary.hpp>
#include <cereal/types/eigen.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

#include <semantic_mapping/VoxelInfo.hpp>
#include <semantic_mapping/VoxelIntegrator.hpp>

namespace semantic_mapping {

class VoxelHashMap {
   public:
    VoxelHashMap(float _resolution, int _n_classes)
        : resolution_(_resolution),
          center_displacement_(_resolution / 2.0),
          n_classes_(_n_classes),
          verbose_(false) {
        integrator_ = std::make_unique<VoxelIntegrator>(
            n_classes_, FusionMethod::W_BAY, UncertaintyType::UNCERTAINTY, 0.3);
    }
    ~VoxelHashMap() {}

    // Voxelization
    // Voxelize a 3D position in space according to resolution and all values
    // positive
    Eigen::Vector3i voxelizePosition(const Eigen::Vector3f &pos) const {
        return Eigen::Vector3i(
            std::round((pos(0) + center_displacement_) / resolution_ + 1000),
            std::round((pos(1) + center_displacement_) / resolution_ + 1000),
            std::round((pos(2) + center_displacement_) / resolution_ + 1000));
    }

    Eigen::Vector3f devoxelizePosition(const Eigen::Vector3i &voxel) const {
        return Eigen::Vector3f(
            float((voxel(0) - 1000) * resolution_) - center_displacement_,
            float((voxel(1) - 1000) * resolution_) - center_displacement_,
            float((voxel(2) - 1000) * resolution_) - center_displacement_);
    }

    // Create key value from 3D point
    uint64_t pointToKey(const Eigen::Vector3f &point) const {
        Eigen::Vector3i voxel = voxelizePosition(point);
        return (uint64_t)voxel(0) | ((uint64_t)voxel(1) << 16) |
               ((uint64_t)voxel(2) << 32);
    }

    // Obtain 3D point from key value
    Eigen::Vector3f keyToPoint(const uint64_t &key) const {
        Eigen::Vector3i voxel;
        voxel(0) = key & 0xFFFF;
        voxel(1) = (key >> 16) & 0xFFFF;
        voxel(2) = (key >> 32) & 0xFFFF;
        return devoxelizePosition(voxel);
    }

    // Setters

    void setIntegrator(const std::shared_ptr<VoxelIntegrator> _integrator) {
        integrator_ = _integrator;
    }
    void setVerbose(bool _verbose) { verbose_ = _verbose; }

    // Getters

    const std::unordered_map<uint64_t, VoxelInfo> &getVoxelHashMapData() const {
        return voxel_hash_map_;
    }

    // Function to get all keys in the map
    std::vector<uint64_t> getKeys() const {
        std::vector<uint64_t> keys;
        for (auto it = voxel_hash_map_.begin(); it != voxel_hash_map_.end();
             it++) {
            keys.push_back(it->first);
        }
        return keys;
    }
    size_t size() const { return voxel_hash_map_.size(); }
    float getVoxelSize() const { return resolution_; }
    float getResolution() const { return resolution_; }
    int getNumClasses() const { return n_classes_; }

    bool tryGetVoxelPtr(const uint64_t &key, VoxelInfo &voxel) const {
        auto it = voxel_hash_map_.find(key);
        if (it != voxel_hash_map_.end()) {
            voxel = it->second;
            return true;
        } else {
            return false;
        }
    }

    // Get voxel from map
    VoxelInfo getVoxel(const uint64_t &key) const {
        auto it = voxel_hash_map_.find(key);
        if (it != voxel_hash_map_.end()) {
            return it->second;
        } else {
            if (verbose_) {
                std::cout << "Voxel " << keyToPoint(key).transpose()
                          << " not initialized, returning default" << std::endl;
            }
            return VoxelInfo(n_classes_);
        }
    }
    VoxelInfo getVoxel(const Eigen::Vector3f &point) const {
        uint64_t key = pointToKey(point);
        return getVoxel(key);
    }

    VoxelInfo* getVoxelPtr(const uint64_t &key) {
        auto it = voxel_hash_map_.find(key);
        if (it != voxel_hash_map_.end()) {
            return &it->second;
        } else {
            if (verbose_) {
                std::cout << "Voxel " << keyToPoint(key).transpose()
                          << " not initialized, returning default" << std::endl;
            }
            return nullptr;
        }
    }

    VoxelInfo* getVoxelPtr(const Eigen::Vector3f &point) {
        uint64_t key = pointToKey(point);
        return getVoxelPtr(key);
    }

    // Get voxel integrator pointer
    std::shared_ptr<VoxelIntegrator> getIntegrator() const {
        return integrator_;
    }

    // Check if voxel exists in map
    bool hasVoxel(const Eigen::Vector3f &point) const {
        uint64_t key = pointToKey(point);
        return voxel_hash_map_.find(key) != voxel_hash_map_.end();
    }

    // Integrate single voxel measurement into map
    void integrateVoxel(const uint64_t &key, const VoxelInfo &voxel_info,
                        bool no_prior = false) {
        assert(voxel_info.getNumClasses() == n_classes_);
        std::unordered_map<uint64_t, VoxelInfo>::iterator it =
            voxel_hash_map_.find(key);
        if (it == voxel_hash_map_.end()) {
            // std::cout << "Adding new voxel " << keyToPoint(key).transpose() << std::endl;
            // Initialize the voxel with neutral information if prior is
            // available (default true)
            if (no_prior) {
                voxel_hash_map_[key] = voxel_info;
            } else {

                VoxelInfo init_voxel_info(n_classes_);
                voxel_hash_map_[key] =
                    integrator_->fuseVoxel(init_voxel_info, voxel_info);
            }
        } else {
            // std::cout << "Updating voxel " << keyToPoint(key).transpose() << std::endl;
            it->second = integrator_->fuseVoxel(it->second, voxel_info);
        }
        // std::cout << "Finished voxel " << keyToPoint(key).transpose() << std::endl;
    }

    // Integrate single voxel measurement into map
    void integrateVoxel(const Eigen::Vector3f &point,
                        const VoxelInfo &voxel_info, bool no_prior = false) {
        uint64_t key = pointToKey(point);
        integrateVoxel(key, voxel_info);
    }

    // The following functions are used for fusion on voxel levels.
    // Neither of them are part of the final or proposed methods but
    // could be interesting in a different setup.

    // Prepare voxels for Bayesian Fusion based on weighted samples
    void prepareVoxelsForFusion(const Eigen::Vector3f &point,
                                const VoxelInfo &voxel_info) {
        uint64_t key = pointToKey(point);
        pre_integration_voxel_storage_[key].push_back(voxel_info);
        stored_voxels_preintegration_ = true;
    }

    // Integrate stored voxels with average on a voxel level and fusion
    void integrateStoredVoxelsAverage() {
        if (stored_voxels_preintegration_) {
            for (auto &voxels : pre_integration_voxel_storage_) {
                // Combine all the measurements in a voxel
                VoxelInfo average_fused_voxels =
                    integrator_->averageVoxels(voxels.second);
                integrateVoxel(voxels.first, average_fused_voxels);
            }
            // Clear pre integration storage
            pre_integration_voxel_storage_.clear();
            stored_voxels_preintegration_ = false;
        } else {
            std::cout << "No voxels stored for integration" << std::endl;
        }
    }

    // Integrate stored voxels with average on a voxel level and fusion
    void integrateStoredVoxelsMC() {
        if (stored_voxels_preintegration_) {
            for (auto &voxels : pre_integration_voxel_storage_) {
                // Combine all the measurements in a voxel
                VoxelInfo average_fused_voxels =
                    integrator_->fuseVoxelsMC(voxels.second);
                integrateVoxel(voxels.first, average_fused_voxels);
            }
            // Clear pre integration storage
            pre_integration_voxel_storage_.clear();
            stored_voxels_preintegration_ = false;
        } else {
            std::cout << "No voxels stored for integration" << std::endl;
        }
    }

    // float getVoxelMapEntropy()
    // {
    //     float entropy = 0.0;
    //     for (auto it = voxel_hash_map_.begin(); it != voxel_hash_map_.end();
    //          it++) {
    //         entropy += it->second.computeEntropy();
    //     }
    //     return entropy;
    // }

    // Dummy function that allows only to print.
    // TODO: This was now moved to the map evaluator.
    // If needed as a runtime srv add a rosnode
    // void evaluateVoxelMap() {
    //     std::vector<std::vector<int>> confusion_matrix;
    //     evaluateVoxelMapConfMatrix(confusion_matrix);
    // }

    // Overload << operator
    friend std::ostream &operator<<(std::ostream &os, const VoxelHashMap &map) {
        os << "Voxel map with " << map.size() << " voxels" << std::endl;
        os << "Resolution: " << map.getResolution() << std::endl;
        os << "Number of classes: " << map.getNumClasses() << std::endl;

        return os;
    }

    // Serialization

    template <class Archive>
    void serialize(Archive &archive) {
        archive(voxel_hash_map_, resolution_, n_classes_);
    }

    template <class Archive>
    void deserialize(Archive &archive) {
        archive(voxel_hash_map_, resolution_, n_classes_);
    }

    bool serializeVoxelHashMap(std::string filename) {
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Could not open file " << filename << std::endl;
            return false;
        }
        {
            cereal::BinaryOutputArchive archive(file);
            // TODO (KNOWN ISSUE): We are just serializing the map. Missing resolution and n_classes!
            serialize(archive);
        }
        file.close();
        return true;
    }

    bool deserializeVoxelHashMap(std::string filename) {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Could not open file " << filename << std::endl;
            return false;
        }
        {
            cereal::BinaryInputArchive archive(file);
            deserialize(archive);
        }
        file.close();
        return true;
    }

   private:
    float resolution_;
    float center_displacement_;
    std::shared_ptr<VoxelIntegrator> integrator_;

    bool verbose_;

    int n_classes_;

    std::unordered_map<uint64_t, VoxelInfo> voxel_hash_map_;

    std::unordered_map<uint64_t, std::vector<VoxelInfo>>
        pre_integration_voxel_storage_;
    bool stored_voxels_preintegration_ = false;
};

}  // namespace semantic_mapping

#endif  // VOXEL_HASH_MAP_H
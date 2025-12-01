#ifndef INSTANT_NGP_BASE_NGP_DATASET_H
#define INSTANT_NGP_BASE_NGP_DATASET_H

#include <string>
#include <array>
#include <filesystem>

namespace ngp {
    struct LoadDatasetParams {
        std::filesystem::path dataset_path;

        enum class DatasetType {
            NeRfSynthetic,
            LLFF,
            NSVF,
            Unknown
        } dataset_type = DatasetType::Unknown;

        [[nodiscard]] const LoadDatasetParams& check() const;
    };

    struct LoadDatasetResult {
        bool success = false;
        std::string message;

        struct Dataset {
            std::array<std::array<float, 3>, 4> xform{};
            const void* pixels                 = nullptr;
            std::array<uint32_t, 2> resolution = {0, 0};
            std::array<float, 2> focal_length  = {1000.f, 1000.f};
        };

        std::vector<Dataset> dataset;

        [[nodiscard]] const LoadDatasetResult& print() const;
    };

    LoadDatasetResult load_dataset(const LoadDatasetParams& params);
}

#endif //INSTANT_NGP_BASE_NGP_DATASET_H

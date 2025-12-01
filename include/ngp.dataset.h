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

        struct Image {
            std::array<std::array<float, 3>, 4> xform{};
            const unsigned char* pixels       = nullptr;
            std::array<size_t, 2> resolution  = {0, 0};
            std::array<float, 2> focal = {1000.f, 1000.f};
            size_t channel                    = 4;
        };

        std::vector<Image> images;

        LoadDatasetResult() = default;
        ~LoadDatasetResult();
        LoadDatasetResult(const LoadDatasetResult&)            = delete;
        LoadDatasetResult& operator=(const LoadDatasetResult&) = delete;
        LoadDatasetResult(LoadDatasetResult&&)                 = default;
        LoadDatasetResult& operator=(LoadDatasetResult&&)      = default;
        [[maybe_unused]] const LoadDatasetResult& print() const;
    };

    std::shared_ptr<LoadDatasetResult> load_dataset(const LoadDatasetParams& params);
}

#endif //INSTANT_NGP_BASE_NGP_DATASET_H

#ifndef INSTANT_NGP_BASE_NGP_CUDA_TRAIN_H
#define INSTANT_NGP_BASE_NGP_CUDA_TRAIN_H

#include <string>
#include <array>
#include <filesystem>

namespace tcnn {
    struct Ray;
}

namespace ngp::cuda {
    struct ResetSessionParams {
        std::filesystem::path config_path;

        struct {
            std::string otype = "L2";
        } loss;

        struct {
            uint32_t n_pos        = 3;
            uint32_t n_input      = 7;
            uint32_t n_output     = 4;
            uint32_t n_dir_dims   = 3;
            uint32_t n_extra_dims = 0;
        } network;

        [[nodiscard]] const ResetSessionParams& check() const;
    };

    struct ResetSessionResult {
        bool success;
        std::string message;
    };

    ResetSessionResult reset_session(const ResetSessionParams& params);


    struct LoadDatasetParams {
        std::filesystem::path dataset_path;

        enum class DatasetType {
            NerfSynthetic,
            LLFF,
            NSVF,
            Unknown
        } dataset_type = DatasetType::Unknown;

        [[nodiscard]] const LoadDatasetParams& check() const;
    };

    struct LoadDatasetResult {
        bool success = false;
        std::string message;

        struct TrainingImageMetadata {
            const void* pixels    = nullptr;
            const float* depth    = nullptr;
            const tcnn::Ray* rays = nullptr;

            std::array<uint32_t, 2> resolution = {0, 0};
            std::array<float, 2> focal_length  = {1000.f, 1000.f};
        };

        std::vector<TrainingImageMetadata> metadata;
        std::vector<std::array<std::array<float, 3>, 4>> xforms;
    };

    LoadDatasetResult load_dataset(const LoadDatasetParams& params);


    struct TrainParams {

    };

    struct TrainResult {
        bool success;
        std::string message;
    };

    TrainResult train(const TrainParams& params);
}

#endif // INSTANT_NGP_BASE_NGP_CUDA_TRAIN_H

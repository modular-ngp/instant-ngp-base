#ifndef INSTANT_NGP_BASE_NGP_CUDA_TRAIN_H
#define INSTANT_NGP_BASE_NGP_CUDA_TRAIN_H

#include <optional>
#include <string>
#include <filesystem>

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

        [[nodiscard]] std::optional<std::string> check() const;
    };

    struct ResetSessionResult {
        bool success;
        std::string message;
    };

    ResetSessionResult reset_session(const ResetSessionParams& params);


    struct LoadDatasetParams {

    };

    struct LoadDatasetResult {
        bool success;
        std::string message;
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

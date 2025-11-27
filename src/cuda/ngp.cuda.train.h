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

    std::optional<std::string> reset_session(const ResetSessionParams& params);
}

#endif // INSTANT_NGP_BASE_NGP_CUDA_TRAIN_H

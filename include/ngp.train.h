#ifndef INSTANT_NGP_BASE_NGP_TRAIN_H
#define INSTANT_NGP_BASE_NGP_TRAIN_H

#include <string>
#include <filesystem>

namespace ngp {
    struct LoadDatasetResult;
}

namespace ngp {
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

        [[maybe_unused]] const ResetSessionResult& print() const;
    };

    ResetSessionResult reset_session(const ResetSessionParams& params);


    struct TrainParams {
        const std::shared_ptr<LoadDatasetResult> dataset_cpu;
        const size_t n_epoch = 1000;

        [[nodiscard]] const TrainParams& check() const;
    };

    struct TrainResult {
        bool success;
        std::string message;

        [[maybe_unused]] const TrainResult& print() const;
    };

    TrainResult train_session(const TrainParams& params);
}

#endif //INSTANT_NGP_BASE_NGP_TRAIN_H

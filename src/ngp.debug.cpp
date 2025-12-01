#include "ngp.dataset.h"
#include "ngp.train.h"

#ifdef INSTANT_NGP_BUILD_WITH_DEBUG_INFO

const ngp::ResetSessionParams& ngp::ResetSessionParams::check() const {
    if (!std::filesystem::exists(config_path)) throw std::runtime_error("[FATAL ERROR] - Invalid config path: " + config_path.string());
    if (!std::filesystem::is_regular_file(config_path)) throw std::runtime_error("[FATAL ERROR] - Config path is not a regular file: " + config_path.string());
    if (config_path.extension() != ".json") throw std::runtime_error("[FATAL ERROR] - Config file is not a JSON file: " + config_path.string());

    return *this;
}

const ngp::TrainParams& ngp::TrainParams::check() const {
    return *this;
}

const ngp::LoadDatasetParams& ngp::LoadDatasetParams::check() const {
    if (!std::filesystem::exists(dataset_path)) throw std::runtime_error("[FATAL ERROR] - Invalid config path: " + dataset_path.string());
    if (!std::filesystem::is_directory(dataset_path)) throw std::runtime_error("[FATAL ERROR] - Config path is not a directory: " + dataset_path.string());
    {
        bool has_json = false;
        for (const auto& entry : std::filesystem::directory_iterator(dataset_path)) {
            if (entry.path().extension() == ".json") {
                has_json = true;
                break;
            }
        }
        if (!has_json) throw std::runtime_error("[FATAL ERROR] - Dataset directory does not contain any JSON files: " + dataset_path.string());
    }

    return *this;
}

#else

const ngp::ResetSessionParams& ngp::ResetSessionParams::check() const {
    return *this;
}

const ngp::TrainParams& ngp::TrainParams::check() const {
    return *this;
}

const ngp::LoadDatasetParams& ngp::LoadDatasetParams::check() const {
    return *this;
}

#endif

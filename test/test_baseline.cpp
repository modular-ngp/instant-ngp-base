#include "ngp.train.h"
#include "ngp.dataset.h"

int main() {
    const ngp::ResetSessionResult& reset_session_result = ngp::reset_session(ngp::ResetSessionParams
    {
        .config_path = "C:/Users/xayah/Desktop/instant-ngp/configs/nerf/base.json",
    }.check()).print();
    const std::shared_ptr<ngp::LoadDatasetResult> load_dataset_result = ngp::load_dataset(ngp::LoadDatasetParams{
        .dataset_path = "C:/Users/xayah/Desktop/instant-ngp/data/nerf-synthetic/lego",
        .dataset_type = ngp::LoadDatasetParams::DatasetType::NeRfSynthetic,
    }.check());
    [[maybe_unused]] const auto& _       = load_dataset_result->print();
    const ngp::TrainResult& train_result = ngp::train_session(ngp::TrainParams{
        .dataset_cpu = load_dataset_result,
        .n_epoch = 1000,
    }.check()).print();
    return 0;
}

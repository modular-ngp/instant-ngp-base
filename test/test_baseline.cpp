#include "../src/cuda/ngp.cuda.train.h"

int main() {
    ngp::cuda::reset_session(ngp::cuda::ResetSessionParams
    {
        .config_path = "C:/Users/xayah/Desktop/instant-ngp/configs/nerf/base.json",
    }.check());
    ngp::cuda::load_dataset(ngp::cuda::LoadDatasetParams{
        .dataset_path = "C:/Users/xayah/Desktop/instant-ngp/data/nerf-synthetic/lego",
    }.check());
    return 0;
}

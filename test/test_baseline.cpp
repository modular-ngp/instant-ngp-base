#include "../src/cuda/ngp.cuda.train.h"

int main() {
    ngp::cuda::reset_session(
    {
        .config_path = "C:/Users/xayah/Desktop/instant-ngp/configs/nerf/base.json",
    });
    // .config_path = "C:/Users/xayah/Desktop/instant-ngp/data/nerf-synthetic/lego",
    return 0;
}

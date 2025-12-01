#ifndef INSTANT_NGP_BASE_NGP_CUDA_SESSION_CUH
#define INSTANT_NGP_BASE_NGP_CUDA_SESSION_CUH

#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/multi_stream.h>

namespace ngp::cuda::hidden {
    struct NGPSession {
        static NGPSession& instance() {
            static NGPSession instance;
            return instance;
        }

        void reset_session(const nlohmann::json& config, const std::string& otype, uint32_t n_pos, uint32_t n_input, uint32_t n_output, uint32_t n_dir_dims, uint32_t n_extra_dims);

        struct ImagesDataGPU {
            const void* pixels      = nullptr;
            tcnn::ivec2 resolution  = tcnn::ivec2(0);
            tcnn::vec2 focal_length = tcnn::vec2(1000.f);
            tcnn::mat4x3 xform      = tcnn::mat4x3::identity();
        };

        std::vector<ImagesDataGPU> dataset_cpu;
        tcnn::GPUMemory<ImagesDataGPU> dataset_gpu;
        std::vector<tcnn::GPUMemory<uint8_t>> pixel_buffers_gpu;
        tcnn::StreamAndEvent m_stream;
        tcnn::default_rng_t m_rng;

        NGPSession(const NGPSession&)            = delete;
        NGPSession& operator=(const NGPSession&) = delete;
        NGPSession(NGPSession&&)                 = delete;
        NGPSession& operator=(NGPSession&&)      = delete;

    private:
        NGPSession()  = default;
        ~NGPSession();

        std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>> m_loss;
        std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> m_optimizer;
        std::shared_ptr<tcnn::Network<float, tcnn::network_precision_t>> m_network;
        std::shared_ptr<tcnn::Encoding<tcnn::network_precision_t>> m_encoding;
        std::shared_ptr<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>> m_trainer;
        uint32_t m_seed = 1337;
    };
}

#endif //INSTANT_NGP_BASE_NGP_CUDA_SESSION_CUH

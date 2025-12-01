#include "ngp.train.h"
#include "ngp.cuda.nerfnetwork.cuh"

#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/trainer.h>

#include <nlohmann/json.hpp>
#include <fstream>

namespace ngp::cuda::hidden {
    struct NGPSession {
        static NGPSession& instance() {
            static NGPSession instance;
            return instance;
        }

        void reset_session(const nlohmann::json& config, const std::string& otype, uint32_t n_pos, uint32_t n_input, uint32_t n_output, uint32_t n_dir_dims, uint32_t n_extra_dims);

        NGPSession(const NGPSession&)            = delete;
        NGPSession& operator=(const NGPSession&) = delete;
        NGPSession(NGPSession&&)                 = delete;
        NGPSession& operator=(NGPSession&&)      = delete;

    private:
        NGPSession()  = default;
        ~NGPSession() = default;

        std::shared_ptr<tcnn::Loss<tcnn::network_precision_t>> m_loss;
        std::shared_ptr<tcnn::Optimizer<tcnn::network_precision_t>> m_optimizer;
        std::shared_ptr<tcnn::Network<float, tcnn::network_precision_t>> m_network;
        std::shared_ptr<tcnn::Encoding<tcnn::network_precision_t>> m_encoding;
        std::shared_ptr<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>> m_trainer;
        uint32_t m_seed = 1337;
    };

    void NGPSession::reset_session(const nlohmann::json& config, const std::string& otype, const uint32_t n_pos, const uint32_t n_input, const uint32_t n_output, const uint32_t n_dir_dims, const uint32_t n_extra_dims) {
        {
            const auto& loss_config     = config["loss"];
            auto loss_config_expand     = loss_config;
            loss_config_expand["otype"] = otype;
            instance().m_loss.reset(tcnn::create_loss<tcnn::network_precision_t>(loss_config_expand));
        }

        {
            const auto& optimizer_config = config["optimizer"];
            instance().m_optimizer.reset(tcnn::create_optimizer<tcnn::network_precision_t>(optimizer_config));
        }

        {
            const auto& network_config      = config["network"];
            const auto& encoding_config     = config["encoding"];
            const auto& dir_encoding_config = config["dir_encoding"];
            const auto& rgb_network_config  = config["rgb_network"];
            instance().m_network            = std::make_shared<NerfNetwork<tcnn::network_precision_t>>(n_pos, n_dir_dims, n_extra_dims, n_pos + 1, encoding_config, dir_encoding_config, network_config, rgb_network_config);
        }

        {
            instance().m_encoding = dynamic_cast<NerfNetwork<tcnn::network_precision_t>*>(m_network.get())->m_pos_encoding;
        }

        {
            instance().m_trainer = std::make_shared<tcnn::Trainer<float, tcnn::network_precision_t, tcnn::network_precision_t>>(m_network, m_optimizer, m_loss, m_seed);
        }
    }
}

ngp::ResetSessionResult ngp::reset_session(const ResetSessionParams& params) {
    cuda::hidden::NGPSession::instance().reset_session(
        nlohmann::json::parse(std::ifstream(params.config_path)),
        params.loss.otype,
        params.network.n_pos,
        params.network.n_input,
        params.network.n_output,
        params.network.n_dir_dims,
        params.network.n_extra_dims
        );
    return {
        true,
    };
}

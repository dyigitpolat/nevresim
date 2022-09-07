#pragma once

#include "simulator/compute_policy/real_valued_compute.hpp"
#include "simulator/chip_weights.hpp"
#include "common/types.hpp"

#include <numeric>

namespace nevresim {

template <typename Config, typename ComputePolicy>
struct NeuronCompute;

template <typename Config>
class NeuronCompute<Config, RealValuedCompute>
{
    using weights_array_t = std::array<Weight<weight_t>, Config::axon_count_>;

    weights_array_t weights_{};
    Bias<weight_t> bias_{};

public:
    constexpr RealValuedCompute::signal_t
    operator()(const auto& incoming_signal)
    {
        return std::max(
            static_cast<RealValuedCompute::signal_t>(0), 
            std::inner_product(
                std::begin(weights_), std::end(weights_),
                std::begin(incoming_signal), 
                static_cast<RealValuedCompute::signal_t>(0)
            ) + bias_);
    }

    constexpr void reset() {}

    constexpr
    void load_weights(const NeuronWeights<Config{}>& weights)
    {
        weights_ = weights.weights_;
        bias_ = weights.bias_;
    }
};

} // namespace nevresim
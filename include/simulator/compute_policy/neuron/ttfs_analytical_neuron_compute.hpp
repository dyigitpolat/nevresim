#pragma once

#include "simulator/compute_policy/ttfs_analytical_compute.hpp"
#include "simulator/chip_weights.hpp"
#include "common/types.hpp"

#include <numeric>

namespace nevresim {

template <typename Config, typename ComputePolicy>
class NeuronCompute;

/// TTFS analytical neuron: ``relu(Σ w_j · x_j + bias) / threshold``.
///
/// This is the stateless, single-pass neuron used by the continuous
/// (event-based) TTFS deployment.  It is identical to the real-valued
/// neuron except for the final division by ``threshold_``, which
/// normalises activations back into [0, 1] (matching the trained
/// ReLU network's scale).
template <typename Config>
class NeuronCompute<Config, TTFSAnalyticalCompute>
{
    using weight_t = typename Config::weight_t;

    using weights_array_t = std::array<Weight<weight_t>, Config::axon_count_>;
    using signal_t = typename TTFSAnalyticalCompute::signal_t;

    weights_array_t weights_{};
    Bias<weight_t> bias_{};
    Threshold<weight_t> threshold_{1};

public:
    constexpr TTFSAnalyticalCompute::signal_t
    operator()(const auto& incoming_signal)
    {
        auto sum = std::inner_product(
            std::cbegin(weights_), std::cend(weights_),
            std::cbegin(incoming_signal), 
            signal_t{0}
        ) + bias_;

        auto activated = std::max(signal_t{0}, sum);

        // Divide by threshold (= parameter_scale) to recover the original
        // floating-point activation magnitude.
        return activated / static_cast<signal_t>(threshold_);
    }

    constexpr void reset() {}

    constexpr
    void load_weights(
        const NeuronWeights<Config{}, typename Config::weight_t>& weights)
    {
        weights_ = weights.weights_;
        bias_ = weights.bias_;
        threshold_ = weights.threshold_;
    }
};

} // namespace nevresim


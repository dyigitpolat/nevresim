#pragma once

#include "simulator/compute_policy/ttfs_quantized_compute.hpp"
#include "simulator/chip_weights.hpp"
#include "common/types.hpp"

#include <numeric>
#include <algorithm>

namespace nevresim {

template <typename Config, typename ComputePolicy>
class NeuronCompute;

/// TTFS quantized neuron — true cycle-based Phase 1 + Phase 2 simulation.
///
/// Implements the B1-model TTFS dynamics with discrete time steps:
///
///   Phase 1:  V = Σ w_j · a_j + bias     (pre-activation)
///   Phase 2:  for k = 0 … S-1:
///                 if V >= θ  → return (S − k) / S    (fire-once)
///                 V += θ / S                          (constant ramp)
///             return 0                                (never fired)
///
/// This neuron is **stateless**: each invocation computes the full
/// Phase 1 + Phase 2 pipeline from scratch, producing a quantised
/// activation in {0, 1/S, 2/S, …, 1}.
///
template <typename Config, int S>
class NeuronCompute<Config, TTFSQuantizedCompute<S>>
{
    using weight_t = typename Config::weight_t;
    using weights_array_t = std::array<Weight<weight_t>, Config::axon_count_>;
    using signal_t = typename TTFSQuantizedCompute<S>::signal_t;

    weights_array_t weights_{};
    Bias<weight_t> bias_{};
    Threshold<weight_t> threshold_{1};

public:
    constexpr signal_t
    operator()(const auto& incoming_signal)
    {
        // Phase 1: V = W · a + b
        signal_t V = std::inner_product(
            std::cbegin(weights_), std::cend(weights_),
            std::cbegin(incoming_signal),
            signal_t{0}
        ) + bias_;

        const signal_t theta = static_cast<signal_t>(threshold_);
        const signal_t ramp  = theta / S;

        // Phase 2: step through S discrete time steps
        for (int k = 0; k < S; ++k)
        {
            if (V >= theta)
            {
                return static_cast<signal_t>(S - k) / S;
            }
            V += ramp;
        }

        // Neuron never fired within S steps
        return signal_t{0};
    }

    constexpr void reset() {} // Stateless

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


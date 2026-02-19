#pragma once

#include "simulator/compute_policy/ttfs_quantized_compute.hpp"
#include "simulator/chip_weights.hpp"
#include "common/types.hpp"

#include <numeric>
#include <algorithm>

namespace nevresim {

template <typename Config, typename ComputePolicy>
class NeuronCompute;

/// TTFS quantized neuron — **stateful** cycle-based Phase 1 + Phase 2.
///
/// Called once per active cycle by ``Core::compute``.
///
///   First call  (Phase 1):  V = Σ w_j · a_j + bias   (initial charge)
///   Each  call  (Phase 2):  if V ≥ θ  → latch output (S − k) / S
///                           V += θ / S                (constant ramp)
///   After S calls:          return cached output
///
/// The neuron fires **at most once**; after that, the output is latched.
/// ``reset()`` must be called between input samples.
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

    // --- Per-sample state (cleared by reset()) ---
    signal_t membrane_potential_{0};
    signal_t output_{0};
    bool initialized_{false};
    bool has_fired_{false};
    int steps_taken_{0};

public:
    constexpr signal_t
    operator()(const auto& incoming_signal)
    {
        // Phase 1: first active cycle — compute initial charge.
        if (!initialized_)
        {
            membrane_potential_ = std::inner_product(
                std::cbegin(weights_), std::cend(weights_),
                std::cbegin(incoming_signal),
                signal_t{0}
            ) + bias_;
            initialized_ = true;
        }

        // All S steps already done — return cached output.
        if (steps_taken_ >= S) return output_;

        const signal_t theta = static_cast<signal_t>(threshold_);

        // Phase 2: fire-once check + constant ramp.
        if (!has_fired_ && membrane_potential_ >= theta)
        {
            output_ = static_cast<signal_t>(S - steps_taken_) / S;
            has_fired_ = true;
        }

        membrane_potential_ += theta / S;
        ++steps_taken_;

        return output_;
    }

    constexpr void reset()
    {
        membrane_potential_ = 0;
        output_ = 0;
        initialized_ = false;
        has_fired_ = false;
        steps_taken_ = 0;
    }

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

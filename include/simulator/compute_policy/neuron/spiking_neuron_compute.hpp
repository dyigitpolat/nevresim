#pragma once

#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/compute_policy/fire_policy/default_fire.hpp"
#include "simulator/chip_weights.hpp"
#include "common/types.hpp"

#include <numeric>

namespace nevresim {

template <typename Config, typename ComputePolicy>
struct NeuronCompute;

template <typename Config, typename FirePolicy>
class NeuronCompute<Config, SpikingCompute<FirePolicy>>
{
    using weight_t = Config::weight_t;
    using weights_array_t = std::array<Weight<weight_t>, Config::axon_count_>;

    weights_array_t weights_{};
    Bias<weight_t> bias_{};
    Threshold<weight_t> threshold_{};
    MembranePotential<weight_t> membrane_potential_{};

    constexpr void leaky_integrate(const auto& incoming_spikes)
    {
        membrane_potential_ += 
            std::inner_product(
                std::cbegin(weights_), std::cend(weights_),
                std::cbegin(incoming_spikes), 
                static_cast<MembranePotential<weight_t>>(0))
            + bias_ 
            - Config::leak_amount_;
    }

    constexpr spike_t fire()
    {
        return FirePolicy::fire(threshold_, membrane_potential_);
    }

public:
    constexpr SpikingCompute<FirePolicy>::signal_t
    operator()(const auto& incoming_signal)
    {
        leaky_integrate(incoming_signal);
        return fire();
    }
    
    constexpr 
    void reset() 
    {
        membrane_potential_ = 0;
    }

    constexpr
    void load_weights(
        const NeuronWeights<Config{}, typename Config::weight_t>& weights)
    {
        threshold_ = weights.threshold_;
        weights_ = weights.weights_;
        bias_ = weights.bias_;
    }
};

} // namespace nevresim
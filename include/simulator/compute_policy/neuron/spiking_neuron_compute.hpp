#pragma once

#include "simulator/compute_policy/spiking_compute.hpp"
#include "simulator/compute_policy/fire_policy/default_fire.hpp"
#include "simulator/chip_weights.hpp"
#include "common/types.hpp"

namespace nevresim {

template <typename Config, typename ComputePolicy>
class NeuronCompute;

template <typename Config, typename FirePolicy, typename IntegrationPolicy>
class NeuronCompute<Config, SpikingCompute<FirePolicy, IntegrationPolicy>>
{
    using weight_t = typename Config::weight_t;
    using threshold_t = typename Config::threshold_t;
    using weights_array_t = std::array<Weight<weight_t>, Config::axon_count_>;
    // The membrane type is OWNED by the integration policy: a per-cycle law
    // holds the unbounded accumulator this neuron always held, and a
    // fixed-width saturating law holds its own register. One declaration site.
    using membrane_t = typename IntegrationPolicy::template membrane_t<weight_t>;

    weights_array_t weights_{};
    Bias<weight_t> bias_{};
    threshold_t threshold_{};
    membrane_t membrane_potential_{};

public:
    constexpr typename SpikingCompute<FirePolicy, IntegrationPolicy>::signal_t
    operator()(const auto& incoming_signal)
    {
        // Integrate AND fire are one step: a per-event law interleaves them by
        // construction, and a per-cycle law is the degenerate case where the
        // whole contribution is reduced before the single compare.
        return IntegrationPolicy::template step<FirePolicy>(
            weights_, incoming_signal, bias_, Config::leak_amount_,
            threshold_, membrane_potential_);
    }

    constexpr
    void reset()
    {
        membrane_potential_ = 0;
    }

    // Final-membrane read port (Q_T = theta*c_T + m_T decode support).
    constexpr
    membrane_t get_membrane_potential() const
    {
        return membrane_potential_;
    }

    constexpr
    threshold_t get_threshold() const
    {
        return threshold_;
    }

    constexpr
    void load_weights(
        const NeuronWeights<Config{}, typename Config::weight_t, typename Config::threshold_t>& weights)
    {
        threshold_ = weights.threshold_;
        weights_ = weights.weights_;
        bias_ = weights.bias_;
    }
};

} // namespace nevresim
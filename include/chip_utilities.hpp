#pragma once

namespace nevresim
{

struct SpikeSource
{
    core_id_t core_{};
    axon_id_t neuron_{};
};

template<std::size_t AxonCount>
struct CoreConnection {
    std::array<SpikeSource, AxonCount> sources_{};
};

template<
    std::size_t AxonCount,
    std::size_t CoreCount,
    std::size_t OutputSize>
struct ChipConfiguration {
    std::array<CoreConnection<AxonCount>, CoreCount> core_sources_{};
    std::array<SpikeSource, OutputSize> output_sources_{};
};

} // namespace nevresim
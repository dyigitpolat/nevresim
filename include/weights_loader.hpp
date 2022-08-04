#pragma once

#include "chip_weights.hpp"
#include "constants.hpp"
#include "types.hpp"

namespace nevresim {

template <
    std::size_t CoreCount,
    std::size_t NeuronCount,
    std::size_t AxonCount >
class WeightsLoader
{
public:
    using chip_weights_t = ChipWeights<CoreCount, NeuronCount, AxonCount>;

public:
    chip_weights_t chip_weights_;

    friend std::istream& operator>>(
        std::istream& weights_stream, 
        WeightsLoader& loader)
    {
        
        for(auto& core : loader.chip_weights_.cores_)
        {
            for(auto& neuron : core.neurons_)
            {
                weights_stream >> neuron.threshold_;
                for(auto& weight : neuron.weights_)
                {
                    weights_stream >> weight;
                }
            }
        }

        return weights_stream;
    }
};

} // namespace nevresim
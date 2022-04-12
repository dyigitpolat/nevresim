#pragma once

#include <vector>

#include "constants.hpp"
#include "types.hpp"

namespace nevresim {

class InputLoader
{
public:
    int target_;
    int batch_size_;
    int input_size_;
    std::vector<raw_input_t> input_;

    friend std::istream& operator>>(
        std::istream& stream, InputLoader& loader)
    {
        weights_stream >> loader.target_;
        weights_stream >> loader.batch_size_;
        weights_stream >> loader.input_size_;

        loader.input_.clear();
        loader.input_.reserve(input_size);
        for(auto& item : loader.input_)
        {
            weights_stream >> item;
        }

        return stream;
    }
};

} // namespace nevresim
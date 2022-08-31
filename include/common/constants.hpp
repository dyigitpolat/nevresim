#pragma once

#include "common/types.hpp"

#include <limits>

namespace nevresim {

constexpr core_id_t k_input_buffer_id{
    std::numeric_limits<core_id_t>::max()
};

constexpr core_id_t k_no_connection{
    std::numeric_limits<core_id_t>::max() - 1
};

} // namespace nevresim

#pragma once

#include "common/xf_common.hpp"

namespace genio {

template <int TYPE,
          int ROWS,
          int COLS,
          int NPC,
          int XFCVDEPTH_IN = _XFCVDEPTH_DEFAULT,
          int XFCVDEPTH_OUT = _XFCVDEPTH_DEFAULT>
void identity(xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_IN>& input,
              xf::cv::Mat<TYPE, ROWS, COLS, NPC, XFCVDEPTH_OUT>& output) {
#pragma HLS INLINE OFF

    const int words = input.rows * (input.cols >> XF_BITSHIFT(NPC));
copy_pixels:
    for (int index = 0; index < words; ++index) {
#pragma HLS PIPELINE II=1
        output.write(index, input.read(index));
    }
}

} // namespace genio

#ifndef XF_FIFO_UTILS_HPP
#define XF_FIFO_UTILS_HPP

#include <algorithm>
#include <ap_int.h>
#include <hls_stream.h>
#include <math.h>

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "common/xf_infra.hpp"

template<int IN_WIDTH, int OUT_WIDTH, int N_ITERATIONS>
void fifoWidthAdapter(hls::stream<ap_uint<IN_WIDTH>> &fifo_in, hls::stream<ap_uint<OUT_WIDTH>> &fifo_out) {
#pragma HLS INLINE off

    static ap_uint<IN_WIDTH + OUT_WIDTH> shift_reg = 0;
    static unsigned int count = 0, updated_counter = 0;

    for (int i = 0; i < N_ITERATIONS; i++) {

        if (OUT_WIDTH > IN_WIDTH) {

            ap_uint<IN_WIDTH> in_value = fifo_in.read();

            shift_reg.range(count + IN_WIDTH - 1, count) = in_value;

            //Shortpath to relax timings
            updated_counter = count + IN_WIDTH;

            if (updated_counter >= OUT_WIDTH) {
                ap_uint<OUT_WIDTH> out_value = shift_reg.range(OUT_WIDTH - 1, 0);

                fifo_out.write(out_value);

                shift_reg = shift_reg >> OUT_WIDTH;

                count += (IN_WIDTH - OUT_WIDTH);
            }else{
                count += IN_WIDTH;
            }

        } else if (OUT_WIDTH < IN_WIDTH) {

            if (count < OUT_WIDTH) {
                ap_uint<IN_WIDTH> in_value = fifo_in.read();

                shift_reg.range(count + IN_WIDTH - 1, count) = in_value;

                ap_uint<OUT_WIDTH> out_value = shift_reg.range(OUT_WIDTH - 1, 0);

                fifo_out.write(out_value);

                shift_reg = shift_reg >> OUT_WIDTH;

                count += (IN_WIDTH - OUT_WIDTH);
            }else{

                ap_uint<OUT_WIDTH> out_value = shift_reg.range(OUT_WIDTH - 1, 0);

                fifo_out.write(out_value);

                shift_reg = shift_reg >> OUT_WIDTH;
                count -= OUT_WIDTH;

            }

        } else {
            // Mismo ancho
            fifo_out.write(fifo_in.read());
        }
    }
}

/*
 * Conversión FIFO simple -> xf::cv::Mat
 *
 * La FIFO contiene directamente palabras ap_uint<W>.
 * Cada palabra se escribe en la posición i de la xf::cv::Mat.
 */
template<int TYPE, int ROWS, int COLS, int NPC>
void fifo2xfMat(hls::stream<ap_uint<XF_PIXELWIDTH(TYPE, NPC)>>& fifo, xf::cv::Mat<TYPE, ROWS, COLS, NPC>& img) {
#pragma HLS INLINE off

    const int words = ROWS * COLS / XF_NPIXPERCYCLE(NPC);

    for (int i = 0; i < words; i++) {
        ap_uint<XF_PIXELWIDTH(TYPE, NPC)> value = fifo.read();
        img.write(i, value);
    }
}


/*
 * Conversión xf::cv::Mat -> FIFO simple
 *
 * Se lee cada palabra de la xf::cv::Mat y se escribe directamente
 * en la FIFO de salida.
 */
template<int TYPE, int ROWS, int COLS, int NPC>
void xfMat2fifo(xf::cv::Mat<TYPE, ROWS, COLS, NPC>& img, hls::stream<ap_uint<XF_PIXELWIDTH(TYPE, NPC)>>& fifo) {
#pragma HLS INLINE off

    const int words = ROWS * COLS / XF_NPIXPERCYCLE(NPC);

    for (int i = 0; i < words; i++) {
        ap_uint<XF_PIXELWIDTH(TYPE, NPC)> value = img.read(i);
        fifo.write(value);
    }
}

#endif

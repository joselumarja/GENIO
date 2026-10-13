#ifndef XF_AXI_STREAM_UTILS_HPP
#define XF_AXI_STREAM_UTILS_HPP

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "common/xf_infra.hpp"
#include "xf_fifo_utils.hpp"

template<int AXI_WIDTH, int USER_WIDTH, int ID_WIDTH, int DEST_WIDTH>
void axiStream2fifo(
    hls::stream<ap_axiu<AXI_WIDTH, USER_WIDTH, ID_WIDTH, DEST_WIDTH>>& axi_stream,
    hls::stream<ap_uint<AXI_WIDTH>>& fifo,
    int words
) {
#pragma HLS INLINE off

    for (int i = 0; i < words; i++) {
        ap_axiu<AXI_WIDTH, USER_WIDTH, ID_WIDTH, DEST_WIDTH> value = axi_stream.read();
        fifo.write(value.data);
    }
}

template<int AXI_WIDTH, int USER_WIDTH, int ID_WIDTH, int DEST_WIDTH>
void fifo2axiStream(
    hls::stream<ap_uint<AXI_WIDTH>>& fifo,
    hls::stream<ap_axiu<AXI_WIDTH, USER_WIDTH, ID_WIDTH, DEST_WIDTH>>& axi_stream,
    int words
) {
#pragma HLS INLINE off

    for (int i = 0; i < words; i++) {
        ap_axiu<AXI_WIDTH, USER_WIDTH, ID_WIDTH, DEST_WIDTH> value;
        value.data = fifo.read();
        value.keep = -1;
        value.strb = -1;
        value.user = 0;
        value.id = 0;
        value.dest = 0;
        value.last = (i == words - 1);
        axi_stream.write(value);
    }
}

template<int AXI_WIDTH, int TYPE, int ROWS, int COLS, int NPC, int USER_WIDTH, int ID_WIDTH, int DEST_WIDTH>
void axiStream2xfMat(
    hls::stream<ap_axiu<AXI_WIDTH, USER_WIDTH, ID_WIDTH, DEST_WIDTH>>& axi_stream,
    xf::cv::Mat<TYPE, ROWS, COLS, NPC>& img
) {
#pragma HLS INLINE off

    const int axi_words = ROWS * COLS * XF_PIXELWIDTH(TYPE, NPC) / AXI_WIDTH;
    const int mat_words = ROWS * COLS / XF_NPIXPERCYCLE(NPC);

    hls::stream<ap_uint<AXI_WIDTH>> axi_fifo;
    hls::stream<ap_uint<XF_PIXELWIDTH(TYPE, NPC)>> mat_fifo;

    axiStream2fifo<AXI_WIDTH, USER_WIDTH, ID_WIDTH, DEST_WIDTH>(axi_stream, axi_fifo, axi_words);
    fifoWidthAdapter<AXI_WIDTH, XF_PIXELWIDTH(TYPE, NPC), mat_words>(axi_fifo, mat_fifo);
    fifo2xfMat<TYPE, ROWS, COLS, NPC>(mat_fifo, img);
}

template<int AXI_WIDTH, int TYPE, int ROWS, int COLS, int NPC, int USER_WIDTH, int ID_WIDTH, int DEST_WIDTH>
void xfMat2axiStream(
    xf::cv::Mat<TYPE, ROWS, COLS, NPC>& img,
    hls::stream<ap_axiu<AXI_WIDTH, USER_WIDTH, ID_WIDTH, DEST_WIDTH>>& axi_stream
) {
#pragma HLS INLINE off

    const int axi_words = ROWS * COLS * XF_PIXELWIDTH(TYPE, NPC) / AXI_WIDTH;
    const int mat_words = ROWS * COLS / XF_NPIXPERCYCLE(NPC);

    hls::stream<ap_uint<XF_PIXELWIDTH(TYPE, NPC)>> mat_fifo;
    hls::stream<ap_uint<AXI_WIDTH>> axi_fifo;

    xfMat2fifo<TYPE, ROWS, COLS, NPC>(img, mat_fifo);
    fifoWidthAdapter<XF_PIXELWIDTH(TYPE, NPC), AXI_WIDTH, axi_words>(mat_fifo, axi_fifo);
    fifo2axiStream<AXI_WIDTH, USER_WIDTH, ID_WIDTH, DEST_WIDTH>(axi_fifo, axi_stream, axi_words);
}

#endif

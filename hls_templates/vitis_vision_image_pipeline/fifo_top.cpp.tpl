#include <ap_int.h>
#include <hls_stream.h>

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "common/xf_infra.hpp"
#include "xf_fifo_utils.hpp"

@INCLUDES@

#define ROWS @ROWS@
#define COLS @COLS@
#define TYPE @TYPE@
#define NPC @NPC@

void @TOP_FUNCTION@(
    hls::stream<ap_uint<XF_PIXELWIDTH(TYPE, NPC)>>& input_fifo,
    hls::stream<ap_uint<XF_PIXELWIDTH(@OUTPUT_TYPE@, @OUTPUT_NPC@)>>& output_fifo
) {
#pragma HLS INTERFACE ap_fifo port=input_fifo
#pragma HLS INTERFACE ap_fifo port=output_fifo
#pragma HLS INTERFACE s_axilite port=return bundle=control
@TOP_PRAGMAS@
#pragma HLS DATAFLOW

    xf::cv::Mat<TYPE, ROWS, COLS, NPC> input_mat(ROWS, COLS);

    fifo2xfMat<TYPE, ROWS, COLS, NPC>(input_fifo, input_mat);

@INTERMEDIATE_DECLARATIONS@

@PIPELINE_BODY@

    xfMat2fifo<@OUTPUT_TYPE@, @OUTPUT_ROWS@, @OUTPUT_COLS@, @OUTPUT_NPC@>(@output_0, output_fifo);
}

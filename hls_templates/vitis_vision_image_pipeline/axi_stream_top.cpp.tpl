#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "common/xf_infra.hpp"
#include "xf_axi_stream_utils.hpp"

@INCLUDES@

#define ROWS @ROWS@
#define COLS @COLS@
#define TYPE @TYPE@
#define NPC @NPC@
#define AXI_WIDTH @AXI_WIDTH@
#define AXI_USER_WIDTH @AXI_USER_WIDTH@
#define AXI_ID_WIDTH @AXI_ID_WIDTH@
#define AXI_DEST_WIDTH @AXI_DEST_WIDTH@

typedef ap_axiu<AXI_WIDTH, AXI_USER_WIDTH, AXI_ID_WIDTH, AXI_DEST_WIDTH> axi_word_t;

void @TOP_FUNCTION@(
    hls::stream<axi_word_t>& input_stream,
    hls::stream<axi_word_t>& output_stream
) {
#pragma HLS INTERFACE axis port=input_stream
#pragma HLS INTERFACE axis port=output_stream
#pragma HLS INTERFACE s_axilite port=return bundle=control
@TOP_PRAGMAS@
#pragma HLS DATAFLOW

    xf::cv::Mat<TYPE, ROWS, COLS, NPC> input_mat(ROWS, COLS);

    axiStream2xfMat<AXI_WIDTH, TYPE, ROWS, COLS, NPC, AXI_USER_WIDTH, AXI_ID_WIDTH, AXI_DEST_WIDTH>(input_stream, input_mat);

@INTERMEDIATE_DECLARATIONS@

@PIPELINE_BODY@

    xfMat2axiStream<AXI_WIDTH, @OUTPUT_TYPE@, @OUTPUT_ROWS@, @OUTPUT_COLS@, @OUTPUT_NPC@, AXI_USER_WIDTH, AXI_ID_WIDTH, AXI_DEST_WIDTH>(@output_0, output_stream);
}

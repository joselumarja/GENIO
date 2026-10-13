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

#define SAFA_FIFO_WIDTH 32
#define INPUT_WIDTH XF_PIXELWIDTH(TYPE, NPC)
#define OUTPUT_WIDTH XF_PIXELWIDTH(@OUTPUT_TYPE@, @OUTPUT_NPC@)
#define INPUT_WORDS ((ROWS * COLS) / XF_NPIXPERCYCLE(NPC))
#define OUTPUT_WORDS ((@OUTPUT_ROWS@ * @OUTPUT_COLS@) / XF_NPIXPERCYCLE(@OUTPUT_NPC@))

static_assert(INPUT_WIDTH <= SAFA_FIFO_WIDTH,
              "SAFA FIFO input packing requires INPUT_WIDTH <= 32");
static_assert(OUTPUT_WIDTH <= SAFA_FIFO_WIDTH,
              "SAFA FIFO output packing requires OUTPUT_WIDTH <= 32");
static_assert((INPUT_WORDS * INPUT_WIDTH) % SAFA_FIFO_WIDTH == 0,
              "SAFA FIFO input frame must contain a whole number of 32-bit words");
static_assert((OUTPUT_WORDS * OUTPUT_WIDTH) % SAFA_FIFO_WIDTH == 0,
              "SAFA FIFO output frame must contain a whole number of 32-bit words");

void @TOP_FUNCTION@(
    hls::stream<ap_uint<SAFA_FIFO_WIDTH>>& BUS_IN,
    hls::stream<ap_uint<SAFA_FIFO_WIDTH>>& BUS_OUT
) {
#pragma HLS INTERFACE ap_fifo port=BUS_IN
#pragma HLS INTERFACE ap_fifo port=BUS_OUT
#pragma HLS INTERFACE ap_ctrl_hs port=return
@TOP_PRAGMAS@
#pragma HLS DATAFLOW

    hls::stream<ap_uint<INPUT_WIDTH>> input_fifo;
    hls::stream<ap_uint<OUTPUT_WIDTH>> output_fifo;
#pragma HLS STREAM variable=input_fifo depth=2
#pragma HLS STREAM variable=output_fifo depth=2

    xf::cv::Mat<TYPE, ROWS, COLS, NPC> input_mat(ROWS, COLS);

    fifoWidthAdapter<SAFA_FIFO_WIDTH, INPUT_WIDTH, INPUT_WORDS>(BUS_IN, input_fifo);
    fifo2xfMat<TYPE, ROWS, COLS, NPC>(input_fifo, input_mat);

@INTERMEDIATE_DECLARATIONS@

@PIPELINE_BODY@

    xfMat2fifo<@OUTPUT_TYPE@, @OUTPUT_ROWS@, @OUTPUT_COLS@, @OUTPUT_NPC@>(@output_0, output_fifo);
    fifoWidthAdapter<OUTPUT_WIDTH, SAFA_FIFO_WIDTH, OUTPUT_WORDS>(output_fifo, BUS_OUT);
}

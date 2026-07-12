# Vitis Vision Image Pipeline Templates

Templates for GENIO-generated HLS image-processing pipelines.

Files:

- `axi_stream_top.cpp.tpl`: top function template with AXI stream ports.
- `fifo_top.cpp.tpl`: top function template with simple FIFO ports.
- `include/xf_fifo_utils.hpp`: FIFO width adapter and FIFO/`xf::cv::Mat` conversion helpers.
- `include/xf_axi_stream_utils.hpp`: raw AXI Stream and `xf::cv::Mat` conversion helpers.

Required placeholders:

- `@TOP_FUNCTION@`
- `@ROWS@`
- `@COLS@`
- `@TYPE@`
- `@NPC@`
- `@INCLUDES@`
- `@INTERMEDIATE_DECLARATIONS@`
- `@PIPELINE_BODY@`
- `@output_0@`
- `@OUTPUT_TYPE@`
- `@OUTPUT_ROWS@`
- `@OUTPUT_COLS@`
- `@OUTPUT_NPC@`

Additional AXI stream placeholders:

- `@AXI_WIDTH@`
- `@AXI_USER_WIDTH@`
- `@AXI_ID_WIDTH@`
- `@AXI_DEST_WIDTH@`

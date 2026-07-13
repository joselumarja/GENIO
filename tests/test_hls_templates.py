from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "hls_templates/vitis_vision_image_pipeline"


def test_fifo_utils_template_contains_conversion_helpers() -> None:
    source = (TEMPLATE_DIR / "include/xf_fifo_utils.hpp").read_text(encoding="utf-8")

    assert "#ifndef XF_FIFO_UTILS_HPP" in source
    assert "void fifoWidthAdapter" in source
    assert "void fifo2xfMat" in source
    assert "void xfMat2fifo" in source
    assert "xf::cv::Mat<TYPE, ROWS, COLS, NPC>" in source


def test_axi_stream_utils_template_contains_raw_axi_helpers() -> None:
    source = (TEMPLATE_DIR / "include/xf_axi_stream_utils.hpp").read_text(encoding="utf-8")

    assert "#ifndef XF_AXI_STREAM_UTILS_HPP" in source
    assert "void axiStream2fifo" in source
    assert "void fifo2axiStream" in source
    assert "void axiStream2xfMat" in source
    assert "void xfMat2axiStream" in source
    assert "AXIvideo2xfMat" not in source
    assert "xfMat2AXIvideo" not in source


def test_fifo_top_template_declares_xf_mat_pipeline_placeholders() -> None:
    source = (TEMPLATE_DIR / "fifo_top.cpp.tpl").read_text(encoding="utf-8")

    assert "void @TOP_FUNCTION@" in source
    assert "#define ROWS @ROWS@" in source
    assert "#define COLS @COLS@" in source
    assert "hls::stream<ap_uint<XF_PIXELWIDTH(TYPE, NPC)>>" in source
    assert "hls::stream<ap_uint<XF_PIXELWIDTH(@OUTPUT_TYPE@, @OUTPUT_NPC@)>>" in source
    assert "fifo2xfMat<TYPE, ROWS, COLS, NPC>" in source
    assert "@PIPELINE_BODY@" in source
    assert "xfMat2fifo<@OUTPUT_TYPE@, @OUTPUT_ROWS@, @OUTPUT_COLS@, @OUTPUT_NPC@>" in source
    assert "(@output_0, output_fifo)" in source
    assert "output_mat" not in source


def test_axi_stream_top_template_declares_interface_placeholders() -> None:
    source = (TEMPLATE_DIR / "axi_stream_top.cpp.tpl").read_text(encoding="utf-8")

    assert "typedef ap_axiu<AXI_WIDTH, AXI_USER_WIDTH, AXI_ID_WIDTH, AXI_DEST_WIDTH> axi_word_t" in source
    assert "void @TOP_FUNCTION@" in source
    assert "#pragma HLS INTERFACE axis port=input_stream" in source
    assert "axiStream2xfMat<AXI_WIDTH, TYPE, ROWS, COLS, NPC" in source
    assert "@PIPELINE_BODY@" in source
    assert "xfMat2axiStream<AXI_WIDTH, @OUTPUT_TYPE@" in source
    assert "output_mat" not in source


def test_default_hls_config_template_exists() -> None:
    source = (TEMPLATE_DIR / "hls_config.cfg").read_text(encoding="utf-8")

    assert "part=" in source
    assert "[hls]" in source
    assert "clock=5" in source
    assert "flow_target=vivado" in source
    assert "syn.file=src/pipeline.cpp" in source
    assert "syn.top=top" in source

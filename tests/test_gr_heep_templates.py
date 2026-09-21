from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "gr_heep_templates"


def test_mcu_gen_template_exposes_gr_heep_configuration_tokens() -> None:
    source = (TEMPLATE_DIR / "config/mcu-gen-config.py.tpl").read_text(
        encoding="utf-8"
    )

    assert "XHeep(BusType.@BUS_TYPE@)" in source
    assert 'CPU("@CPU@")' in source
    assert "memory_ss.add_ram_banks(@RAM_BANKS@)" in source
    assert "num_channels=@DMA_NUM_CHANNELS@" in source
    assert 'hw_fifo_mode="yes"' in source
    assert "hw_fifo_channels = @HW_FIFO_CHANNELS@" in source
    assert "external_interrupts = 2" in source
    assert '"SAFA"' in source
    assert '"OBI_Traffic_Generator"' in source


def test_safa_wrapper_template_uses_dynamic_hls_top_module() -> None:
    source = (TEMPLATE_DIR / "safa/safa_wrapper.sv.tpl").read_text(
        encoding="utf-8"
    )

    assert "@HLS_TOP_MODULE@ i_hls_top" in source
    assert ".BUS_IN_dout" in source
    assert ".BUS_OUT_din" in source
    assert ".ap_start" in source
    assert "REG_ACTIVE_CYCLES" in source
    assert "REG_INPUT_STALLS" in source
    assert "REG_OUTPUT_STALLS" in source
    assert "REG_DMA_PUSH_STALLS" in source
    assert "REG_DMA_POP_STALLS" in source
    assert " top i_hls_top" not in source


def test_hls_component_core_template_uses_dynamic_top_and_rtl_files() -> None:
    source = (
        TEMPLATE_DIR / "safa/hls_accelerator_component.core.tpl"
    ).read_text(encoding="utf-8")

    assert "@HLS_RTL_FILES@" in source
    assert "toplevel: @HLS_TOP_MODULE@" in source
    assert "@HLS_DESIGN_SYNTHESIZEZ_FILES" not in source


def test_application_templates_only_use_dma_for_safa_dataflow() -> None:
    applications = tuple(
        sorted((TEMPLATE_DIR / "applications").glob("*/main.c.tpl"))
    )
    assert applications

    forbidden_auxiliary_dma = (
        "TRAFFIC_DMA_CHANNEL",
        "traffic_transaction",
        "traffic_source_target",
        "traffic_destination_target",
        "configure_traffic_dma",
        "initialize_traffic",
        "traffic_checksum",
        "GENIO_METRIC:traffic_words",
        "GENIO_METRIC:traffic_checksum",
    )
    for application in applications:
        source = application.read_text(encoding="utf-8")
        assert all(token not in source for token in forbidden_auxiliary_dma), application
        assert "accelerator_transaction.hw_fifo_en = 1" in source, application
        assert "accelerator_transaction.channel = GENIO_DMA_CHANNEL" in source, application
        assert "dma_load_transaction(&accelerator_transaction)" in source, application
        assert "dma_launch(&accelerator_transaction)" in source, application
        assert "GENIO_METRIC:safa_active_cycles" in source, application
        assert "GENIO_STATUS:%d" in source, application

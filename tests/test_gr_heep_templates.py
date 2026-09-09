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
    assert "external_interrupts = 1" in source
    assert '"SAFA"' in source


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


def test_mem_to_flash_application_template_preserves_safa_metric_protocol() -> None:
    application_dir = TEMPLATE_DIR / "applications/genio_trans_mem_flash"
    source = (application_dir / "main.c.tpl").read_text(encoding="utf-8")
    header = (application_dir / "main.h.tpl").read_text(encoding="utf-8")
    config = (application_dir / "genio_app_config.h.tpl").read_text(
        encoding="utf-8"
    )

    assert "accelerator_source.ptr = (uint8_t *)image_input" in source
    assert "accelerator_destination.ptr = (uint8_t *)image_output" in source
    assert "DMA_TRIG_SLOT_SPI_FLASH" not in source
    assert "w25q128jw_erase_and_write_standard" in source
    assert "GENIO_OUTPUT_WORDS > GENIO_INPUT_WORDS" in source
    assert "dma_load_transaction(transaction)" in source
    assert "dma_launch(transaction)" in source
    assert "dma_completed && status.done" in source
    assert "capture_safa_counters();" in source
    assert "safa_abort(&safa)" not in source
    assert "GENIO_PERF_BEGIN(application)" in source
    assert "GENIO_METRIC:safa_active_cycles" in source
    assert "GENIO_METRIC:safa_input_stall_cycles" in source
    assert "GENIO_METRIC:safa_output_stall_cycles" in source
    assert "GENIO_STATUS:%d" in source
    assert '.xheep_data_flash_only' not in header
    assert "@IMAGE_WORDS@" in header
    assert "@FLASH_OUTPUT_OFFSET@" in config

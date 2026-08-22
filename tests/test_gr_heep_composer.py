import json
from pathlib import Path

import pytest

from genio import (
    ComposerError,
    GRHeepConfigurationComposer,
    GRHeepConfigurationPackage,
    HLSRTLArtifact,
    Individual,
    StageChoice,
)


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS_PATH = ROOT / "search_space/stages/definitions"
TEMPLATES_PATH = ROOT / "gr_heep_templates"


def make_individual() -> Individual:
    return Individual.from_slots(
        id="gr_heep_overlay",
        scenario="gr_heep_overlay",
        design={
            "system": {
                "cpu": "cv32e20",
                "bus_type": "onetoM",
                "dma_fifo_depth": 8,
                "accelerator_fifo_depth": 8,
                "accelerator_fifo_almost_full_margin": 2,
                "unbound_example": "ignored",
            }
        },
        slots=[StageChoice(slot=0, stage="nop")],
    )


def make_composer(**kwargs) -> GRHeepConfigurationComposer:
    return GRHeepConfigurationComposer(
        DEFINITIONS_PATH,
        templates_path=TEMPLATES_PATH,
        **kwargs,
    )


def test_gr_heep_composer_renders_configuration_and_application_overlay() -> None:
    package = make_composer(
        parameter_bindings={
            "cpu": "CPU",
            "bus_type": "BUS_TYPE",
            "dma_fifo_depth": "DMA_FIFO_DEPTH",
        }
    ).compose(make_individual())

    assert isinstance(package, GRHeepConfigurationPackage)
    assert package.entrypoint == "config/mcu-gen-config.py"
    assert set(package.files) == {
        "config/mcu-gen-config.py",
        "sw/applications/genio_target/genio_app_config.h",
        "sw/applications/genio_target/genio_perf.h",
        "sw/applications/genio_target/main.c",
        "sw/applications/genio_target/main.h",
        "sw/applications/genio_target/safa.c",
        "sw/applications/genio_target/safa.h",
        "sw/applications/genio_target/safa_regs.h",
    }
    config = package.files["config/mcu-gen-config.py"]
    assert "XHeep(BusType.onetoM)" in config
    assert 'CPU("cv32e20")' in config
    assert "fifo_depth=8" in config
    assert "memory_ss.add_ram_banks([32] * 6)" in config
    assert "@" not in config
    assert "GENIO_PERF_BEGIN(application)" in package.files[
        "sw/applications/genio_target/main.c"
    ]
    assert "genio_perf_init();" in package.files[
        "sw/applications/genio_target/main.c"
    ]
    assert "dma_launch(&accelerator_transaction)" in package.files[
        "sw/applications/genio_target/main.c"
    ]
    assert "dma_launch(&traffic_transaction)" in package.files[
        "sw/applications/genio_target/main.c"
    ]
    assert '#include "gr_heep.h"' in package.files[
        "sw/applications/genio_target/main.c"
    ]
    assert "SAFA_PERIPH_START_ADDRESS" in package.files[
        "sw/applications/genio_target/main.c"
    ]
    assert "image_input" in package.files["sw/applications/genio_target/main.h"]
    assert package.metadata["system_design"]["unbound_example"] == "ignored"
    assert package.metadata["unbound_system_parameters"] == ["unbound_example"]
    assert "UNBOUND_EXAMPLE" not in package.metadata["rendered_configuration"]


def test_gr_heep_composer_defaults_match_current_gen_heep_configuration() -> None:
    individual = Individual.from_slots(
        id="defaults",
        scenario="defaults",
        design={"system": {}},
        slots=[StageChoice(slot=0, stage="nop")],
    )
    package = make_composer().compose(individual)
    config = package.files["config/mcu-gen-config.py"]

    assert "XHeep(BusType.NtoM)" in config
    assert 'CPU("cv32e40px")' in config
    assert "memory_ss.add_ram_banks([32] * 6)" in config
    assert 'if 4:\n        memory_ss.add_ram_banks_il(\n            4,\n            16,' in config
    assert "num_channels=4" in config
    assert "num_master_ports=2" in config
    assert "num_channels_per_master_port=2" in config
    assert "fifo_depth=4" in config
    assert 'addr_mode="yes"' in config
    assert 'subaddr_mode="yes"' in config
    assert 'zero_padding="yes"' in config
    assert "hw_fifo_channels = [0]" in config


def test_gr_heep_composer_resolves_search_space_memory_configuration() -> None:
    individual = Individual.from_slots(
        id="memory_configuration",
        scenario="memory_configuration",
        design={
            "system": {
                "memory_total_kib": 256,
                "memory_bank_size_kib": 32,
                "memory_interleaved_ratio": 50,
                "memory_placement": "input_output_interleaved",
            }
        },
        slots=[StageChoice(slot=0, stage="nop")],
    )

    package = make_composer().compose(individual)
    config = package.files["config/mcu-gen-config.py"]

    assert "memory_ss.add_ram_banks([32] * 4)" in config
    assert "if 4:" in config
    assert "        4," in config
    assert "        32," in config
    assert "LinkerSection.by_size(\"code\", 0, 0x00010000)" in config
    rendered = package.metadata["rendered_configuration"]
    assert rendered["RAM_BANKS"] == "[32] * 4"
    assert rendered["MEMORY_TOTAL_KIB"] == "256"
    assert rendered["MEMORY_CONTINUOUS_KIB"] == "128"
    assert rendered["MEMORY_INTERLEAVED_KIB"] == "128"


def test_gr_heep_composer_applies_native_system_hyperparameters() -> None:
    package = make_composer().compose(make_individual())

    config = package.files["config/mcu-gen-config.py"]
    assert "XHeep(BusType.onetoM)" in config
    assert 'CPU("cv32e20")' in config
    assert "fifo_depth=8" in config
    assert package.metadata["rendered_configuration"]["ACCELERATOR_FIFO_DEPTH"] == "8"
    assert package.metadata["rendered_configuration"][
        "ACCELERATOR_FIFO_ALMOST_FULL_MARGIN"
    ] == "2"


def test_gr_heep_composer_rejects_invalid_safa_fifo_margin() -> None:
    individual = Individual.from_slots(
        id="bad_fifo",
        scenario="bad_fifo",
        design={
            "system": {
                "accelerator_fifo_depth": 2,
                "accelerator_fifo_almost_full_margin": 2,
            }
        },
        slots=[StageChoice(slot=0, stage="nop")],
    )

    with pytest.raises(ComposerError, match="smaller than accelerator_fifo_depth"):
        make_composer().compose(individual)


def test_xheep_search_space_system_keys_have_composer_equivalents() -> None:
    specification = json.loads(
        (ROOT / "search_space/tests/insect_xheep_exploration_pipeline.json").read_text(
            encoding="utf-8"
        )
    )
    system_keys = set(specification["design_spaces"]["system"])

    assert system_keys == {
        "cpu",
        "bus_type",
        "memory_total_kib",
        "memory_bank_size_kib",
        "memory_interleaved_ratio",
        "memory_placement",
        "dma_fifo_depth",
        "accelerator_fifo_depth",
        "accelerator_fifo_almost_full_margin",
    }

    individual = Individual.from_slots(
        id="all_system_keys",
        scenario="all_system_keys",
        design={
            "system": {
                "cpu": "cv32e20",
                "bus_type": "NtoM",
                "memory_total_kib": 256,
                "memory_bank_size_kib": 32,
                "memory_interleaved_ratio": 50,
                "memory_placement": "shared_data",
                "dma_fifo_depth": 8,
                "accelerator_fifo_depth": 8,
                "accelerator_fifo_almost_full_margin": 2,
            }
        },
        slots=[StageChoice(slot=0, stage="nop")],
    )
    composer = make_composer()
    package = composer.compose(individual)
    config = package.files["config/mcu-gen-config.py"]
    wrapper = composer.render_safa_integration(
        top_function="top",
        rtl_files=("top.v",),
        configuration=package.metadata["rendered_configuration"],
    )["hw/vendor/safa/rtl/safa_wrapper.sv"]

    assert 'CPU("cv32e20")' in config
    assert "XHeep(BusType.NtoM)" in config
    assert "memory_ss.add_ram_banks([32] * 4)" in config
    assert "LinkerSection.by_size(\"code\", 0, 0x00018000)" in config
    assert "fifo_depth=8" in config
    assert "FIFO_DEPTH = 8" in wrapper
    assert "FIFO_ALMOST_FULL_MARGIN = 2" in wrapper
    assert package.metadata["unbound_system_parameters"] == []


def test_gr_heep_composer_rejects_incompatible_interleaved_placement() -> None:
    individual = Individual.from_slots(
        id="bad_memory_placement",
        scenario="bad_memory_placement",
        design={
            "system": {
                "memory_total_kib": 256,
                "memory_bank_size_kib": 32,
                "memory_interleaved_ratio": 0,
                "memory_placement": "input_output_interleaved",
            }
        },
        slots=[StageChoice(slot=0, stage="nop")],
    )

    with pytest.raises(ComposerError, match="requires interleaved banks"):
        make_composer().compose(individual)


@pytest.mark.parametrize(
    ("system", "message"),
    (
        (
            {
                "memory_total_kib": 250,
                "memory_bank_size_kib": 32,
                "memory_interleaved_ratio": 50,
                "memory_placement": "shared_data",
            },
            "must be divisible",
        ),
        (
            {
                "memory_total_kib": 256,
                "memory_bank_size_kib": 32,
                "memory_interleaved_ratio": 100,
                "memory_placement": "shared_data",
            },
            "between 0 and 99",
        ),
        (
            {
                "memory_total_kib": 256,
                "memory_bank_size_kib": 32,
                "memory_placement": "shared_data",
            },
            "requires memory_total_kib",
        ),
        (
            {
                "memory_total_kib": 192,
                "memory_bank_size_kib": 16,
                "memory_interleaved_ratio": 50,
                "memory_placement": "shared_data",
            },
            "must be a power of two",
        ),
    ),
)
def test_gr_heep_composer_rejects_invalid_incremental_memory(
    system,
    message,
) -> None:
    individual = Individual.from_slots(
        id="bad_memory",
        scenario="bad_memory",
        design={"system": system},
        slots=[StageChoice(slot=0, stage="nop")],
    )

    with pytest.raises(ComposerError, match=message):
        make_composer().compose(individual)


def test_gr_heep_composer_rejects_unknown_binding_token() -> None:
    with pytest.raises(ComposerError, match="unknown tokens"):
        make_composer(parameter_bindings={"cpu": "UNKNOWN"})


def test_gr_heep_composer_rejects_unsafe_bound_value() -> None:
    composer = make_composer(parameter_bindings={"cpu": "CPU"})
    individual = Individual.from_slots(
        id="unsafe",
        scenario="unsafe",
        design={"system": {"cpu": 'cv32e20\nprint("injected")'}},
        slots=[StageChoice(slot=0, stage="nop")],
    )

    with pytest.raises(ComposerError, match="Invalid value"):
        composer.compose(individual)


def test_gr_heep_package_materializes_validated_overlay(tmp_path) -> None:
    package = make_composer().compose(make_individual())

    package.materialize(tmp_path)

    assert (tmp_path / "config/mcu-gen-config.py").is_file()
    assert (tmp_path / "sw/applications/genio_target/main.c").is_file()
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["package_type"] == "gr_heep_overlay"


def test_gr_heep_package_rejects_escaping_overlay_path(tmp_path) -> None:
    package = GRHeepConfigurationPackage(files={"../outside.txt": "bad"})

    with pytest.raises(ComposerError, match="Invalid GR-HEEP overlay path"):
        package.materialize(tmp_path)


def test_gr_heep_package_rejects_unowned_overlay_path(tmp_path) -> None:
    package = GRHeepConfigurationPackage(files={"Makefile": "bad"})

    with pytest.raises(ComposerError, match="not owned by GENIO"):
        package.materialize(tmp_path)


def test_gr_heep_package_rejects_symlink_escape(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "config").symlink_to(outside, target_is_directory=True)
    package = GRHeepConfigurationPackage(
        files={"config/mcu-gen-config.py": "bad"}
    )

    with pytest.raises(ComposerError, match="escapes through a symlink"):
        package.materialize(package_dir)


def test_gr_heep_composer_renders_dynamic_safa_integration() -> None:
    files = make_composer(
        configuration_defaults={
            "ACCELERATOR_FIFO_DEPTH": 8,
            "ACCELERATOR_FIFO_ALMOST_FULL_MARGIN": 2,
        }
    ).render_safa_integration(
        top_function="safa_accelerator",
        rtl_files=("safa_accelerator.v", "pipeline.sv", "types.vh"),
    )

    wrapper = files["hw/vendor/safa/rtl/safa_wrapper.sv"]
    core = files["hw/vendor/safa/hls_ip/hls_accelerator_component.core"]
    assert "safa_accelerator i_hls_top" in wrapper
    assert "parameter int unsigned FIFO_DEPTH = 8" in wrapper
    assert "parameter int unsigned FIFO_ALMOST_FULL_MARGIN = 2" in wrapper
    assert "toplevel: safa_accelerator" in core
    assert "      - safa_accelerator.v" in core
    assert "pipeline.sv: {file_type: systemVerilogSource}" in core
    assert "types.vh: {is_include_file: true}" in core
    assert "@HLS_TOP_MODULE@" not in wrapper
    assert "@HLS_TOP_MODULE@" not in core
    assert "@HLS_RTL_FILES@" not in core


@pytest.mark.parametrize(
    ("top_function", "rtl_files", "message"),
    (
        ("bad-top", ("top.v",), "Invalid HLS top module"),
        ("top", (), "at least one HLS RTL file"),
        ("top", ("../top.v",), "Invalid HLS RTL path"),
        ("top", ("top.cpp",), "Unsupported HLS RTL file"),
        ("top", ("top.v", "top.v"), "Duplicate HLS RTL file"),
    ),
)
def test_gr_heep_composer_rejects_invalid_safa_inputs(
    top_function,
    rtl_files,
    message,
) -> None:
    with pytest.raises(ComposerError, match=message):
        make_composer().render_safa_integration(
            top_function=top_function,
            rtl_files=rtl_files,
        )


def test_gr_heep_composer_renders_late_application_config() -> None:
    config = make_composer().render_application_config(
        {
            "INPUT_ROWS": 48,
            "INPUT_COLS": 48,
            "OUTPUT_ROWS": 24,
            "OUTPUT_COLS": 24,
            "INPUT_WORDS": 1728,
            "OUTPUT_WORDS": 144,
            "TIMEOUT_CYCLES": 100000,
        }
    )

    assert "#define GENIO_INPUT_ROWS 48" in config
    assert "#define GENIO_OUTPUT_COLS 24" in config
    assert "#define GENIO_INPUT_WORDS 1728" in config
    assert "#define GENIO_OUTPUT_WORDS 144" in config
    assert "#define GENIO_TIMEOUT_CYCLES 100000" in config
    assert "@" not in config


def test_gr_heep_composer_renders_complete_hls_artifact_overlay(tmp_path) -> None:
    top_path = tmp_path / "safa_accelerator.v"
    helper_path = tmp_path / "helper.v"
    top_path.write_text("module safa_accelerator; endmodule\n", encoding="utf-8")
    helper_path.write_text("module helper; endmodule\n", encoding="utf-8")
    artifact = HLSRTLArtifact(
        name="rtl_hls_synthesis",
        producer="hls_image_pipeline_synthesis",
        individual_id="individual",
        origin="hls_synthesis",
        top_function="safa_accelerator",
        verilog_paths=(top_path, helper_path),
        metadata={
            "interface": "safa_fifo",
            "input_rows": 48,
            "input_cols": 48,
            "output_rows": 48,
            "output_cols": 48,
            "input_words": 1728,
            "output_words": 576,
        },
    )

    overlay = make_composer().render_hls_artifact_overlay(artifact)

    assert overlay["hw/vendor/safa/hls_ip/safa_accelerator.v"] == (
        "module safa_accelerator; endmodule\n"
    )
    assert overlay["hw/vendor/safa/hls_ip/helper.v"] == (
        "module helper; endmodule\n"
    )
    assert "safa_accelerator i_hls_top" in overlay[
        "hw/vendor/safa/rtl/safa_wrapper.sv"
    ]
    app_config = overlay["sw/applications/genio_target/genio_app_config.h"]
    assert "#define GENIO_INPUT_WORDS 1728" in app_config
    assert "#define GENIO_OUTPUT_WORDS 576" in app_config


def test_gr_heep_composer_embeds_dataset_image_in_main_header(tmp_path) -> None:
    cv = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image_path = tmp_path / "sample.png"
    image = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    assert cv.imwrite(str(image_path), image)
    top_path = tmp_path / "top.v"
    top_path.write_text("module top; endmodule\n", encoding="utf-8")
    artifact = HLSRTLArtifact(
        name="rtl_hls_synthesis",
        producer="hls_image_pipeline_synthesis",
        individual_id="individual",
        origin="hls_synthesis",
        top_function="top",
        verilog_paths=(top_path,),
        metadata={
            "interface": "safa_fifo",
            "input_rows": 1,
            "input_cols": 2,
            "output_rows": 1,
            "output_cols": 2,
            "input_words": 2,
            "output_words": 1,
        },
    )

    overlay = make_composer().render_hls_artifact_overlay(
        artifact,
        image_path=image_path,
    )

    header = overlay["sw/applications/genio_target/main.h"]
    assert "0x04030201u, 0x00000605u," in header
    assert "@IMAGE_WORDS@" not in header


def test_gr_heep_composer_rejects_non_safa_hls_artifact(tmp_path) -> None:
    top_path = tmp_path / "top.v"
    top_path.write_text("module top; endmodule\n", encoding="utf-8")
    artifact = HLSRTLArtifact(
        name="rtl_hls_synthesis",
        producer="hls_image_pipeline_synthesis",
        individual_id="individual",
        origin="hls_synthesis",
        top_function="top",
        verilog_paths=(top_path,),
        metadata={"interface": "fifo"},
    )

    with pytest.raises(ComposerError, match="using 'safa_fifo'"):
        make_composer().render_hls_artifact_overlay(artifact)

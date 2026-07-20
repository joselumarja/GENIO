from dataclasses import replace
from pathlib import Path

import pytest

from genio import (
    Composer,
    ComposerError,
    HLSExecutionPackage,
    HLSImagePipelineComposer,
    Individual,
    PythonExecutionPackage,
    PythonImagePipelineComposer,
    SearchSpace,
    StageChoice,
    StageDefinitionNotFoundError,
)


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS_PATH = ROOT / "search_space/stages/definitions"


class DummyComposer(Composer):
    def compose(self, individual: Individual) -> dict:
        return {
            "metadata": self.artifact_metadata(individual),
            "stages": [
                {
                    "slot": choice.slot,
                    "stage": choice.stage,
                    "parameters": dict(choice.parameters),
                    "definition_id": definition["id"],
                }
                for choice, definition in self.active_stage_definitions(individual)
            ],
        }


def test_base_composer_skips_nop_and_composes_non_nop_steps():
    composer = DummyComposer(DEFINITIONS_PATH)
    individual = Individual.from_slots(
        id="individual_001",
        scenario="simple_threshold_pipeline",
        genotype=(0, 1),
        search_index=1,
        slots=[
            StageChoice(slot=0, stage="nop"),
            StageChoice(
                slot=1,
                stage="threshold",
                parameters={
                    "threshold": 120,
                    "maxval": 255,
                    "threshold_type": "binary",
                },
            ),
        ],
    )

    artifact = composer.compose(individual)

    assert artifact["metadata"] == {
        "composer": "DummyComposer",
        "scenario": "simple_threshold_pipeline",
        "search_index": 1,
    }
    assert artifact["stages"] == [
        {
            "slot": 1,
            "stage": "threshold",
            "parameters": {
                "threshold": 120,
                "maxval": 255,
                "threshold_type": "binary",
            },
            "definition_id": "threshold",
        }
    ]


def test_base_composer_exposes_active_choices_without_nop():
    composer = DummyComposer(DEFINITIONS_PATH)
    individual = Individual.from_slots(
        id="individual_001",
        scenario="simple_threshold_pipeline",
        design={"hls": {"npc": "XF_NPPC2", "pipeline_ii": 1, "use_uram": True}},
        slots=[
            StageChoice(slot=0, stage="nop"),
            StageChoice(
                slot=1,
                stage="threshold",
                parameters={
                    "threshold": 120,
                    "maxval": 255,
                    "threshold_type": "binary",
                },
            ),
        ],
    )

    assert list(composer.active_choices(individual)) == [
        StageChoice(
            slot=1,
            stage="threshold",
            parameters={
                "threshold": 120,
                "maxval": 255,
                "threshold_type": "binary",
            },
        )
    ]


def test_base_composer_exposes_active_stage_definitions():
    composer = DummyComposer(DEFINITIONS_PATH)
    individual = Individual.from_slots(
        id="individual_001",
        scenario="simple_threshold_pipeline",
        design={"hls": {"npc": "XF_NPPC2", "pipeline_ii": 1, "use_uram": True}},
        slots=[
            StageChoice(
                slot=0,
                stage="threshold",
                parameters={
                    "threshold": 120,
                    "maxval": 255,
                    "threshold_type": "binary",
                },
            ),
        ],
    )

    active = list(composer.active_stage_definitions(individual))

    assert active[0][0] == StageChoice(
        slot=0,
        stage="threshold",
        parameters={
            "threshold": 120,
            "maxval": 255,
            "threshold_type": "binary",
        },
    )
    assert active[0][1]["id"] == "threshold"


def test_base_composer_raises_for_unknown_stage():
    composer = DummyComposer(DEFINITIONS_PATH)
    individual = Individual.from_slots(
        id="individual_001",
        scenario="unknown_stage_pipeline",
        slots=[StageChoice(slot=0, stage="missing_stage")],
    )

    with pytest.raises(StageDefinitionNotFoundError, match="missing_stage"):
        composer.compose(individual)


def test_python_execution_package_materializes_files(tmp_path):
    package = PythonExecutionPackage(
        entrypoint="pipeline.py:run",
        files={
            "pipeline.py": "def run(image):\n    return image\n",
            "config/settings.py": "VALUE = 1\n",
        },
        requirements=("numpy", "opencv-python"),
        metadata={"scenario": "test"},
    )

    package_dir = package.materialize(tmp_path / "package")

    assert package_dir == tmp_path / "package"
    assert (package_dir / "pipeline.py").read_text(encoding="utf-8") == (
        "def run(image):\n    return image\n"
    )
    assert (package_dir / "config/settings.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert (package_dir / "requirements.txt").read_text(encoding="utf-8") == (
        "numpy\nopencv-python\n"
    )


def test_hls_execution_package_materializes_files_and_metadata(tmp_path):
    package = HLSExecutionPackage(
        entrypoint="hls_config.cfg",
        files={
            "hls_config.cfg": "part=xa7a100tcsg324-1I\n\n[hls]\nsyn.top=image_pipeline_top\n",
            "src/pipeline.cpp": "void image_pipeline_top() {}\n",
            "include/pipeline.hpp": "#pragma once\n",
        },
        metadata={
            "package_type": "hls_image_pipeline",
            "top_function": "image_pipeline_top",
            "source_files": ["src/pipeline.cpp"],
        },
    )

    package_dir = package.materialize(tmp_path / "package")

    assert package_dir == tmp_path / "package"
    assert package.entrypoint == "hls_config.cfg"
    assert (package_dir / "hls_config.cfg").read_text(encoding="utf-8").startswith(
        "part=xa7a100tcsg324-1I"
    )
    assert (package_dir / "src/pipeline.cpp").read_text(encoding="utf-8") == (
        "void image_pipeline_top() {}\n"
    )
    assert '"top_function": "image_pipeline_top"' in (
        package_dir / "metadata.json"
    ).read_text(encoding="utf-8")


def test_python_image_pipeline_composer_generates_execution_package():
    composer = PythonImagePipelineComposer(DEFINITIONS_PATH)
    individual = Individual.from_slots(
        id="individual_001",
        scenario="simple_threshold_pipeline",
        genotype=(0, 1),
        search_index=1,
        slots=[
            StageChoice(slot=0, stage="bgr_to_gray"),
            StageChoice(
                slot=1,
                stage="threshold",
                parameters={
                    "threshold": 120,
                    "maxval": 255,
                    "threshold_type": "binary_inv",
                },
            ),
        ],
    )

    package = composer.compose(individual)
    source = package.files["pipeline.py"]

    assert package.entrypoint == "pipeline.py:run"
    assert package.requirements == ("opencv-python", "numpy")
    assert package.metadata["implementation_source"] == "opencv"
    assert "def run(image):" in source
    assert "stage_0_output = cv.cvtColor(current, cv.COLOR_BGR2GRAY)" in source
    assert (
        "_, stage_1_output = cv.threshold(current, 120, 255, cv.THRESH_BINARY_INV)"
        in source
    )
    assert "return current" in source


def test_python_image_pipeline_composer_maps_morphology_shape_enums():
    composer = PythonImagePipelineComposer(DEFINITIONS_PATH)
    individual = Individual.from_slots(
        id="individual_001",
        scenario="morphology_pipeline",
        slots=[
            StageChoice(
                slot=0,
                stage="erode",
                parameters={
                    "kernel_shape": "rect",
                    "kernel_rows": 3,
                    "kernel_cols": 3,
                    "iterations": 1,
                },
            ),
        ],
    )

    package = composer.compose(individual)
    source = package.files["pipeline.py"]

    assert "kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))" in source
    assert "'rect'" not in source


def test_hls_image_pipeline_composer_generates_fifo_package():
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
        rows=64,
        cols=128,
    )
    individual = Individual.from_slots(
        id="individual_001",
        scenario="simple_threshold_pipeline",
        design={"hls": {"npc": "XF_NPPC2", "pipeline_ii": 1, "use_uram": True}},
        slots=[
            StageChoice(slot=0, stage="bgr_to_gray"),
            StageChoice(
                slot=1,
                stage="threshold",
                parameters={
                    "threshold": 120,
                    "maxval": 255,
                    "threshold_type": "binary",
                },
            ),
        ],
    )

    package = composer.compose(individual)
    source = package.files["src/pipeline.cpp"]

    assert isinstance(package, HLSExecutionPackage)
    assert package.entrypoint == "hls_config.cfg"
    assert "part=" in package.files["hls_config.cfg"]
    assert "package.output.format=ip_catalog" in package.files["hls_config.cfg"]
    assert "package.output.syn=false" in package.files["hls_config.cfg"]
    assert package.metadata["top_function"] == "top"
    assert package.metadata["interface"] == "fifo"
    assert package.metadata["include_dirs"] == ("include",)
    assert package.metadata["required_backend_resources"] == ("vitis_libraries_path",)
    assert package.metadata["output_type"] == "XF_8UC1"
    assert "#define ROWS 64" in source
    assert "#define COLS 128" in source
    assert "#define NPC XF_NPPC2" in source
    assert "#pragma HLS PIPELINE II=1" in source
    assert "BIND_STORAGE variable=input_mat" not in source
    assert "npc=" not in package.files["hls_config.cfg"]
    assert "pipeline_ii=" not in package.files["hls_config.cfg"]
    assert "use_uram=" not in package.files["hls_config.cfg"]
    assert "#include \"imgproc/xf_cvt_color.hpp\"" in source
    assert "#include \"imgproc/xf_threshold.hpp\"" in source
    assert "hls::stream<ap_uint<XF_PIXELWIDTH(XF_8UC1, XF_NPPC2)>>& output_fifo" in source
    assert "int rows" not in source
    assert "int cols" not in source
    assert "port=rows" not in source
    assert "port=cols" not in source
    assert "xf::cv::Mat<TYPE, ROWS, COLS, NPC> input_mat(ROWS, COLS);" in source
    assert "xf::cv::Mat<XF_8UC1, 64, 128, XF_NPPC2> stage_0_output(64, 128);" in source
    assert "xf::cv::bgr2gray<XF_8UC3, XF_8UC1, 64, 128, XF_NPPC2>(input_mat, stage_0_output);" in source
    assert "xf::cv::Threshold<XF_THRESHOLD_TYPE_BINARY, XF_8UC1, 64, 128, XF_NPPC2>(stage_0_output, stage_1_output, 120, 255);" in source
    assert "xfMat2fifo<XF_8UC1, 64, 128, XF_NPPC2>(stage_1_output, output_fifo);" in source


def test_hls_image_pipeline_composer_generates_axi_stream_package():
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
        interface="axi_stream",
        rows=32,
        cols=32,
        axi_width=64,
        axi_user_width=1,
        axi_id_width=0,
        axi_dest_width=0,
    )
    individual = Individual.from_slots(
        id="individual_001",
        scenario="gray_pipeline",
        slots=[StageChoice(slot=0, stage="bgr_to_gray")],
    )

    package = composer.compose(individual)
    source = package.files["src/pipeline.cpp"]

    assert package.metadata["interface"] == "axi_stream"
    assert "include/xf_axi_stream_utils.hpp" in package.files
    assert "#define AXI_WIDTH 64" in source
    assert "#define AXI_USER_WIDTH 1" in source
    assert "int rows" not in source
    assert "int cols" not in source
    assert "port=rows" not in source
    assert "port=cols" not in source
    assert "axiStream2xfMat<AXI_WIDTH, TYPE, ROWS, COLS, NPC" in source
    assert "xfMat2axiStream<AXI_WIDTH, XF_8UC1, 32, 32, XF_NPPC1" in source


@pytest.mark.parametrize(
    (
        "stage",
        "parameters",
        "image_type",
        "npc",
        "expected_output_type",
        "expected_output_npc",
    ),
    (
        (
            "canny",
            {"low_threshold": 50, "high_threshold": 150},
            "XF_8UC1",
            "XF_NPPC1",
            "XF_2UC1",
            "XF_NPPC32",
        ),
        (
            "channel_extract",
            {"channel": 0},
            "XF_8UC3",
            "XF_NPPC1",
            "XF_8UC1",
            "XF_NPPC1",
        ),
        (
            "channel_extract",
            {"channel": 0},
            "XF_16UC4",
            "XF_NPPC1",
            "XF_16UC1",
            "XF_NPPC1",
        ),
        (
            "convert_scale_abs",
            {"alpha": 1.0, "beta": 0.0},
            "XF_8UC1",
            "XF_NPPC1",
            "XF_8UC1",
            "XF_NPPC1",
        ),
        (
            "remap",
            {"interpolation": "nearest"},
            "XF_8UC3",
            "XF_NPPC2",
            "XF_8UC3",
            "XF_NPPC2",
        ),
        (
            "scharr",
            {"border_type": "XF_BORDER_CONSTANT"},
            "XF_8UC1",
            "XF_NPPC8",
            "XF_16SC1",
            "XF_NPPC8",
        ),
        (
            "sobel",
            {"filter_width": 3, "border_type": "XF_BORDER_CONSTANT"},
            "XF_8UC3",
            "XF_NPPC1",
            "XF_16SC3",
            "XF_NPPC1",
        ),
    ),
)
def test_hls_image_pipeline_composer_resolves_output_type_and_npc(
    stage,
    parameters,
    image_type,
    npc,
    expected_output_type,
    expected_output_npc,
):
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
        rows=64,
        cols=128,
        image_type=image_type,
        npc=npc,
    )
    individual = Individual.from_slots(
        id=f"{stage}_individual",
        scenario=f"{stage}_pipeline",
        slots=[StageChoice(slot=0, stage=stage, parameters=parameters)],
    )

    package = composer.compose(individual)
    source = package.files["src/pipeline.cpp"]

    assert package.metadata["output_type"] == expected_output_type
    assert package.metadata["npc"] == expected_output_npc
    assert (
        f"xf::cv::Mat<{expected_output_type}, 64, 128, {expected_output_npc}> "
        "stage_0_output(64, 128);"
    ) in source
    assert "@OUT_TYPE" not in source
    assert "@OUT_NPC" not in source
    assert (
        f"xfMat2fifo<{expected_output_type}, 64, 128, {expected_output_npc}>"
    ) in source


@pytest.mark.parametrize(
    ("stage", "parameters", "image_type", "npc", "error_match"),
    (
        (
            "canny",
            {"low_threshold": 50, "high_threshold": 150},
            "XF_8UC3",
            "XF_NPPC1",
            "does not support input type",
        ),
        (
            "canny",
            {"low_threshold": 50, "high_threshold": 150},
            "XF_8UC1",
            "XF_NPPC2",
            "does not support NPC",
        ),
        (
            "channel_extract",
            {"channel": 0},
            "XF_8UC1",
            "XF_NPPC1",
            "does not support input type",
        ),
        (
            "convert_scale_abs",
            {"alpha": 1.0, "beta": 0.0},
            "XF_8UC3",
            "XF_NPPC1",
            "does not support input type",
        ),
        (
            "remap",
            {"interpolation": "nearest"},
            "XF_16UC1",
            "XF_NPPC1",
            "does not support input type",
        ),
        (
            "sobel",
            {"filter_width": 3, "border_type": "XF_BORDER_CONSTANT"},
            "XF_8UC1",
            "XF_NPPC2",
            "does not support NPC",
        ),
    ),
)
def test_hls_image_pipeline_composer_rejects_invalid_output_state_inputs(
    stage,
    parameters,
    image_type,
    npc,
    error_match,
):
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
        rows=64,
        cols=128,
        image_type=image_type,
        npc=npc,
    )
    individual = Individual.from_slots(
        id=f"invalid_{stage}_individual",
        scenario=f"invalid_{stage}_pipeline",
        slots=[StageChoice(slot=0, stage=stage, parameters=parameters)],
    )

    with pytest.raises(ComposerError, match=error_match):
        composer.compose(individual)


def test_hls_image_pipeline_composer_requires_canny_columns_divisible_by_32():
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
        rows=64,
        cols=130,
        image_type="XF_8UC1",
        npc="XF_NPPC1",
    )
    individual = Individual.from_slots(
        id="invalid_canny_columns",
        scenario="canny_pipeline",
        slots=[
            StageChoice(
                slot=0,
                stage="canny",
                parameters={"low_threshold": 50, "high_threshold": 150},
            )
        ],
    )

    with pytest.raises(ComposerError, match="divisible by 32"):
        composer.compose(individual)


def test_hls_image_pipeline_composer_substitutes_use_uram_token():
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
        rows=64,
        cols=128,
    )
    individual = Individual.from_slots(
        id="individual_001",
        scenario="resize_pipeline",
        design={"hls": {"use_uram": True}},
        slots=[
            StageChoice(
                slot=0,
                stage="resize",
                parameters={"out_rows": 32, "out_cols": 64, "interpolation": "area"},
            ),
        ],
    )

    package = composer.compose(individual)
    source = package.files["src/pipeline.cpp"]

    assert (
        "xf::cv::resize<XF_INTERPOLATION_AREA, XF_8UC3, 64, 128, 32, 64, "
        "XF_NPPC1, true, 2>"
    ) in source
    assert '#include "imgproc/xf_resize.hpp"' in source
    assert "@USE_URAM" not in source
    assert "use_uram=" not in package.files["hls_config.cfg"]


def test_hls_image_pipeline_composer_uses_repository_custom_header():
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
        rows=64,
        cols=128,
        image_type="XF_8UC1",
    )
    individual = Individual.from_slots(
        id="custom_identity",
        scenario="custom_pipeline",
        slots=[StageChoice(slot=0, stage="custom_identity")],
    )

    package = composer.compose(individual)
    source = package.files["src/pipeline.cpp"]

    assert '#include "genio/identity.hpp"' in source
    assert (
        "genio::identity<XF_8UC1, 64, 128, XF_NPPC1>"
        "(input_mat, stage_0_output);"
    ) in source


def test_hls_image_pipeline_composer_uses_2023_resize_signature():
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
        vitis_version="2023.1",
        rows=64,
        cols=128,
    )
    individual = Individual.from_slots(
        id="resize_2023",
        scenario="resize_pipeline",
        design={"hls": {"use_uram": False}},
        slots=[
            StageChoice(
                slot=0,
                stage="resize",
                parameters={"out_rows": 32, "out_cols": 64, "interpolation": "area"},
            )
        ],
    )

    package = composer.compose(individual)
    source = package.files["src/pipeline.cpp"]

    assert (
        "xf::cv::resize<XF_INTERPOLATION_AREA, XF_8UC3, 64, 128, 32, 64, "
        "XF_NPPC1, 2>"
    ) in source
    assert "XF_NPPC1, false, 2" not in source
    assert package.metadata["vitis_version"] == "2023.1"


def test_hls_image_pipeline_composer_rejects_unsupported_resize_version():
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
        vitis_version="2024.1",
        rows=64,
        cols=128,
    )
    individual = Individual.from_slots(
        id="resize_unsupported",
        scenario="resize_pipeline",
        slots=[
            StageChoice(
                slot=0,
                stage="resize",
                parameters={"out_rows": 32, "out_cols": 64, "interpolation": "area"},
            )
        ],
    )

    with pytest.raises(ComposerError, match="does not support Vitis '2024.1'"):
        composer.compose(individual)


def test_hls_token_replacement_is_exact_and_non_recursive():
    rendered = HLSImagePipelineComposer._replace_tokens(
        "@TYPE @ROWS @TYPE_EXTRA K_ROWS EXTRA_K_ROWS",
        {
            "@TYPE": "@ROWS",
            "@ROWS": "64",
            "K_ROWS": "3",
        },
    )

    assert rendered == "@ROWS 64 @TYPE_EXTRA 3 EXTRA_K_ROWS"


@pytest.mark.parametrize(
    "include",
    ("/absolute/header.hpp", "../header.hpp", "bad\\header.hpp", 'bad"header.hpp'),
)
def test_hls_image_pipeline_composer_rejects_non_portable_includes(include):
    with pytest.raises(ComposerError, match="invalid portable include"):
        HLSImagePipelineComposer._implementation_includes(
            [include],
            stage="custom_identity",
        )


def test_hls_image_pipeline_composer_renders_insect_segmentation_sample():
    search_space = SearchSpace(
        ROOT / "search_space/tests/insect_segmentation_pipeline.json",
        DEFINITIONS_PATH,
    )
    individual = search_space.from_index(0)
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
        rows=2160,
        cols=3840,
    )

    package = composer.compose(individual)
    source = package.files["src/pipeline.cpp"]

    assert "xf::cv::resize" in source
    assert "xf::cv::bgr2gray" in source
    assert "#include \"imgproc/xf_duplicateimage.hpp\"" in source
    assert "xf::cv::duplicateMat<XF_8UC1" in source
    assert "xf::cv::equalizeHist" in source
    assert "xf::cv::equalizeHist<XF_8UC1, 1080, 1920, XF_NPPC1, false>" in source
    assert "xf::cv::Threshold" in source
    assert "unsigned char kernel_" in source
    assert "XF_INTERPOLATION_AREA" in source
    assert "XF_SHAPE_RECT" in source
    assert "@" not in source
    assert package.metadata["output_type"] == "XF_8UC1"
    assert package.metadata["npc"] == "XF_NPPC1"
    assert package.metadata["hls_design"]["npc"] == "XF_NPPC1"


def test_hls_image_pipeline_composer_uses_npc_from_design_space():
    search_space = SearchSpace(
        ROOT / "search_space/tests/insect_segmentation_pipeline.json",
        DEFINITIONS_PATH,
    )
    hls_design = dict(search_space.scenario.design_spaces["hls"])
    hls_design["npc"] = ("XF_NPPC1", "XF_NPPC2")
    search_space = SearchSpace.from_scenario(
        replace(
            search_space.scenario,
            design_spaces={
                **search_space.scenario.design_spaces,
                "hls": hls_design,
            },
        )
    )
    genotype = [0 for _ in search_space.genotype_lengths]
    npc_gene_index = len(search_space.slot_lengths) + tuple(
        search_space.scenario.design_spaces["hls"]
    ).index("npc")
    genotype[npc_gene_index] = 1
    individual = search_space.from_genotype(tuple(genotype))
    composer = HLSImagePipelineComposer(
        DEFINITIONS_PATH,
        templates_path=ROOT / "hls_templates/vitis_vision_image_pipeline",
        rows=2160,
        cols=3840,
    )

    package = composer.compose(individual)
    source = package.files["src/pipeline.cpp"]

    assert individual.design["hls"]["npc"] == "XF_NPPC2"
    assert package.metadata["npc"] == "XF_NPPC2"
    assert "#define NPC XF_NPPC2" in source

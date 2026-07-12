from pathlib import Path
from random import Random

import pytest

from genio.search_space import SearchScenarioSpec, SlotSpec
from genio import SearchSpace, StageChoice


ROOT = Path(__file__).resolve().parents[1]
TESTS_PATH = ROOT / "search_space/tests"
DEFINITIONS_PATH = ROOT / "search_space/stages/definitions"

TEST = "insect_segmentation_pipeline.json"

search_space = SearchSpace(TESTS_PATH / TEST, DEFINITIONS_PATH)

individual = search_space.sample()

index = search_space.to_index(individual)

genotype = search_space.to_genotype(individual)

index_compare = search_space.genotype_to_index(genotype)

population1 = search_space.sample_population(10, unique=True)
population2 = search_space.sample_balanced_population(10, unique=True)

slot, gene = search_space.sample_slot(5)

slot_balanced, gene_balanced = search_space.sample_slot_balanced(5)

print("hola")

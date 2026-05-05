from pathlib import Path

import numpy as np
import petab_select

import amici
import petab
import pypesto
import pypesto.petab
from path_config import PETAB_DIR, MODEL_SELECTION_RESULTS_DIR

petab_select_problem = petab_select.Problem.from_yaml(
    PETAB_DIR / "model_selection" / "petab_select_mixed_noise" / "petab_select_problem_famos.yaml"
)

n_model_starts = int(128)
n_parameters = 16

rng = np.random.default_rng()
indices_array = rng.integers(0, 1, size=(n_model_starts, n_parameters), endpoint=True)

unique_indices_array, counts = np.unique(indices_array, axis=0, return_counts=True)
assert unique_indices_array.shape == indices_array.shape, f"Not unique indices; array shapes differ. unique: {unique_indices_array.shape}. all: {indices_array.shape}."
assert (counts == 1).all(), "Not unique indices; not all counts are one."


output_path = MODEL_SELECTION_RESULTS_DIR / "input" / "famos_mixed_noise" / "predecessor_models"
output_path.mkdir(exist_ok=False, parents=True)

np.savetxt(output_path.parent / "indices.txt", indices_array, delimiter='\t')

models = []
for indices in indices_array:
    model = petab_select_problem.model_space.model_subspaces['M'].indices_to_model(indices)
    models.append(model)

for model_index, model in enumerate(models):
    model.to_yaml(output_path / f"{model_index}.yaml")


# my old pre-model selection routine
# petab_select_problem = petab_select.Problem.from_yaml('petab/full_selection_log/petab_select_problem_famos.yaml')

# candidate_space = petab_select.BruteForceCandidateSpace()

# limit = np.inf

# models = petab_select_problem.model_space.search(
#     candidate_space=candidate_space,
#     limit=limit,
# )

# output_path = Path(__file__).resolve().parent / "input" / "famos"

# rng = np.random.default_rng(123)
# model_indices = rng.integers(0, 4096, size=48)

# for j, i in enumerate(model_indices):
#     models[i].to_yaml(output_path / f"{j}.yaml")


# compile the full model once


yaml_file = PETAB_DIR / "model_selection" / "petab_select_mixed_noise" / "full_petab_problem.yaml"

petab_problem = petab.Problem.from_yaml(yaml_file)

pypesto_importer = pypesto.petab.PetabImporter(petab_problem, hierarchical=True)
model = pypesto_importer.create_model(force_compile=True)


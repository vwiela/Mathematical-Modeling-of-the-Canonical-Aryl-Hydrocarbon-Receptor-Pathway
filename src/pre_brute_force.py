from pathlib import Path

import numpy as np
import petab_select
import itertools

import amici
import petab
import pypesto
import pypesto.petab
from path_config import PETAB_DIR, MODEL_SELECTION_RESULTS_DIR

petab_select_problem = petab_select.Problem.from_yaml(
    PETAB_DIR / "model_selection" / "petab_select_all_ligands_brute_force_mixed_noise" / "petab_select_problem_brute_force.yaml"
)

# def generate_vectors(length):
#     # Generate vectors with exactly one 1
#     one_ones = [list(x) for x in itertools.combinations(range(length), 1)]
    
#     # Generate vectors with exactly two 1s
#     two_ones = [list(x) for x in itertools.combinations(range(length), 2)]
    
#     vectors = []
    
#     # Create binary vectors for one 1
#     for positions in one_ones:
#         vector = [0] * length
#         for pos in positions:
#             vector[pos] = 1
#         vectors.append(vector)
    
#     # Create binary vectors for two 1s
#     for positions in two_ones:
#         vector = [0] * length
#         for pos in positions:
#             vector[pos] = 1
#         vectors.append(vector)
    
#     return vectors

# # Get all vectors of length 16 with exactly 1 or 2 ones
# indices_array = generate_vectors(16)

output_path = MODEL_SELECTION_RESULTS_DIR / "input" / "all_ligands_brute_force_mixed_noise" / "predecessor_models"
output_path.mkdir(exist_ok=False, parents=True)

# np.savetxt(output_path.parent / "indices.txt", indices_array, delimiter='\t')

# models = []
# for indices in indices_array:
#     model = petab_select_problem.model_space.model_subspaces['M'].indices_to_model(indices)
#     models.append(model)

# for model_index, model in enumerate(models):
#     model.to_yaml(output_path / f"{model_index}.yaml")


#pre-model selection routine for all ligands, where each model is in a different subspace.
for subsp_indx, subsp in petab_select_problem.model_space.model_subspaces.items():
    subsp_number = subsp_indx.split("M")[1]
    list(subsp.get_models([k for k,v in subsp.parameters.items() if v ==["estimate"]]))[0].to_yaml(output_path / f"{subsp_number}.yaml")


# compile the full model once


yaml_file = PETAB_DIR / "model_selection" / "petab_select_all_ligands_brute_force_mixed_noise" / "full_petab_problem.yaml"

petab_problem = petab.Problem.from_yaml(yaml_file)

pypesto_importer = pypesto.petab.PetabImporter(petab_problem, hierarchical=False)
model = pypesto_importer.create_model(force_compile=True)


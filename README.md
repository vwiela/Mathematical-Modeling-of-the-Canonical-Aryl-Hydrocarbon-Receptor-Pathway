# Mathematical Modeling of the Canonical Aryl Hydrocarbon Receptor Pathway

This repository contains code, PEtab problems, experimental data, and precomputed result files accompanying the project and publication **"Mathematical Modeling of the Canonical Aryl Hydrocarbon Receptor Pathway"**. The preprint is available on bioRxiv.

The repository is intended to support two reproduction workflows:

1. Recreate the figures and analyses from the included result files.
2. Rerun model calibration and model selection jobs, including the HPC scripts used for the publication.

## Environment

The analyses were run with **Python 3.10.12** and the package versions listed in [requirements.txt](requirements.txt).

From the repository root, create and activate an environment, then install dependencies:

```bash
python3.10 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Some scripts compile AMICI models from PEtab problems. For these steps, a working C/C++ build toolchain and CMake are required in addition to the Python packages.

## Repository Layout

- [data/](data): Raw experimental data used for model construction and visualization.
- [petab/](petab): PEtab problems for individual ligand fits, the joint all-ligand model, and PEtab Select model-selection problems.
- [results/](results): Precomputed optimization/model-selection results and generated figures used by the notebooks.
- [src/](src): Python helper functions and scripts used for calibration, model selection, and postprocessing.
- [src/input/](src/input): YAML input models used as starting points for PEtab Select/model-selection jobs.
- [src/best_models_scripts/](src/best_models_scripts): Scripts for additional optimization/evaluation of selected model-selection candidates.
- [Data_and_model_fittings.ipynb](Data_and_model_fittings.ipynb): Notebook for loading the data, visualizing ligand-specific and joint model fits, and regenerating the corresponding figures.
- [Model_selection.ipynb](Model_selection.ipynb): Notebook for loading and visualizing model-selection results.

## Main Reproduction Path

For most readers, the recommended workflow is to use the precomputed results in [results/](results) and rerun the notebooks from the repository root.

1. Install the environment described above.
2. Start Jupyter from the repository root.
3. Run [Data_and_model_fittings.ipynb](Data_and_model_fittings.ipynb) to reproduce data/model-fit visualizations.
4. Run [Model_selection.ipynb](Model_selection.ipynb) to reproduce model-selection visualizations.

The notebooks use repository-relative paths. They expect the main folders `data/`, `petab/`, `results/`, and `src/` to remain in the layout shown above.

## Rerunning Optimizations

The scripts in [src/](src) were used for the computationally expensive optimization and model-selection runs. They are primarily intended for cluster/HPC use and may require substantial runtime.

Important scripts include:

- [src/optimize_mixed_noise.py](src/optimize_mixed_noise.py): Multistart optimization for selected PEtab problems.
- [src/optimize_mixed_noise_sacess.py](src/optimize_mixed_noise_sacess.py): saCeSS-based optimization for mixed-noise models.
- [src/pre_brute_force.py](src/pre_brute_force.py): Generates predecessor-model YAML files for brute-force model selection.
- [src/brute_force_model_selection.py](src/brute_force_model_selection.py): Runs brute-force PEtab Select model selection jobs.
- [src/post_brute_force.py](src/post_brute_force.py): Postprocesses brute-force model-selection results.
- [src/evaluate_best_models.py](src/evaluate_best_models.py): Evaluates selected/best model-selection candidates.

Most model-selection scripts use `SLURM_ARRAY_TASK_ID` or a command-line index to select the job-specific model. For example:

```bash
python src/brute_force_model_selection.py --script-index 1
```

For cluster runs, output and temporary directories can be overridden with environment variables, for example:

```bash
export AHR_MODEL_SELECTION_OUTPUT_DIR=/path/to/output
export AHR_SACESS_TMPDIR=/path/to/tmp
python src/brute_force_model_selection.py --script-index 1
```

If these variables are not set, outputs are written below [results/](results) using repository-relative paths.

## Result Folders

- [results/bmm_model/](results/bmm_model): pyPESTO/AMICI result files from calibrating the model to individual ligands and to all ligands jointly.
- [results/figures/](results/figures): Figures generated from the data/model-fitting notebooks.
- [results/model_selection/](results/model_selection): Model-selection result files, including brute-force and FAMoS-related outputs.

Large result files may be stored externally for archival purposes. If a result file is missing locally, retrieve the corresponding archived results before rerunning the notebooks.

## Notes

- Run notebooks and scripts from the repository root unless stated otherwise.
- The PEtab problems are the canonical source for model definitions, observables, measurements, conditions, and parameter tables.
- The HPC scripts are included for transparency and reproducibility, but the notebooks are the most direct way to reproduce the publication figures from the included results.

from functools import partial
import logging
import os
from pathlib import Path
import pandas as pd
import math

import amici
import pypesto
import pypesto.petab
import pypesto.hierarchical
import pypesto.optimize as optimize
import pypesto.visualize as visualize
import fides
import petab
import petab_select
import os
import numpy as np
import libsbml
import time
import datetime
from pypesto.optimize.ess import SacessFidesFactory
from pypesto.select import SacessMinimizeMethod
from pypesto.store import save_to_hdf5
from pypesto.store import read_from_hdf5
from pathlib import Path
from scipy.stats import chi2
import dill
import logging
import amici.swig_wrappers

import argparse

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PETAB_DIR = PROJECT_ROOT / "petab"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_SELECTION_DIR = RESULTS_DIR / "model_selection" / "all_ligands_sacess_brute_force_mixed_noise"
BEST_MODEL_OPTIM_DIR = MODEL_SELECTION_DIR / "best_models_optim"
SACESS_TMP_DIR = RESULTS_DIR / "tmp" / "sacess"

parser = argparse.ArgumentParser()
parser.add_argument("--job-index", type=int, default=-1)
parser.add_argument("--script-index", type=int, default=-1)
args = parser.parse_args()

# reduce amici output such that the log-files don't explode.
amici.swig_wrappers.logger.setLevel(logging.CRITICAL)

# if do several runs.
pypesto.logging.log(level=logging.ERROR, name="pypesto.petab", console=True)

job_array_id = os.environ.get('SLURM_ARRAY_TASK_ID', 0)

if __name__ == '__main__':

    def objective_customizer(objective):
        objective.amici_solver.setRelativeTolerance(1e-12)
        objective.amici_solver.setAbsoluteTolerance(1e-8)
        objective.amici_solver.setMaxSteps(10**6)
        objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod_adjoint)
        pass

    root_path = SCRIPT_DIR

#    models_to_sample = ['100',
#	 '11',
#	 '114',
#	 '117',
#	 '318',
#	 '328',
#	 '331',
#	 '335',
#	 '360',
#	 '535',
#	 '64',
#	 '66',
#	 '77',
#	 '78',
#	 '8',
#	 '83',
#	 '87',
#	 '88',
#	 '97',
#	 '99']
    models_to_sample = ['88']
	 
    # two tasks per model with 6-times running this script for that model.
    model_no = models_to_sample[int(args.job_index/2)]
    model_run = args.script_index - math.floor(int(job_array_id)/2)*12
    unique_tag = f"M{model_no}_{model_run}"

    # set general options
    model = "bmm_model_reduced"
    data = "BpA_3MC_ICZ_mixed_noise"
    model_name = f"{model}_{data}"
    yaml_file = PETAB_DIR / "model_selection" / "petab_select_all_ligands_brute_force_mixed_noise" / "full_petab_problem.yaml"
    
    petab_problem = petab.Problem.from_yaml(yaml_file)

    for file in os.listdir(os.path.join(root_path, "input/")):
        if f"{model_no}" == file.split(".")[0]:
            model = petab_select.model.Model.from_yaml(os.path.join(root_path, f"input/{file}"))
    
    pypesto_problem = pypesto.select.model_to_pypesto_problem(model)
    objective_customizer(pypesto_problem.objective)

    # configure saCeSS
    ess_init_args = pypesto.optimize.ess.get_default_ess_options(num_workers=8, dim=pypesto_problem.dim)
    wall_time_for_sacess = 60*60*8
    for x in ess_init_args:
        x['local_optimizer'] = SacessFidesFactory(fides_kwargs={
        'verbose': 0,
        'hessian_update': fides.BFGS()
        })

    optimizer = pypesto.optimize.SacessOptimizer(
        max_walltime_s=wall_time_for_sacess,
        tmpdir=str(SACESS_TMP_DIR / unique_tag),
        sacess_loglevel=logging.ERROR,
        ess_loglevel = logging.ERROR,
        ess_init_args=ess_init_args
    )
    start = time.time()
    result = optimizer.minimize(problem=pypesto_problem,
                        startpoint_method=pypesto.startpoint.UniformStartpoints())
    end = time.time()

    print('Optimization took ', start-end, ' seconds.')

    
    ###hdf5 export
    BEST_MODEL_OPTIM_DIR.mkdir(parents=True, exist_ok=True)
    result_filename = BEST_MODEL_OPTIM_DIR / f"{unique_tag}.hdf5"

    hdf5_writer = save_to_hdf5.OptimizationResultHDF5Writer(str(result_filename))
    hdf5_writer.write(result, overwrite = True)
    hdf5_writer = save_to_hdf5.ProblemHDF5Writer(str(result_filename))
    hdf5_writer.write(pypesto_problem, overwrite = True)

import amici
import pypesto
import pypesto.petab
import pypesto.hierarchical
import pypesto.optimize as optimize
import pypesto.visualize as visualize
import fides
import petab
import os
import numpy as np
import pandas as pd
import libsbml
import time
import datetime
from pypesto.store import save_to_hdf5
from pypesto.store import read_from_hdf5
from pathlib import Path
from scipy.stats import chi2
import dill
import logging

from path_config import PETAB_DIR, RESULTS_DIR



if __name__ == '__main__':

    root_path = Path(__file__).resolve().parent

    # set general options
    engine = pypesto.engine.MultiProcessEngine()


    model = "bmm_model_reduced"
    data = "BpA_3MC_mixed_noise"
    optimizer = "sacess"
    model_name = f"{model}_{data}"
    yaml_file = PETAB_DIR / data / "full_petab_problem.yaml"


    petab_problem = petab.Problem.from_yaml(yaml_file)
    # set new model name if different optimizers are used
    model_name=f'{model_name}_{optimizer}'
   

    pypesto_importer = pypesto.petab.PetabImporter(petab_problem, model_name = model_name)
    model = pypesto_importer.create_model(force_compile=False, verbose=False)
    pypesto_objective = pypesto_importer.create_objective()

    # set the solver
    pypesto_objective.amici_solver.setMaxSteps(100000)
    pypesto_objective.amici_solver.setRelativeTolerance(1e-12)
    pypesto_objective.amici_solver.setAbsoluteTolerance(1e-8)
    pypesto_objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod_adjoint)
    #pypesto_objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod_forward)


    pypesto_problem = pypesto_importer.create_problem(pypesto_objective)


    # configure saCeSS
    ess_init_args = pypesto.optimize.ess.get_default_ess_options(num_workers=8, dim=pypesto_problem.dim)
    wall_time_for_sacess = 60*60*5
    local_optimizer = pypesto.optimize.FidesOptimizer()
    for x in ess_init_args:
        x['local_optimizer'] = local_optimizer


    optimizer = pypesto.optimize.SacessOptimizer(
        max_walltime_s=wall_time_for_sacess,
        sacess_loglevel=logging.ERROR,
        ess_loglevel=logging.ERROR,
        ess_init_args=ess_init_args
    )


    start = time.time()
    result = optimizer.minimize(problem=pypesto_problem,
                        startpoint_method=pypesto.startpoint.UniformStartpoints())

    end = time.time()

    print('Optimization took ', start-end, ' seconds.')


    ###hdf5 export

    result_filename = RESULTS_DIR / "bmm_model" / f"{model_name}.hdf5"
    result_filename.parent.mkdir(parents=True, exist_ok=True)

    hdf5_writer = save_to_hdf5.OptimizationResultHDF5Writer(str(result_filename))
    hdf5_writer.write(result, overwrite = True)
    hdf5_writer = save_to_hdf5.ProblemHDF5Writer(str(result_filename))
    hdf5_writer.write(pypesto_problem, overwrite = True)

    # create ensemble from history

    ensemble = pypesto.ensemble.Ensemble.from_optimization_history(
        result = result,
        rel_cutoff = chi2.ppf(q=95/ 100, df=1)
    )

    # pickle ensemble to file

    ensemble_filename = RESULTS_DIR / "ensembles" / "bmm_model" / f"{model_name}.pkl"
    ensemble_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(ensemble_filename, 'wb') as f:
        dill.dump(ensemble, f) 

    # delete history
    for s in result.optimize_result.list:
        s.history = None

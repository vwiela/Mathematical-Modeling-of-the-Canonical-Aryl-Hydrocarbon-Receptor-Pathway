'''
Assumes that the script evalaute_best_models_ensembles.py has ran 
successfully and that the full-results for each model has been stored.
'''

import warnings
warnings.filterwarnings('ignore')

import copy
import pypesto
import petab
import fides
import pypesto.petab
import pypesto.select
import pypesto.optimize as optimize
from pypesto.ensemble import Ensemble
import pypesto.visualize as visualize
import pypesto.visualize.model_fit
import amici
import numpy as np
import numpy.matlib as nm
import matplotlib.pyplot as plt
import pandas as pd
import libsbml
import sys
from pypesto.store import save_to_hdf5
import dill
import pypesto.hierarchical
from pypesto.store import read_from_hdf5
import pypesto.sample as sample
import os
from pathlib import Path
import csv
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import euclidean_distances

from pypesto.result import OptimizerResult, Result

# import model_selection_functionalities as msfunc
import petab_select
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PETAB_DIR = PROJECT_ROOT / "petab"
MODEL_SELECTION_DIR = PROJECT_ROOT / "results" / "model_selection" / "all_ligands_sacess_brute_force_mixed_noise"
BEST_MODEL_OPTIM_DIR = MODEL_SELECTION_DIR / "best_models_optim"
FIGURES_DIR = MODEL_SELECTION_DIR / "figures"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})

textwidth_pt = 472
inch_per_pt = 1 / 72.27
fig_width = textwidth_pt * inch_per_pt
fig_height = fig_width * 0.65

def own_waterfall_plot(
    values,
    colors=None,
    ref_index=None,
    ref_value=None,
    ax=None,
    line_kwargs=None,
    scatter_kwargs=None,
):
    """
    Waterfall-style plot:
    - values are sorted ascending
    - points are connected by a line
    - optional colors per point
    - one reference point highlighted in red

    Parameters
    ----------
    values : array-like
        1D list/array of values to plot.
    colors : array-like or single color, optional
        Base colors for points. Can be:
        - single matplotlib color (same for all points)
        - array-like of shape (n,) or (n, 4)
    ref_index : int, optional
        Index of the reference point in the ORIGINAL `values` array.
    ref_value : float, optional
        Value of the reference point. The closest value in `values`
        will be taken as the reference.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    line_kwargs : dict, optional
        Extra kwargs for ax.plot().
    scatter_kwargs : dict, optional
        Extra kwargs for ax.scatter().
    """
    values = np.asarray(values)
    n = len(values)

    # order values ascending
    order = np.argsort(values)
    sorted_vals = values[order]
    x = np.arange(n)

    # prepare axis
    if ax is None:
        fig, ax = plt.subplots()

    if line_kwargs is None:
        line_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}

    # draw line
    ax.plot(x, sorted_vals, linestyle="-", marker=None, **line_kwargs)

    # ----- handle base colors (avoid NumPy fixed-width string arrays!) -----
    if colors is None:
        # Use a plain Python list of color strings
        base_colors = ["C0"] * n
    else:
        colors = np.asarray(colors)
        if colors.ndim == 0:
            # single color → broadcast
            base_colors = [colors.item()] * n
        else:
            if len(colors) != n:
                raise ValueError("If `colors` is array-like, it must have same length as `values`.")
            # # reorder with the sorting
            # base_colors = list(colors[order])  # list avoids string truncation issues
            base_colors = list(colors)

    # ----- determine reference point index in SORTED coordinates -----
    ref_sorted_idx = None

    if ref_index is not None:
        # ref_index is in ORIGINAL coordinates
        ref_sorted_idx = int(np.where(order == ref_index)[0][0])

    elif ref_value is not None:
        # closest value to ref_value
        diffs = np.abs(values - ref_value)
        original_idx = int(np.argmin(diffs))
        ref_sorted_idx = int(np.where(order == original_idx)[0][0])

    # color the reference point red
    if ref_sorted_idx is not None:
        base_colors = list(base_colors)  # ensure mutability
        base_colors[ref_sorted_idx] = "red"

    # draw points
    ax.scatter(x, sorted_vals, c=base_colors, zorder=3, **scatter_kwargs)

    ax.set_xlabel("Optimization run (sorted)", fontsize=12)
    ax.set_ylabel("Obj. function value", fontsize=12)
    ax.set_title("")

    return fig, ax

def objective_customizer(objective):
    objective.amici_solver.setRelativeTolerance(1e-12)
    objective.amici_solver.setAbsoluteTolerance(1e-8)
    objective.amici_solver.setMaxSteps(10**6)
    objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod_adjoint)
    pass

def vector_to_result(free_par_vector, pypesto_problem):
    fake_result = Result()
    optimizer_result = OptimizerResult(
    id="fake_result_for_problem_with_no_estimated_parameters",
    x=pypesto_problem.objective.pre_post_processor.preprocess(free_par_vector),
    fval=pypesto_problem.objective(free_par_vector),
    grad=None,
    hess=None,
    res=None,
    sres=None,
    n_fval=1,
    n_grad=0,
    n_hess=0,
    n_res=0,
    n_sres=0,
    x0=[],
    fval0=pypesto_problem.objective(free_par_vector),
    history=None,
    exitflag=0,
    time=10,
    message="Fake result for problem with no estimated parameters.",
)
    fake_result.optimize_result.append(optimizer_result)
    fake_result.problem = pypesto_problem
    return fake_result

def aic_from_nllh(select_model, nllh):
    n_par = Counter(select_model.parameters.values())["estimate"]
    aic = 2*(n_par+nllh)
    return aic


models_to_consider = ['M100',
 'M11',
 'M114',
 'M117',
 'M360',
 'M64',
 'M78',
 'M8',
 'M83',
 'M87',
 'M88',
 'M97',
 'M99']
 

model = "bmm_model_reduced"
data = "BpA_3MC_ICZ_mixed_noise"
model_name = f"{model}_{data}"
yaml_file = PETAB_DIR / data / "full_petab_problem.yaml"

petab_problem = petab.Problem.from_yaml(yaml_file)

output_path = BEST_MODEL_OPTIM_DIR
root_path = SCRIPT_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

models_dict = {}
full_results_dict = {}
best_nllhs_dict = {}
best_aics_dict = {}
for model_id in models_to_consider:

    # set the solver
    best_fvals = []
    for model_run in range(12):
        unique_tag = f"{model_id}_{model_run}"
        # get the result
        result = pypesto.store.read_result(os.path.join(output_path, f"{unique_tag}.hdf5"), problem=False, sample=False)
        best_fvals.append(result.optimize_result.fval[0])

    # get the nllh from model selection and insert
    report = pd.read_csv(MODEL_SELECTION_DIR / "report" / "brute_force_report_short.tsv", sep='\t').set_index("model_id")
    report_id = f"{model_id}-00000000000000000000000000000000000000000000000"
    select_fval = report.loc[report_id, "NLLH"]
    best_fvals.insert(0, select_fval)
    best_fvals = np.array(best_fvals)

    with open(MODEL_SELECTION_DIR / f"{model_id}_waterfall_data.dill", "wb") as f:
        dill.dump(best_fvals, f)

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(best_fvals)))  

    fig, ax = own_waterfall_plot(
        best_fvals,
        colors=colors,
        ref_index=0,
        line_kwargs={
        "color": "black", 
        "linewidth": 1,
        },
        scatter_kwargs={
                "marker": "o",      
                "s": 60,            
                "edgecolor": "black",
                "linewidth": 0.5,
            },
    )

    fig.savefig(FIGURES_DIR / f"{model_id}_starts_waterfall_plot.pdf", dpi=300, bbox_inches='tight')





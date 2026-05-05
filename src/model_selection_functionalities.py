from toposort import toposort

import numpy as np
from functools import partial
import logging
import os
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import fides
import amici
import petab
import petab_select
import pypesto.engine
import pypesto.logging
import pypesto.optimize
import pypesto.select
import pypesto.logging

from petab_select.constants import Method
from pypesto.result import OptimizerResult, Result

# copy method to add hierarchical
from petab_select import Model, parameter_string_to_value
from petab_select.constants import PETAB_PROBLEM
from pypesto.objective import Objective
from pypesto.petab import PetabImporter
from pypesto.problem import Problem
from petab.C import ESTIMATE, NOMINAL_VALUE

from typing import Dict, Iterable

import functionalities as func



# functions to covnert petab_select model to pypesto problem.
def correct_x_guesses(
    x_guesses: Iterable[Dict[str, float]],
    model: Model,
    petab_problem: petab.Problem = None,
    hierarchical: bool = False
):
    """Fix startpoint guesses passed between models of different sizes.

    Any parameter values in `x_guess` for parameters that are not estimated
    should be corrected by replacing them with
    - their corresponding values in `row` if possible, else
    - their corresponding nominal values in the `petab_problem.parameter_df`.
    """
    # TODO reconsider whether correcting is a good idea (`x_guess` is no longer
    # the latest MLE then). Similar todo exists in
    # `ModelSelectorMethod.new_model_problem`.
    # TODO move to PEtab Select?
    # TODO may be issues, e.g. differences in bounds of parameters between
    #      different model PEtab problems is not accounted for, or if there are
    #      different parameters in the old/new PEtab problem.
    if petab_problem is None:
        petab_problem = petab_select.ui.model_to_petab(model=model)[
            PETAB_PROBLEM
        ]
    corrected_x_guesses = None
    if x_guesses is not None:
        corrected_x_guesses = []
        for x_guess in x_guesses:
            corrected_x_guess = []
            for parameter_id in petab_problem.parameter_df.index:
                if hierarchical:
                    if not pd.isna(petab_problem.parameter_df.loc[parameter_id, "parameterType"]):
                        continue
                # Use the `x_guess` value, if the parameter is to be estimated.
                if (
                    petab_problem.parameter_df[ESTIMATE].loc[parameter_id] == 1
                    and parameter_id in x_guess
                ):
                    corrected_value = x_guess[parameter_id]
                # Else use the PEtab Select model parameter value, if defined.
                elif parameter_id in model.parameters:
                    corrected_value = parameter_string_to_value(
                        model.parameters[parameter_id]
                    )
                # Default to the PEtab parameter table nominal value
                else:
                    corrected_value = petab_problem.parameter_df.loc[
                        parameter_id, NOMINAL_VALUE
                    ]
                corrected_x_guess.append(corrected_value)
            corrected_x_guesses.append(corrected_x_guess)
    return corrected_x_guesses

def hier_model_to_pypesto_problem(
    model: Model,
    objective: Objective = None,
    x_guesses: Iterable[Dict[str, float]] = None,
    hierarchical = True,
    petab_problem = None,
    model_name = None,
) -> Problem:
    """Create a pyPESTO problem from a PEtab Select model.

    Parameters
    ----------
    model:
        The model.
    objective:
        The pyPESTO objective.
    x_guesses:
        Startpoints to be used in the multi-start optimization. For example,
        this could be the maximum likelihood estimate from another model.
        Each dictionary has parameter IDs as keys, and parameter values as
        values.
        Values in `x_guess` for parameters that are not estimated will be
        ignored and replaced with their value from the PEtab Select model, if
        defined, else their nominal value in the PEtab parameters table.

    Returns
    -------
    Problem
        The pyPESTO select problem.
    """
    if petab_problem is None:
        petab_problem = petab_select.ui.model_to_petab(model=model)[PETAB_PROBLEM]
    else:
        print("Make sure the specified petab_problem correspond to the one in the model.")
    corrected_x_guesses = None
    if x_guesses is not None:
        corrected_x_guesses = correct_x_guesses(
            x_guesses=x_guesses,
            model=model,
            petab_problem=petab_problem,
            hierarchical = hierarchical
        )
    if model_name is None:
        importer = PetabImporter(petab_problem, hierarchical=hierarchical)
    else:
        importer = PetabImporter(petab_problem, hierarchical=hierarchical, model_name = model_name)
    if objective is None:
        amici_model = importer.create_model(
            non_estimated_parameters_as_constants=False,
        )
        objective = importer.create_objective(
            model=amici_model,
        )
    pypesto_problem = importer.create_problem(
        objective=objective,
        x_guesses=corrected_x_guesses,
    )
    return pypesto_problem

# customize of the objective used in the model selection
def objective_customizer(objective):
    objective.amici_solver.setRelativeTolerance(1e-11)
    objective.amici_solver.setAbsoluteTolerance(1e-16)
    objective.amici_solver.setMaxSteps(10**5)
    objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod_adjoint)
    pass

# function to create result object from parameter vector 
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

# function to evaluate yaml_file
def evaluate_from_yaml_file(yaml_file, petab_select_problem, petab_problem=None):
    
    models = petab_select.model.models_from_yaml_list(yaml_file, petab_problem)
    m= petab_select.ui.best(petab_select_problem, models)
    petab_problem = m.petab_problem
    # get AICC and fval
    AICC = m.get_criterion(petab_select.constants.Criterion.AICC)
    fval = m.get_criterion(petab_select.constants.Criterion.NLLH)
    print("AICC: ", AICC, "\n fval: ", fval)
    pyProb = hier_model_to_pypesto_problem(m, 
                                           petab_problem=petab_problem, 
                                           model_name="current_best_Model")
    

    par_vector = [m.estimated_parameters[x_id]
                  if x_id in m.estimated_parameters.keys()
                  else petab_problem.x_nominal_scaled[np.where(petab_problem.parameter_df.index == x_id)[0][0]]
        for x_id in np.array(pyProb.x_names)[pyProb.x_free_indices]
    ]
    result = vector_to_result(par_vector, pyProb)
    objective_customizer(pyProb.objective)
    axs = func.test_custom_visualize_model_fit(result, 
                                               pyProb, 
                                               petab_problem, 
                                               observable_transform="log")

    return fval, AICC, axs


# function to parse reports for all models within in AICc thresholds

def parse_reports(report_dir,
                  petab_select_problem,
                  best_AICc,
                  AICc_threshold = 10):
    reports = []
    for filename in os.listdir(report_dir): 
        report_path = os.path.join(report_dir, filename)
        try:
            reports.append(pd.read_csv(f'{report_path}', sep='\t').set_index('model_id'))
        except:
            pass

    considered_models = pd.DataFrame(columns=["model_id", "AICc", "AIC", "BIC" "Akaike weight", "n_par estimated", "estimated parameter"])

    for report in reports:
        for model_id in report.index:
            AICc = report.loc[model_id, "AICc"]
            AIC = report.loc[model_id, "AIC"]
            BIC = report.loc[model_id, "BIC"]
            if AICc - best_AICc < AICc_threshold:
                subspace_id = model_id.split('_')[0]
                model_indices = [int(i) for i in model_id.split('_')[1]]
                n_est = sum(np.array(model_indices))
                est_par = [k for k,v in petab_select_problem.model_space.model_subspaces[subspace_id].indices_to_parameters(model_indices).items() if v == 'estimate']
                considered_models = pd.concat([considered_models, pd.DataFrame({"model_id": [model_id], 
                                                            "AICc": [AICc],
                                                            "AIC": [AIC],
                                                            "BIC": [BIC],
                                                            "delta AICc":  [best_AICc - AICc],
                                                            "Akaike weight": [np.exp((best_AICc - AICc)/2)], 
                                                            "n_par estimated": [n_est], 
                                                            "estimated parameter": [est_par] 
                                                                })],
                                                            ignore_index=True)
                considered_models.reset_index()

    # get correct akaike weights
    weight_sum = sum(considered_models["Akaike weight"])
    considered_models["Akaike weight"] = considered_models["Akaike weight"]/weight_sum

    aggregated_df = considered_models.groupby("model_id").agg({"Akaike weight": "mean", 
                                            "AICc": "mean", 
                                            "AIC": "mean",
                                            "BIC": "mean",
                                            "delta AICc": "mean", 
                                            "n_par estimated": "mean",
                                            "estimated parameter": "min"}).sort_values(by="AICc", ascending=True)
    
    # recalculate the Akaike weight
    aggregated_df["Akaike weight"] = np.exp(aggregated_df["delta AICc"]/2)
    weight_sum = sum(aggregated_df["Akaike weight"])
    aggregated_df["Akaike weight"] = aggregated_df["Akaike weight"]/weight_sum
    
    aggregated_df["occurence"] = [considered_models["model_id"].value_counts()[mid] for mid in aggregated_df.index]
    return aggregated_df




from tslearn.metrics import dtw, cdist_dtw
from tslearn.clustering import TimeSeriesKMeans

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from tslearn.metrics import cdist_soft_dtw
from sklearn.metrics import euclidean_distances
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib

def plot_clustered_trajectories(petab_problem, rdatas_dict, model, legend=True, **kwargs):
    conditions = sorted(petab_problem.condition_df.index) # since amici gives rdatas with alphabetically sorted conditions
    observable_ids = petab_problem.get_observable_ids()

    sim_timepoints = rdatas_dict[list(rdatas_dict.keys())[0]][1].t
    ticks_times = []
    for t in np.array(range(0,56,8)):
            if 3600*t <= sim_timepoints[-1]:
                    ticks_times.append(t)
    
    n_clust = model.labels_.max() + 1
          
    colors = cm.tab10.colors
    # cmap = cm.winter
    # colors = cmap(np.linspace(0, 1, n_clust))

    fig, ax = plt.subplots(3,2, figsize=(15,15))

    for i_model, (model_name, rdatas) in enumerate(rdatas_dict.items()):
            cluster_id = model.labels_[i_model]
            color= colors[cluster_id]
            # color = colors(float(cluster_id//n_clust))
            for i_cond, rdata in enumerate(rdatas):
                for obs_index in range(len(observable_ids)):
                    # first the ahrr plot obs_id 0 but rdata 1
                    ax[i_cond, obs_index].plot(rdata.t, rdata.y[:,obs_index], color=color, label=model_name)
                    ax[i_cond, obs_index].set_title(f"{conditions[i_cond]}: {observable_ids[obs_index]}")
                    ax[i_cond, obs_index].set_xlabel("Time [s]")
                    ax[i_cond, obs_index].set_xticks(np.array(ticks_times)*3600)
                    ax[i_cond, obs_index].set_xticklabels([f'{t}' for t in ticks_times])
    if legend:
        leg = fig.legend(labels=rdatas_dict.keys(), loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.add_artist(leg)
    fig.tight_layout(pad=2.0)
    return fig, ax



def plot_dendrogram(model, rdatas_dict, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    n_clust = model.labels_.max() + 1      
    # colors = cm.Dark2.colors
    colors = [matplotlib.colors.to_hex(c) for c in cm.tab10.colors]
    dflt_col = "#808080"   # Unclustered gray
    # Create a color dictionary for the clusters
    color_dict = {model.labels_[i_model]: colors[model.labels_[i_model]] for i_model, (model_name, rdatas) in enumerate(rdatas_dict.items())}

    # Helper to find all leaves under a node -----------------
    def _collect_leaves(node_id: int):
        """
        Return a list of original-sample indices (0..n_samples-1)
        that sit under the given node in the linkage tree.
        """
        if node_id < n_samples:
            return [node_id]
        left, right = linkage_matrix[node_id - n_samples, :2].astype(int)
        return _collect_leaves(left) + _collect_leaves(right)

    # -----------------------------
    # link_color_func – colours every branch
    # -----------------------------
    def link_color_func(node_id: int):
        """
        Colour a branch according to the cluster that *all*
        leaves underneath belong to.  
        If a branch mixes leaves from different clusters,
        paint it grey.
        """
        if node_id < n_samples:
            return color_dict[model.labels_[node_id]]
        leaves = _collect_leaves(node_id)
        leaf_clusters = {model.labels_[leaf] for leaf in leaves}
        if len(leaf_clusters) == 1:           # pure branch
            return color_dict[leaf_clusters.pop()]
        return dflt_col                         # mixed branch

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, color_threshold=None, link_color_func=link_color_func, labels=list(rdatas_dict.keys()), **kwargs)


def plot_model_class_stack(
    model_runs,
    classes,
    title="Class distribution per model (percent)",
    tick_fontsize=8,
    bar_width=0.7,          # used in vertical mode
    label_every=1,          # used in vertical mode
    fig_width_per_bar=0.5,  # used in vertical mode
    min_fig_width=10,
    fig_height=6,
    cluster_order=None,
    annotate=True,
    annot_fontsize=10,
    text_fontsize=12,
    min_pct_label=3.0,
    fmt="{:.0f}%",
    colors=None,
    class_labels=None,
    # --- new ---
    orientation="h",        # "v" (vertical) or "h" (horizontal). Default "h" to help labels.
    bar_height=0.7,         # used in horizontal mode
    fig_height_per_bar=0.4, # used in horizontal mode
    y_label_every=1,        # used in horizontal mode
    model_order=None        # optional list of model names (top-to-bottom for horizontal)
):
    # DataFrame
    df = pd.DataFrame({"model_run": model_runs, "class": classes})
    df["model"] = df["model_run"].astype(str).str.split("_", n=1).str[0]

    # Counts per (model, class)
    counts = df.groupby(["model", "class"]).size().unstack(fill_value=0)

    # Deterministic class order
    class_cols = sorted(counts.columns)
    counts = counts.reindex(class_cols, axis=1)

    # Totals and percentages
    totals = counts.sum(axis=1)
    percentages = counts.div(totals.replace(0, np.nan), axis=0) * 100.0
    percentages = percentages.fillna(0)

    # Dominant class and share to order bars
    dominant_class = percentages.idxmax(axis=1)
    dominant_share = percentages.max(axis=1)

    # Decide cluster group order
    if cluster_order is None:
        uniq = sorted(class_cols)
        if 0 in uniq:
            uniq.remove(0)
            cluster_order = [0] + uniq
        else:
            cluster_order = uniq

    # Build order within each cluster group
    order_idx = []
    for c in cluster_order:
        mask = dominant_class[dominant_class == c].index
        sorted_models = dominant_share.loc[mask].sort_values(ascending=True).index.tolist()
        order_idx.extend(sorted_models)

    percentages = percentages.loc[order_idx]
    totals = totals.loc[order_idx]

    # Optional explicit model order (top-to-bottom for horizontal plots)
    # If provided, ensure we include only models that exist and append any
    # unspecified models afterwards in the previously computed order.
    if model_order is not None:
        # Preserve only models that are present in our data
        provided = [m for m in model_order if m in percentages.index]
        # Append remaining models that weren't specified, preserving their order
        remaining = [m for m in percentages.index if m not in provided]
        new_index = provided + remaining
        # Reindex percentages and totals accordingly
        percentages = percentages.reindex(new_index)
        totals = totals.reindex(new_index)

    # Colors
    color_list = None
    n_classes = len(percentages.columns)
    if colors is None:
        color_list = None
    elif isinstance(colors, (list, tuple)):
        if len(colors) < n_classes:
            raise ValueError(f"Provided {len(colors)} colors but need {n_classes}.")
        color_list = list(colors)[:n_classes]
    elif isinstance(colors, str):
        cmap = matplotlib.cm.get_cmap(colors)
        color_list = [cmap(i/(max(1, n_classes-1))) for i in range(n_classes)]
    else:
        color_list = None

    if orientation.lower().startswith("v"):
        # Vertical mode (existing behavior)
        xticklabels_full = [f"{m}\n(n={int(totals.loc[m])})" for m in percentages.index]
        xticklabels = [lbl if (i % label_every == 0) else "" for i, lbl in enumerate(xticklabels_full)]
        n_bars = len(percentages.index)
        fig_w = max(min_fig_width, fig_width_per_bar * n_bars)

        ax = percentages.plot(kind="bar", stacked=True, width=bar_width, figsize=(fig_w, fig_height),
                              color=color_list)
        ax.set_ylabel(r"Percentage of runs ($\%$)")
        ax.set_xlabel("Model (total runs)")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: rf"{int(y)}$\%$"))
        ax.set_xticklabels(xticklabels, rotation=0, ha="center", fontsize=tick_fontsize)

        # Annotations inside stacks
        if annotate:
            cum = np.zeros(n_bars)
            x_positions = np.arange(n_bars)
            for ci in range(n_classes):
                heights = percentages.iloc[:, ci].to_numpy()
                for i, h in enumerate(heights):
                    if h >= min_pct_label:
                        y = cum[i] + h / 2.0
                        ax.text(x_positions[i], y, fmt.format(h), ha="center", va="center", fontsize=7)
                cum += heights

        xlabel = "Model (total runs)"
        ax.set_xlabel(xlabel)

    else:
        # Horizontal mode (transposed): better for long labels
        yticklabels_full = [f"{m} (n={int(totals.loc[m])})" for m in percentages.index]
        yticklabels = [lbl if (i % y_label_every == 0) else "" for i, lbl in enumerate(yticklabels_full)]
        n_bars = len(percentages.index)
        fig_h = max(fig_height, fig_height_per_bar * n_bars)

        ax = percentages.plot(kind="barh", stacked=True, width=bar_height, figsize=(min_fig_width, fig_h),
                              color=color_list)
        fig  = ax.figure
        ax.set_xlabel(r"Percentage of runs ($\%$)", fontsize=text_fontsize)
        ax.set_ylabel("Model (total runs)", fontsize=text_fontsize)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: rf"{int(y)}"))
        ax.set_yticklabels(yticklabels, rotation=0, va="center", fontsize=tick_fontsize)

        # Annotations inside stacks
        if annotate:
            cum = np.zeros(n_bars)
            y_positions = np.arange(n_bars)
            for ci in range(n_classes):
                widths = percentages.iloc[:, ci].to_numpy()
                for i, w in enumerate(widths):
                    if w >= min_pct_label:
                        x = cum[i] + w / 2.0
                        ax.text(x, y_positions[i], fmt.format(w), ha="center", va="center", fontsize=annot_fontsize)
                cum += widths

        # If a model_order was provided, the user likely expects the first entry
        # to appear at the top. Matplotlib places the first row at the bottom
        # for horizontal bar charts, so invert the y-axis to match "top-to-bottom" order.
        if model_order is not None:
            try:
                ax.invert_yaxis()
            except Exception:
                pass

    # Title and legend
    ax.set_title(title, fontsize=text_fontsize+2)
    if class_labels:
        legend_labels = [class_labels.get(c, str(c)) for c in percentages.columns]
    else:
        legend_labels = [str(c) for c in percentages.columns]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, legend_labels, title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize=text_fontsize)

    # plt.margins(x=0.02, y=0.02)
    # plt.tight_layout()
    plt.show()
    return fig, ax

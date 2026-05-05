import amici
import pypesto
import pypesto.petab
import pypesto.optimize as optimize
import pypesto.visualize as visualize
import fides
from amici.petab_objective import rdatas_to_simulation_df
import amici.parameter_mapping

import petab
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from typing import Union
import pandas as pd
import libsbml
import time
import datetime
from pypesto.store import save_to_hdf5
from pypesto.store import read_from_hdf5

import copy
from path_config import FIGURES_DIR


# Some functions that are repeatedly called for the model evaluation, so I can store them once.


def compute_converged_points_single(result, threshold=None):

    # threshold at which two points are assumed to
    # be "equal" (converged)
    if threshold is None:
        threshold = 1e-2

    # get the best result
    best_result = result.optimize_result.as_list('x')[0]['x']

    n_starts = len(result.optimize_result.as_list('x'))

    # compare best result with all other results
    # |best_result - other_result| < threshold
    conv_points = 0
    for i in range(0, n_starts):

        other_result = result.optimize_result.as_list('x')[i]['x']

        if other_result is not None:
            converged = (np.abs((other_result - best_result)) < (threshold * len(other_result))).all()

            if converged:
                conv_points += 1

    return conv_points

def model_fit_evaluate(pypesto_result, pypesto_problem, petab_problem):

    fig, axis = plt.subplots(1,2,figsize=(16,8))
    pypesto.visualize.waterfall(pypesto_result, ax=axis[0])
    pypesto.visualize.parameters(pypesto_result,
                                size=(15,10),
                                start_indices=20,
                                balance_alpha=False,
                                ax=axis[1])
    
    fig

    custom_visualize_model_fit(pypesto_result, pypesto_problem, petab_problem, start_index=0)

    print (f'The number of converged points is {compute_converged_points_single(pypesto_result)}\n')
    return 

def compute_converged_points_single(result, threshold=None):

    # threshold at which two points are assumed to
    # be "equal" (converged)
    if threshold is None:
        threshold = 1e-2

    # get the best result
    best_result = result.optimize_result.as_list()[0]['fval']

    n_starts = len(result.optimize_result.as_list('x'))

    # compare best result with all other results
    # |best_result - other_result| < threshold
    conv_points = 0
    for i in range(0, n_starts):

        other_result = result.optimize_result.as_list()[i]['fval']

        if other_result is not None:
            converged = np.abs((other_result - best_result)) < threshold 

            if converged:
                conv_points += 1

    return conv_points

def custom_visualize_model_fit(
        pypesto_result,
        original_pypesto_problem,
        petab_problem,
        start_index: int = 0,
        ncols: int = 2,
        plot_trajectories=False,
        save=False,
        figsize = None,
        figurepath = FIGURES_DIR / "model_fit.png",
        objective_customizer = None,
) -> Union[matplotlib.axes.Axes, None]:
    
    pypesto_problem = original_pypesto_problem
    # pypesto_problem.objective= original_pypesto_problem.objective
    if objective_customizer is None:
        pypesto_problem.objective.amici_solver.setMaxSteps(100000)
        pypesto_problem.objective.amici_solver.setRelativeTolerance(1e-12)
        pypesto_problem.objective.amici_solver.setAbsoluteTolerance(1e-8)
        pypesto_problem.objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod_adjoint)

    else:
        objective_customizer(pypesto_problem.objective)

    parameter_mapping = pypesto_problem.objective.parameter_mapping
    edatas = copy.deepcopy(pypesto_problem.objective.edatas)
    amici_model = pypesto_problem.objective.amici_model
    amici_solver = pypesto_problem.objective.amici_solver
    
    
    sorted_conditions = sorted(petab_problem.condition_df.index) # since amici gives rdatas with alphabetically sorted conditions
    conditions = ["MC3", "ICZ", "BpA"] # ordering of conditions as we want it to be plotted later.
    ncond = len(conditions)

    observable_ids = petab_problem.get_observable_ids()
    meas_df = petab_problem.measurement_df
    meas_timepoints = meas_df['time'].unique()
    ntimepoints = len(meas_timepoints)
    sim_timepoints = np.linspace(0, meas_timepoints[-1], 100)

    if len(pypesto_result.optimize_result.list[start_index]['x']) == len(pypesto_problem.x_free_indices):
        x = pypesto_result.optimize_result.list[start_index]['x']
    else:
        x = pypesto_result.optimize_result.list[start_index]['x'][
            pypesto_problem.x_free_indices
        ]
    objective_result = pypesto_problem.objective(x, return_dict=True)

    x_dct = dict(zip(pypesto_problem.x_names, pypesto_result.optimize_result.list[start_index]["x"]))

    try:
        inner_parameters = objective_result['inner_parameters']
        inner_par_dict = dict(zip(pypesto_problem.inner_x_names, inner_parameters))
        x_dct.update(inner_par_dict)
    except:
        pass

    for j in range(ncond):
        edatas[j].setTimepoints(sim_timepoints)

    amici.parameter_mapping.fill_in_parameters(
        edatas=edatas,
        problem_parameters=x_dct,
        scaled_parameters=True,
        parameter_mapping=parameter_mapping,
        amici_model=amici_model,
    )

    rdatas = amici.runAmiciSimulations(
                amici_model,
                amici_solver,
                edatas,
            )

    # set ticks times
    ticks_times = []
    for t in np.array(range(0,56,8)):
        if 3600*t <= sim_timepoints[-1]:
            ticks_times.append(t)
        
    ticks_times = np.array(ticks_times)
    ylimits = [99,9]
    if figsize == None:
        figsize=(20,ncond*6)
    fig, axs = plt.subplots(ncond,ncols,figsize=figsize, constrained_layout=True)
    if ncond == 1:
        i_cond = 0
        for obs_index in range(len(observable_ids)):
            # first the ahrr plot obs_id 0 but rdata 1
            axs[obs_index].plot(rdatas[0].t, rdatas[0].y[:, obs_index], label='Simulation')
            axs[obs_index].fill_between(rdatas[0].t, rdatas[0].y[:, obs_index]+rdatas[0].sigmay[:,obs_index], rdatas[0].y[:, obs_index]-rdatas[0].sigmay[:,obs_index], alpha=0.1, label="Std. Dev.")
            axs[obs_index].set_title(f"{observable_ids[obs_index]}", fontsize=12)
            cond = sorted_conditions[i_cond]
            meas_means = [np.mean(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)
                            for t in meas_timepoints]
            meas_sem = [np.std(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)/np.sqrt(len(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)-1)
                            for t in meas_timepoints]
            axs[obs_index].scatter(meas_timepoints, 
                                meas_means, 
                                label='Data', color="black")
            axs[obs_index].errorbar(meas_timepoints, 
                                    meas_means, 
                                    yerr=meas_sem, 
                                    fmt='o', color="black")
            axs[obs_index].set_xlabel("Time [h]", fontsize=12)
            axs[obs_index].set_xticks(ticks_times*3600)
            axs[obs_index].set_xticklabels([f'{t}' for t in ticks_times], fontsize=10)
            axs[obs_index].yaxis.set_tick_params(labelsize=10)
            axs[obs_index].legend(fontsize=12)
    else:
        for i_cond, rdata in enumerate(rdatas):
            cond = sorted_conditions[i_cond]
            cond_index = conditions.index(cond)
            for obs_index in range(len(observable_ids)):
                # first the ahrr plot obs_id 0 but rdata 1
                axs[cond_index, obs_index].plot(rdata.t, rdata.y[:,obs_index], label='simulation')
                axs[cond_index, obs_index].fill_between(rdata.t, rdata.y[:,obs_index]+rdata.sigmay[:,obs_index], rdata.y[:,obs_index]-rdata.sigmay[:,obs_index], alpha=0.1, label="std. deviation")
                # axs[cond_index, obs_index].set_title(f"{conditions[cond_index]}: {observable_ids[obs_index]}", fontsize=26)
                axs[cond_index, obs_index].set_title(f"{observable_ids[obs_index]}", fontsize=12)
                meas_means = [np.mean(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)
                                    for t in meas_timepoints]
                meas_sem = [np.std(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)/np.sqrt(len(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)-1)
                                    for t in meas_timepoints]
                axs[cond_index, obs_index].scatter(meas_timepoints, 
                                        meas_means, 
                                        label='data', color="black")
                axs[cond_index, obs_index].errorbar(meas_timepoints, 
                                            meas_means, 
                                            yerr=meas_sem, 
                                            fmt='o', color="black")
                axs[cond_index, obs_index].set_xlabel("Time [h]", fontsize=14)
                axs[cond_index, obs_index].set_xticks(ticks_times*3600)
                axs[cond_index, obs_index].set_xticklabels([f'{t}' for t in ticks_times], fontsize=10)
                axs[cond_index, obs_index].yaxis.set_tick_params(labelsize=10)
                axs[cond_index, obs_index].set_ylim(0, ylimits[obs_index])
                axs[cond_index, obs_index].legend(fontsize=14)

    # fig.tight_layout()
    if save:
        figure_dir = os.path.dirname(os.fspath(figurepath))
        if figure_dir:
            os.makedirs(figure_dir, exist_ok=True)
        plt.savefig(figurepath, dpi=500)      


    print(f'The objective function value is {pypesto_problem.objective(x)}\n')

    if plot_trajectories:
        rdata = rdatas[0]
        for state_name in amici_model.getStateNames():
            state_id = amici_model.getStateNames().index(state_name)
            plt.figure()
            plt.plot(rdata.t,
                    rdata.x[:, state_id],
            )
            plt.title(state_name,)
   

    return fig, axs

def custom_visualize_ensembles(
        par_ensemble,
        original_pypesto_problem,
        petab_problem,
        quantiles: float = [0.025, 0.975],
        plot_measurements = False,
        plot_trajectories = False,
        return_dict = False,
        observable_transform = "lin",
        figsize=None,
        savefig = False,
        figurepath= FIGURES_DIR / "ensemble_fit.png",
        objective_customizer = None,
) -> Union[matplotlib.axes.Axes, None]:
    
    rdatas_list = []

    observable_ids = petab_problem.get_observable_ids()
    meas_df = copy.deepcopy(petab_problem.measurement_df)
    meas_timepoints = meas_df['time'].unique()
    sim_timepoints = np.linspace(0, meas_timepoints[-1], 400)

    conditions = sorted(petab_problem.condition_df.index) # since amici gives rdatas with alphabetically sorted conditions
    ncond = len(conditions)

    rdatas_list = []

    pypesto_problem = original_pypesto_problem

    if objective_customizer is None:
        pypesto_problem.objective.amici_solver.setMaxSteps(10**6)
        pypesto_problem.objective.amici_solver.setRelativeTolerance(1e-12)
        pypesto_problem.objective.amici_solver.setAbsoluteTolerance(1e-8)
        pypesto_problem.objective.amici_solver.setSensitivityMethod(amici.SensitivityMethod_adjoint)
    else:
        objective_customizer(pypesto_problem.objective)

    parameter_mapping = pypesto_problem.objective.parameter_mapping

    for index in range(par_ensemble.n_vectors):

        edatas = copy.deepcopy(pypesto_problem.objective.edatas)
        amici_model = pypesto_problem.objective.amici_model
        amici_solver = pypesto_problem.objective.amici_solver

        x = par_ensemble.x_vectors[:,index]

        objective_result = pypesto_problem.objective(x, return_dict=True)
        
        x_full = pypesto_problem.get_full_vector(par_ensemble.x_vectors[:,index], x_fixed_vals = pypesto_problem.x_fixed_vals)

        x_dct = dict(zip(pypesto_problem.x_names, x_full))
        try:
            inner_parameters = objective_result['inner_parameters']
            inner_par_dict = dict(zip(pypesto_problem.inner_x_names, inner_parameters))
            x_dct.update(inner_par_dict)
        except:
            pass

        for j in range(ncond):
            edatas[j].setTimepoints(sim_timepoints)

        amici.parameter_mapping.fill_in_parameters(
            edatas=edatas,
            problem_parameters=x_dct,
            scaled_parameters=True,
            parameter_mapping=parameter_mapping,
            amici_model=amici_model,
            )

        rdatas = amici.runAmiciSimulations(
            amici_model,
            amici_solver,
            edatas,
        )
        # if rdatas[0].status != 0 or rdatas[1].status != 0:
        #     print(f'Error in simulation {index}')
        # else:
        #     rdatas_list.append(rdatas)
        rdatas_list.append(rdatas)

    obs_means = []
    obs_stds = []
    obs_quantiles = []
    state_means = []
    state_stds = []
    state_quantiles = []

    for i_cond in range(ncond):

        observable_simulations = []
        state_simulations = []
        for index in range(len(rdatas_list)):
            rdata = rdatas_list[index][i_cond]
            observable_simulations.append(rdata.y)

            if plot_trajectories:
                state_simulations.append(rdata.x)

        
        # compute mean and quantiles
        obs_means.append(np.mean(np.array(observable_simulations), axis=0))
        obs_stds.append(np.std(np.array(observable_simulations), axis=0))
        obs_quantiles.append(np.quantile(np.array(observable_simulations), quantiles, axis=0))

            
        if plot_trajectories:
            state_means.append(np.mean(np.array(state_simulations), axis=0))
            state_stds.append(np.std(np.array(state_simulations), axis=0))
            state_quantiles.append(np.quantile(np.array(state_simulations), quantiles, axis=0))
    
    obs_means = np.array(obs_means)
    obs_stds = np.array(obs_stds)
    obs_quantiles = np.array(obs_quantiles)
    state_means = np.array(state_means)
    state_stds = np.array(state_stds)
    state_quantiles = np.array(state_quantiles)

    ## Plotting

    # set ticks times for plotting labels in hours
    ticks_times = []
    for t in np.array(range(0,56,8)):
        if 3600*t <= sim_timepoints[-1]:
            ticks_times.append(t)

    ticks_times = np.array(ticks_times)
    if figsize == None:
        figsize=(20,ncond*6)

    # plot observable mean with confidence region
    if observable_transform == 'log':
        meas_df["measurement"] = np.exp(meas_df["measurement"])
        fig, axs = plt.subplots(ncond,len(observable_ids),figsize=figsize, constrained_layout=True)
        if ncond == 1:
            cond_index = 0
            for obs_index in range(len(observable_ids)):
                axs[obs_index].plot(sim_timepoints, np.exp(obs_means[cond_index, :, obs_index]), label='Ensemble mean')
                axs[obs_index].fill_between(sim_timepoints, np.exp(obs_quantiles[cond_index,0,:,obs_index]), np.exp(obs_quantiles[cond_index,1,:,obs_index]), alpha=0.1, label=r"$95\%$ region")
                axs[obs_index].set_title(f"{observable_ids[obs_index]}", fontsize=12)
                if plot_measurements:
                    meas_means = [np.mean(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)
                                for t in meas_timepoints]
                    meas_sem = [np.std(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)/np.sqrt(len(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)-1)
                                for t in meas_timepoints]
                    axs[obs_index].scatter(meas_timepoints, 
                                    meas_means, 
                                    label='data', color="black")
                    axs[obs_index].errorbar(meas_timepoints, 
                                        meas_means, 
                                        yerr=meas_sem, 
                                        fmt='o', color="black")
                axs[obs_index].set_xlabel("Time [h]", fontsize=12)
                axs[obs_index].set_xticks(ticks_times*3600)
                axs[obs_index].set_xticklabels([f'{t}' for t in ticks_times], fontsize=10)
                axs[obs_index].yaxis.set_tick_params(labelsize=10)
                axs[obs_index].legend(fontsize=12)
        else:
            for cond_index in range(ncond):
                cond = conditions[cond_index]
                for obs_index in range(len(observable_ids)):
                    axs[cond_index, obs_index].plot(sim_timepoints, np.exp(obs_means[cond_index, :, obs_index]), label='Ensemble mean')
                    axs[cond_index, obs_index].fill_between(sim_timepoints, np.exp(obs_quantiles[cond_index,0,:,obs_index]), np.exp(obs_quantiles[cond_index,1,:,obs_index]), alpha=0.1, label=r"$95\%$ region")
                    axs[cond_index, obs_index].set_title(f"{observable_ids[obs_index]}", fontsize=12)
                    if plot_measurements:
                        meas_means = [np.mean(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)
                                    for t in meas_timepoints]
                        meas_sem = [np.std(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)/np.sqrt(len(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)-1)
                                    for t in meas_timepoints]
                        axs[cond_index, obs_index].scatter(meas_timepoints, 
                                        meas_means, 
                                        label='data', color="black")
                        axs[cond_index, obs_index].errorbar(meas_timepoints, 
                                            meas_means, 
                                            yerr=meas_sem, 
                                            fmt='o', color="black")
                    axs[cond_index, obs_index].set_xlabel("Time [h]", fontsize=12)
                    axs[cond_index, obs_index].set_xticks(ticks_times*3600)
                    axs[cond_index, obs_index].set_xticklabels([f'{t}' for t in ticks_times], fontsize=10)
                    axs[i_cond, obs_index].yaxis.set_tick_params(labelsize=10)
                    axs[cond_index, obs_index].legend(fontsize=12)
    else:
        fig, axs = plt.subplots(ncond,len(observable_ids),figsize=figsize, constrained_layout=True)
        if ncond == 1:
            cond_index = 0
            for obs_index in range(len(observable_ids)):
                axs[obs_index].plot(sim_timepoints, obs_means[cond_index, :, obs_index], label='Ensemble mean')
                axs[obs_index].fill_between(sim_timepoints, obs_quantiles[cond_index,0,:,obs_index], obs_quantiles[cond_index,1,:,obs_index], alpha=0.1, label=r"$95\%$ region")
                axs[obs_index].set_title(f"{conditions[i_cond]}: {observable_ids[obs_index]}", fontsize=12)
                if plot_measurements:
                    meas_means = [np.mean(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)
                                for t in meas_timepoints]
                    meas_sem = [np.std(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)/np.sqrt(len(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)-1)
                                for t in meas_timepoints]
                    axs[obs_index].scatter(meas_timepoints, 
                                    meas_means, 
                                    label='data', color="black")
                    axs[obs_index].errorbar(meas_timepoints, 
                                        meas_means, 
                                        yerr=meas_sem, 
                                        fmt='o', color="black")
                axs[obs_index].set_xlabel("Time [h]", fontsize=12)
                axs[obs_index].set_xticks(ticks_times*3600)
                axs[obs_index].set_xticklabels([f'{t}' for t in ticks_times], fontsize=10)
                axs[obs_index].yaxis.set_tick_params(labelsize=10)
                axs[obs_index].legend(fontsize=12)
        else:
            for cond_index in range(ncond):
                cond = conditions[cond_index]
                for obs_index in range(len(observable_ids)):
                    axs[cond_index, obs_index].plot(sim_timepoints, obs_means[cond_index, :, obs_index], label='Ensemble mean')
                    axs[cond_index, obs_index].fill_between(sim_timepoints, obs_quantiles[cond_index,0,:,obs_index], obs_quantiles[cond_index,1,:,obs_index], alpha=0.1, label=r"$95\%$ region")
                    axs[i_cond, obs_index].set_title(f"{conditions[i_cond]}: {observable_ids[obs_index]}", fontsize=12)
                    if plot_measurements:
                        meas_means = [np.mean(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)
                                    for t in meas_timepoints]
                        meas_sem = [np.std(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)/np.sqrt(len(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)-1)
                                    for t in meas_timepoints]
                        axs[cond_index, obs_index].scatter(meas_timepoints, 
                                        meas_means, 
                                        label='data', color="black")
                        axs[cond_index, obs_index].errorbar(meas_timepoints, 
                                            meas_means, 
                                            yerr=meas_sem, 
                                            fmt='o', color="black")
                    axs[cond_index, obs_index].set_xlabel("Time [h]", fontsize=12)
                    axs[cond_index, obs_index].set_xticks(ticks_times*3600)
                    axs[cond_index, obs_index].set_xticklabels([f'{t}' for t in ticks_times], fontsize=10)
                    axs[i_cond, obs_index].yaxis.set_tick_params(labelsize=10)
                    axs[cond_index, obs_index].legend(fontsize=12)

    fig.tight_layout()
    if savefig:
        figure_dir = os.path.dirname(os.fspath(figurepath))
        if figure_dir:
            os.makedirs(figure_dir, exist_ok=True)
        plt.savefig(figurepath)

    if plot_trajectories:
        for state_id in range(len(amici_model.getStateNames())):
            plt.figure()
            plt.plot(rdata.t,
                    state_means[:, state_id],
                    label='simulation'
            )
            plt.fill_between(rdata.t,
                            state_quantiles[0, :, state_id],
                            state_quantiles[1, :, state_id],
                            alpha=0.3,
            )
            plt.title(amici_model.getStateNames()[state_id])
            plt.legend()

    if return_dict:
        info_dict = {
            'Parameter Vectors': par_ensemble.x_vectors, 
            'Simulation Timepoints': rdata.t,
            'Simulation Observables': observable_simulations,
            'Simulation States': state_simulations,
            'Observables Mean': obs_means,
            'Observables Standard Deviation': obs_stds,
            'Observables Quantiles': obs_quantiles,
            }
        return info_dict
    else:
        return fig, axs



def test_custom_visualize_model_fit(
        pypesto_result,
        original_pypesto_problem,
        petab_problem,
        start_index: int = 0,
        plot_trajectories: bool = False,
        observable_transform : str = None,
        combine_conditions: bool = False,
) -> Union[matplotlib.axes.Axes, None]:
    
    pypesto_problem = original_pypesto_problem
    # pypesto_problem.objective.amici_solver.setMaxSteps(10**6)
    parameter_mapping = pypesto_problem.objective.parameter_mapping
    edatas = copy.deepcopy(pypesto_problem.objective.edatas)
    amici_model = pypesto_problem.objective.amici_model
    amici_solver = pypesto_problem.objective.amici_solver
    
    
    conditions = sorted(petab_problem.condition_df.index) # since amici gives rdatas with alphabetically sorted conditions
    ncond = len(conditions)

    observable_ids = petab_problem.get_observable_ids()
    meas_df = copy.deepcopy(petab_problem.measurement_df)
    meas_timepoints = meas_df['time'].unique()
    ntimepoints = len(meas_timepoints)
    sim_timepoints = np.linspace(0, meas_timepoints[-1], 400)

    x = pypesto_result.optimize_result.list[start_index]['x'][
        original_pypesto_problem.x_free_indices
    ]
    objective_result = pypesto_problem.objective(x, return_dict=True)

    x_dct = dict(zip(pypesto_problem.x_names, pypesto_result.optimize_result.list[start_index].x))

    inner_parameters = objective_result['inner_parameters']
    inner_par_dict = dict(zip(pypesto_problem.inner_x_names, inner_parameters))
    x_dct.update(inner_par_dict)

    for j in range(ncond):
        edatas[j].setTimepoints(sim_timepoints)

    amici.parameter_mapping.fill_in_parameters(
        edatas=edatas,
        problem_parameters=x_dct,
        scaled_parameters=True,
        parameter_mapping=parameter_mapping,
        amici_model=amici_model,
    )

    rdatas = amici.runAmiciSimulations(
                amici_model,
                amici_solver,
                edatas,
            )
    
    ahrr_noise = inner_par_dict['sigma_AhRR']
    cyp_noise = inner_par_dict['sigma_CyP']
    obs_noise = [cyp_noise, ahrr_noise]

    # set ticks times
    ticks_times = []
    for t in np.array(range(0,56,8)):
        if 3600*t <= sim_timepoints[-1]:
            ticks_times.append(t)
        
    ticks_times = np.array(ticks_times)
    if observable_transform == 'log':
        meas_df["measurement"] = np.exp(meas_df["measurement"])
        if combine_conditions:
            fig, axs = plt.subplots(1,2,figsize=(16,6), constrained_layout=True)
        else:
            fig, axs = plt.subplots(ncond,2,figsize=(16,ncond*6), constrained_layout=True)
        if ncond == 1:
            i_cond = 0
            for obs_index in range(len(observable_ids)):
                    # first the ahrr plot obs_id 0 but rdata 1
                axs[obs_index].plot(rdatas[0].t, np.exp(rdatas[0].y[:, obs_index]), label='simulation')
                axs[obs_index].fill_between(rdatas[0].t, np.exp(rdatas[0].y[:, obs_index]+obs_noise[obs_index]), np.exp(rdatas[0].y[:, obs_index]-obs_noise[obs_index]), alpha=0.1)
                axs[obs_index].set_title(f"{observable_ids[obs_index]}")
                cond = conditions[i_cond]
                meas_means = [np.mean(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)
                                for t in meas_timepoints]
                meas_sem = [np.std(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)/np.sqrt(len(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)-1)
                                for t in meas_timepoints]
                axs[obs_index].scatter(meas_timepoints, 
                                    meas_means, 
                                    label='data')
                axs[obs_index].errorbar(meas_timepoints, 
                                        meas_means, 
                                        yerr=meas_sem, 
                                        fmt='o')
                axs[obs_index].set_xlabel("Time [h]")
                axs[obs_index].set_xticks(ticks_times*3600)
                axs[obs_index].set_xticklabels([f'{t}' for t in ticks_times])
                axs[obs_index].legend()
        else:
            if combine_conditions:
                for i_cond, rdata in enumerate(rdatas):
                    for obs_index in range(len(observable_ids)):
                        # first the ahrr plot obs_id 0 but rdata 1
                        cond = conditions[i_cond]
                        axs[obs_index].plot(rdata.t, np.exp(rdata.y[:,obs_index]), label=f'{cond} simulation')
                        axs[obs_index].fill_between(rdata.t, np.exp(rdata.y[:,obs_index]+obs_noise[obs_index]), np.exp(rdata.y[:,obs_index]-obs_noise[obs_index]), alpha=0.1)
                        axs[obs_index].set_title(f"{conditions[i_cond]}: {observable_ids[obs_index]}")
                        meas_means = [np.mean(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)
                                            for t in meas_timepoints]
                        meas_sem = [np.std(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)/np.sqrt(len(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)-1)
                                            for t in meas_timepoints]
                        axs[obs_index].scatter(meas_timepoints, 
                                                meas_means, 
                                                label=f'{cond} data')
                        axs[obs_index].errorbar(meas_timepoints, 
                                                    meas_means, 
                                                    yerr=meas_sem, 
                                                    fmt='o')
                        axs[obs_index].set_xlabel("Time [h]")
                        axs[obs_index].set_xticks(ticks_times*3600)
                        axs[obs_index].set_xticklabels([f'{t}' for t in ticks_times])
                        axs[obs_index].legend()
            else:    
                for i_cond, rdata in enumerate(rdatas):
                    for obs_index in range(len(observable_ids)):
                        # first the ahrr plot obs_id 0 but rdata 1
                        cond = conditions[i_cond]
                        axs[i_cond, obs_index].plot(rdata.t, np.exp(rdata.y[:,obs_index]), label=f'{cond} simulation')
                        axs[i_cond, obs_index].fill_between(rdata.t, np.exp(rdata.y[:,obs_index]+obs_noise[obs_index]), np.exp(rdata.y[:,obs_index]-obs_noise[obs_index]), alpha=0.1)
                        axs[i_cond, obs_index].set_title(f"{conditions[i_cond]}: {observable_ids[obs_index]}")
                        meas_means = [np.mean(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)
                                            for t in meas_timepoints]
                        meas_sem = [np.std(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)/np.sqrt(len(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)-1)
                                            for t in meas_timepoints]
                        axs[i_cond, obs_index].scatter(meas_timepoints, 
                                                meas_means, 
                                                label=f'{cond} data', color="black")
                        axs[i_cond, obs_index].errorbar(meas_timepoints, 
                                                    meas_means, 
                                                    yerr=meas_sem, 
                                                    fmt='o', color="black")
                        axs[i_cond, obs_index].set_xlabel("Time [h]")
                        axs[i_cond, obs_index].set_xticks(ticks_times*3600)
                        axs[i_cond, obs_index].set_xticklabels([f'{t}' for t in ticks_times])
                        axs[i_cond, obs_index].legend()

        fig.suptitle(f'objective function value is {pypesto_result.optimize_result.get_for_key("fval")[start_index]}', fontsize=12)
        fig.tight_layout()

        print(f'The objective function value is {pypesto_result.optimize_result.get_for_key("fval")[start_index]}\n')

        if plot_trajectories:
            rdata = rdatas[0]
            for state_name in amici_model.getStateNames():
                state_id = amici_model.getStateNames().index(state_name)
                plt.figure()
                plt.plot(rdata.t,
                        rdata.x[:, state_id],
                )
                plt.title(state_name,)

        return axs
    else:
        if combine_conditions:
            fig, axs = plt.subplots(1,2,figsize=(16,6), constrained_layout=True)
        else:
            fig, axs = plt.subplots(ncond,2,figsize=(16,ncond*6), constrained_layout=True)
        if ncond == 1:
            i_cond = 0
            for obs_index in range(len(observable_ids)):
                    # first the ahrr plot obs_id 0 but rdata 1
                axs[obs_index].plot(rdatas[0].t, rdatas[0].y[:, obs_index], label='simulation')
                axs[obs_index].fill_between(rdatas[0].t, rdatas[0].y[:, obs_index]+obs_noise[obs_index], rdatas[0].y[:, obs_index]-obs_noise[obs_index], alpha=0.1)
                axs[obs_index].set_title(f"{observable_ids[obs_index]}")
                cond = conditions[i_cond]
                meas_means = [np.mean(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)
                                for t in meas_timepoints]
                meas_sem = [np.std(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)/np.sqrt(len(meas_df[(meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)-1)
                                for t in meas_timepoints]
                axs[obs_index].scatter(meas_timepoints, 
                                    meas_means, 
                                    label='data', 
                                    color="black")
                axs[obs_index].errorbar(meas_timepoints, 
                                        meas_means, 
                                        yerr=meas_sem, 
                                        fmt='o', color="black")
                axs[obs_index].set_xlabel("Time [h]")
                axs[obs_index].set_xticks(ticks_times*3600)
                axs[obs_index].set_xticklabels([f'{t}' for t in ticks_times])
                axs[obs_index].legend()
        else:
            if combine_conditions:
                for i_cond, rdata in enumerate(rdatas):
                    for obs_index in range(len(observable_ids)):
                        # first the ahrr plot obs_id 0 but rdata 1
                        cond = conditions[i_cond]
                        axs[obs_index].plot(rdata.t, rdata.y[:,obs_index], label=f'{cond} simulation')
                        axs[obs_index].fill_between(rdata.t, rdata.y[:,obs_index]+obs_noise[obs_index], rdata.y[:,obs_index]-obs_noise[obs_index], alpha=0.1)
                        axs[obs_index].set_title(f"{conditions[i_cond]}: {observable_ids[obs_index]}")
                        meas_means = [np.mean(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)
                                            for t in meas_timepoints]
                        meas_sem = [np.std(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)/np.sqrt(len(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)-1)
                                            for t in meas_timepoints]
                        axs[obs_index].scatter(meas_timepoints, 
                                                meas_means, 
                                                label=f'{cond} data', 
                                                color="black")
                        axs[obs_index].errorbar(meas_timepoints, 
                                                    meas_means, 
                                                    yerr=meas_sem, 
                                                    fmt='o',
                                                    color="black")
                        axs[obs_index].set_xlabel("Time [h]")
                        axs[obs_index].set_xticks(ticks_times*3600)
                        axs[obs_index].set_xticklabels([f'{t}' for t in ticks_times])
                        axs[obs_index].legend()
            else:    
                    for i_cond, rdata in enumerate(rdatas):
                        for obs_index in range(len(observable_ids)):
                            # first the ahrr plot obs_id 0 but rdata 1
                            axs[i_cond, obs_index].plot(rdata.t, rdata.y[:,obs_index], label='simulation')
                            axs[i_cond, obs_index].fill_between(rdata.t, rdata.y[:,obs_index]+obs_noise[obs_index], rdata.y[:,obs_index]-obs_noise[obs_index], alpha=0.1)
                            axs[i_cond, obs_index].set_title(f"{conditions[i_cond]}: {observable_ids[obs_index]}")
                            cond = conditions[i_cond]
                            meas_means = [np.mean(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)
                                                for t in meas_timepoints]
                            meas_sem = [np.std(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)/np.sqrt(len(meas_df[(meas_df['simulationConditionId'] == cond) & (meas_df['time'] == t) & (meas_df['observableId'] == observable_ids[obs_index])]['measurement'].values)-1)
                                                for t in meas_timepoints]
                            axs[i_cond, obs_index].scatter(meas_timepoints, 
                                                    meas_means, 
                                                    label='data')
                            axs[i_cond, obs_index].errorbar(meas_timepoints, 
                                                        meas_means, 
                                                        yerr=meas_sem, 
                                                        fmt='o')
                            axs[i_cond, obs_index].set_xlabel("Time [h]")
                            axs[i_cond, obs_index].set_xticks(ticks_times*3600)
                            axs[i_cond, obs_index].set_xticklabels([f'{t}' for t in ticks_times])
                            axs[i_cond, obs_index].legend()

        fig.tight_layout()

        print(f'The objective function value is {pypesto_result.optimize_result.get_for_key("fval")[start_index]}\n')

        if plot_trajectories:
            rdata = rdatas[0]
            for state_name in amici_model.getStateNames():
                state_id = amici_model.getStateNames().index(state_name)
                plt.figure()
                plt.plot(rdata.t,
                        rdata.x[:, state_id],
                )
                plt.title(state_name,)

        return axs    

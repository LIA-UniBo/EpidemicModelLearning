import json
import numpy as np
import pandas as pd
import covasim as cv
import seaborn as sns
import matplotlib.pyplot as plt

from src.data import get_regional_data
from src.interventions import get_sampling_interventions, get_interventions, get_calibration_interventions

if __name__ == '__main__':
    sequence = np.random.choice(['W', 'Y', 'O', 'R'], size=8)

    with open('res/parameters.json', 'r') as json_file:
        j = json.load(json_file)

    time_interval = 20
    initial_params = j['initial_params']
    initial_params['n_days'] = 251 + len(sequence) * time_interval
    intervention_params = j['intervention_params']
    data = get_regional_data(4.46e6 / (initial_params['pop_size'] * initial_params['pop_scale']))

    # initial_params['end_day'] = data['date'].iloc[-1]
    # interventions = get_calibration_interventions(intervention_params)
    # interventions = get_interventions(periods={0: 'init'}, parameters=intervention_params, daily_tests=0)
    interventions = get_sampling_interventions(sequence, intervention_params, time_interval)

    sim = cv.Sim(pars=initial_params, interventions=interventions, datafile=data)
    msim = cv.MultiSim(sim)
    msim.run(n_runs=3)
    msim.mean()
    msim.plot(
        ['n_severe', 'n_critical', 'cum_diagnoses', 'cum_deaths', 'new_diagnoses', 'new_deaths', 'n_infectious', 'n_susceptible', 'new_tests'],
        fig_args={'figsize': (20, 10)},
        scatter_args={'s': 5},
        plot_args={'lw': 2},
        interval=45,
        n_cols=3
    )

    df = pd.DataFrame()
    df['hosp'] = msim.results['n_severe'].values + msim.results['n_critical'].values
    df['diag'] = msim.results['new_diagnoses'].values
    df['diag'] = df['diag'].rolling(7).mean()
    df['dead'] = msim.results['new_deaths'].values
    df['dead'] = df['dead'].rolling(7).mean()
    df['susc'] = msim.results['n_susceptible'].values
    df = df.iloc[252:].reset_index(drop=True)

    _, ax = plt.subplots(2, 2, sharex='row', figsize=(20, 10), tight_layout=True)
    sns.lineplot(x=df.index, y=df['hosp'], ax=ax[0, 0])
    g = sns.lineplot(x=df.index, y=df['diag'], ax=ax[0, 1])
    g.set_xticks(np.arange(len(df))[::time_interval])
    g.set_xticklabels(sequence)
    sns.lineplot(x=df.index, y=df['susc'], ax=ax[1, 1])
    g = sns.lineplot(x=df.index, y=df['dead'], ax=ax[1, 0])
    g.set_xticks(np.arange(len(df))[::time_interval])
    g.set_xticklabels(data['date'].values[252:252+len(df):time_interval])
    plt.show()

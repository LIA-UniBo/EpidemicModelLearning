import numpy as np
import pandas as pd

from interventions import get_delta

URL = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv'


def get_regional_data(scaling_factor: float = 1, region: str = 'Emilia-Romagna') -> pd.DataFrame:
    df = pd.read_csv(URL)
    # extract data of a single region
    df = df[df['denominazione_regione'] == region]
    # reindex using date
    df.index = pd.Series([pd.to_datetime(d).date() for d in df['data']], name='date')
    # keep just some columns and rename them
    df = df[['ricoverati_con_sintomi', 'terapia_intensiva', 'totale_casi', 'deceduti', 'tamponi']]
    df.columns = ['n_severe', 'n_critical', 'cum_diagnoses', 'cum_deaths', 'new_tests']
    # compute daily tests from cumulative value
    df['new_tests'] = [t2 - t1 for t1, t2 in zip([0] + list(df['new_tests'][:-1]), df['new_tests'])]
    # rescaling according to given factor
    df = (df / scaling_factor).clip(lower=0.0)
    # reset index to get date as a column
    return df.reset_index()


def get_real_samples(region: str, zones: dict, scaling_factor: float = 1, time_interval: int = 21) -> pd.DataFrame:
    zones = [
        (range(get_delta(d) - time_interval - 1, get_delta(d) + time_interval), z)
        for d, z in zones.items()
    ]

    samples = {
        days: (init_zone, actuated_zone)
        for (_, init_zone), (days, actuated_zone) in zip([(range(0), 'W')] + zones[:-2], zones[:-1])
    }

    df = get_regional_data(scaling_factor, region)
    real_data = []
    for period, zones in samples.items():
        temp = df.iloc[period]
        row = np.concatenate((
            temp['n_severe'].values[1:] + temp['n_critical'].values[1:],
            temp['cum_diagnoses'].values[1:] - temp['cum_diagnoses'].values[:-1],
            temp['cum_deaths'].values[1:] - temp['cum_deaths'].values[:-1]
        )).reshape(3, -1).transpose().flatten()
        real_data.append(list(row) + list(zones))
    columns = [f'{c}_{d}' for d in range(0, 2 * time_interval) for c in ['hosp', 'diag', 'dead']]
    columns += ['init_zone', 'actuated_zone']
    return pd.DataFrame(real_data, columns=columns)

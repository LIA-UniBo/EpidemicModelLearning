import pandas as pd

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

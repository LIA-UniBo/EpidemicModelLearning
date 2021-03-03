import pandas as pd

URL = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv'

def get_regional_data(scaling_factor: float = 1, region: str = 'Emilia-Romagna') -> pd.DataFrame:
    df = pd.read_csv(URL)
    # extract data of a single region
    df = df[df['denominazione_regione'] == region]
    # reindex using date
    df.index = [pd.to_datetime(d).date() for d in df['data']]
    # keep hospitalized and (cumulative) tests only
    df = df[['totale_ospedalizzati', 'tamponi']]
    df.columns = ['hospitalized', 'tests']
    # compute daily tests from cumulative value
    df['tests'] = [t2 - t1 for t1, t2 in zip([0] + list(df['tests'][:-1]), df['tests'])]
    # rescaling according to given factor
    return (df / scaling_factor).clip(lower=0)

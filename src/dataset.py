import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

one_hot_zones = {
    'W': [1, 0, 0, 0],
    'Y': [0, 1, 0, 0],
    'O': [0, 0, 1, 0],
    'R': [0, 0, 0, 1],
}


def process_dataset(data, rolling_days=7, val_split=0.2):
    def process_series(series):
        # retrieve initial and actuated zone and map to one hot vector
        init_zone = one_hot_zones[series['init_zone']]
        actuated_zone = one_hot_zones[series['actuated_zone']]
        # get input and output data (first half of the series and second half of the series)
        df = pd.DataFrame(series.values[:-2].reshape(-1, 3), columns=['hosp', 'diag', 'dead'])
        output_df = df.tail(len(df) // 2).copy()
        input_df = df.head(len(df) // 2).copy()
        # perform rolling average on hospitalized cases
        input_df['hosp'] = input_df['hosp'].rolling(rolling_days).mean()
        # return the last two weeks of data with the zones as input features
        input_vector = list(input_df.iloc[rolling_days - 1:].values.transpose().flatten()) + init_zone + actuated_zone
        # return the peak of hospitalized and the number of dead individuals in the second half period
        output_vector = [output_df['hosp'].max(), output_df['dead'].iloc[-1] - input_df['dead'].iloc[-1]]
        return input_vector, output_vector

    x, y = [], []
    for _, s in data.iterrows():
        inputs, outputs = process_series(s)
        x.append(inputs)
        y.append(outputs)

    xs, ys = StandardScaler(), MinMaxScaler()
    if val_split is None:
        x, y = xs.fit_transform(x), ys.fit_transform(y)
        return (x, y), (xs, ys)
    else:
        xt, xv, yt, yv = train_test_split(np.array(x), np.array(y), test_size=val_split, shuffle=True, random_state=0)
        xt, yt = xs.fit_transform(x), ys.fit_transform(y)
        xv, yv = xs.transform(xv), ys.transform(yv)
        return (xt, yt), (xv, yv), (xs, ys)

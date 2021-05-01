from typing import Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

one_hot_zones = {
    'W': [1, 0, 0, 0],
    'Y': [0, 1, 0, 0],
    'O': [0, 0, 1, 0],
    'R': [0, 0, 0, 1],
}


def process_dataset(data: pd.DataFrame, val_split: Optional[float] = 0.2, scale_data: bool = True,
                    rolling_days: int = 7) -> tuple:
    def process_series(series: pd.Series):
        # retrieve initial and actuated zone and map to one hot vector
        init_zone = one_hot_zones[series['init_zone']]
        actuated_zone = one_hot_zones[series['actuated_zone']]
        # get input and output data (first half of the series and second half of the series)
        df = pd.DataFrame(series.values[:-2].reshape(-1, 3).astype('float'), columns=['hosp', 'diag', 'dead'])
        output_df = df.tail(len(df) // 2).copy()
        input_df = df.head(len(df) // 2).copy()
        # perform rolling average on each column then return the last two weeks of data with the zones as features
        input_df = input_df.rolling(rolling_days).mean()
        input_vector = list(input_df.iloc[rolling_days - 1:].values.transpose().flatten()) + init_zone + actuated_zone
        # return the peak of hospitalized and the number of dead and diagnosed individuals in the second half period
        output_vector = [output_df['hosp'].max(), output_df['diag'].sum(), output_df['dead'].sum()]
        return input_vector, output_vector

    def get_scalers(xx: np.array, yy: np.array):
        x_scaler = Scaler(data=xx, methods={idx: 'std' if idx < 45 else None for idx in range(len(xx))})
        y_scaler = Scaler(data=yy, methods='minmax')
        return x_scaler, y_scaler

    x, y = [], []
    for _, s in data.iterrows():
        inputs, outputs = process_series(s)
        x.append(inputs)
        y.append(outputs)
    x, y = np.array(x), np.array(y)

    if val_split is None:
        xt, xv, yt, yv = x, None, y, None
    else:
        xt, xv, yt, yv = train_test_split(x, y, test_size=val_split, shuffle=True, random_state=0)

    xs, ys = None, None
    if scale_data:
        xs, ys = get_scalers(xt, yt)
        xt, yt = xs.transform(xt), ys.transform(yt)
        if val_split is not None:
            xv, yv = xs.transform(xv), ys.transform(yv)

    output = [(xt, yt)]
    if val_split is not None:
        output += [(xv, yv)]
    if scale_data:
        output += [(xs, ys)]
    return tuple(output)


class Scaler:
    def __init__(self, data: Any, methods: Any = 'std'):
        super(Scaler, self).__init__()
        # handle non-pandas data
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(np.array(data))
        # handle all-the-same methods
        if not isinstance(methods, dict):
            methods = {column: methods for column in data.columns}
        # default values (translation = 0, scaling = 1)
        self.translation = np.zeros_like(data.iloc[0])
        self.scaling = np.ones_like(data.iloc[0])
        # compute factors
        for idx, column in enumerate(data.columns):
            method = methods.get(column)
            values = data[column]
            if method in ['std', 'standardize']:
                self.translation[idx] = values.mean()
                self.scaling[idx] = values.std()
            elif method in ['norm', 'normalize', 'minmax']:
                self.translation[idx] = values.min()
                self.scaling[idx] = values.max() - values.min()
            elif method in ['zero', 'max', 'zeromax']:
                self.translation[idx] = 0.0
                self.scaling[idx] = values.max()
            elif isinstance(method, tuple):
                minimum, maximum = method
                self.translation[idx] = minimum
                self.scaling[idx] = maximum - minimum
            elif method is not None:
                raise ValueError(f'Method {method} is not supported')

    def transform(self, data: Any):
        return (data - self.translation) / self.scaling

    def inverse_transform(self, data: Any):
        return (data * self.scaling) + self.translation

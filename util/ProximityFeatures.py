from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class ProximityFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, window=7, future_prox=True) -> None:
        super().__init__()
        self.window = window
        self.future_prox = future_prox

    def fit(self, X:pd.DataFrame,y):
        X = X.copy()
        X['y_c'] = y-1
        broken_when = X.groupby(['when_dt']).y_c.sum()

        self.time_map = broken_when\
            .sort_index()\
            .rolling(window=self.window, center=self.future_prox, min_periods=1)\
            .sum()

        self.pos_map = X\
            .sort_values('when_dt')\
            .groupby(['main.pos', 'when_dt'])\
            .y_c.sum().reset_index()\
            .groupby('main.pos')\
            .apply(lambda x: x.set_index('when_dt').y_c.rolling(window=self.window, center=self.future_prox, min_periods=1).sum())
        
        return self
    
    def transform(self, X:pd.DataFrame,y=None):
        assert all(col in X.columns for col in ['main.pos', 'fun.pos', 'when_dt']) ,'Required columns are missing'
        X['after_big_leap'] = X.when_dt.between(big_leap, big_leap_proximity_end)
        X['time_proximity'] = X.when_dt.map(self.time_map)
        X['pos_proximity'] = X.apply(lambda row: self.pos_map[(row['main.pos'], row['when_dt'])] if (row['main.pos'], row['when_dt']) in self.pos_map.index else pd.NA, axis=1)
        X = X.drop(columns=['main.pos','fun.pos','when_dt'])
        X.fillna(0, inplace=True)
        return X
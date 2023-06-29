import numpy as np
import pandas as pd


def apply_summaries(df: pd.DataFrame, funcs: dict) -> pd.DataFrame:
    regions = df.columns.str.extract(r'(\w\d\w)\d{1,4}')[0].unique()
    colnames = [func+'_'+reg for func in funcs for reg in regions]

    X = df.values.reshape(df.shape[0],24,2048)
    X = np.hstack([np.apply_along_axis(f,2,X) for f in funcs.values()])

    return pd.DataFrame(X, columns=colnames)
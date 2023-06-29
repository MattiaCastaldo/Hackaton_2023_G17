import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_cv_results(cv, param_continous, param_cat):
    res = pd.DataFrame(cv.cv_results_)
    cols = lambda rex: res.columns.to_series().filter(regex=rex).tolist()
    res = res.melt(id_vars=cols(r'param_.*'), value_vars=cols(r'split\d+_.*'))

    sns.set(rc={'figure.figsize':(10,3)})
    sns.scatterplot(res, x=param_continous, y = 'value', hue = param_cat)

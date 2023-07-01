import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_cv_results(cv, param_continous, param_cat):
    res = pd.DataFrame(cv.cv_results_)
    sns.set(rc={'figure.figsize':(10,3)})
    sns.scatterplot(res, x=param_continous, y = 'mean_test_score', hue = param_cat)

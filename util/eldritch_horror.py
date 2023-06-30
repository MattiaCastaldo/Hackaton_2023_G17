import os
import pandas as pd
import numpy as np

# DO NOT LOOK INTO THIS

def get_tres(filename):
    df = pd.read_csv(filename)
    idx = np.where(df.y.values == 3)[0]
    return set(idx.flatten())

not_tres = set()
tres = get_tres('data\\shitty_submissions\\tres_30perc.csv')

for filename in os.listdir('data\\shitty_submissions'):
    if 'tres' in filename:
        continue
    not_tres = not_tres | get_tres('data\\shitty_submissions\\'+filename)


for filename in os.listdir('data\\shitty_submissions'):
    if 'tres' not in filename:
        continue
    tres = tres & get_tres('data\\shitty_submissions\\'+filename)




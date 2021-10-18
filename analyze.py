import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
from sklearn.linear_model import LinearRegression

def getDF(fname = sys.stdin):
    cum = pd.DataFrame()
    for f in fname:
        if len(f) == 1: continue
        f = f.replace('\x13','').replace('\x11','').replace('\n','')
        nums = np.fromstring(f, sep = ',')
        nums = np.reshape(nums, (-1, 5))
        df = pd.DataFrame(nums, columns = ['V','I','R','t','?'])
        cum = cum.append(df)
    return cum

if __name__ == '__main__':
    df = getDF()
    R = LinearRegression().fit(df.I.values.reshape(-1, 1),
        df.V.values.reshape(-1,1)).coef_
    print(R)
    df.plot('I','V')
    plt.show()


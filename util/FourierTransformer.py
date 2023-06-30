from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from numpy.fft import fft, ifft, fftfreq

def topk(a, k, axis=2):
    return np.argpartition(-a, k, axis=axis)[...,:k]
'''
    @param k: how many top features we want to consider for each region, class and (maybe) samples
    @param aggregate: different ways to aggregate the top k results:
        - median: 
        - union:
        - intersection:
        - exclusive:
'''

class FourierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, k=10, aggregate='union'):
        self.k = k
        self.aggregate = aggregate

    def fit(self, X, y=None):
        if self.aggregate == 'low_filter':
            # assert type(self.k) == list, 'When predefined k must be the frequency list!'
            self.freq_idx = [list(range(self.k)) for _ in range(24)]
            return self

        X = X.reshape(X.shape[0], 24, 2048)
        F = FourierTransformer.fourier_magnitude(X)[:,:,:1024]

        if self.aggregate == 'median':
            F_byclass = self.get_perclass_values(F, y)
            F_median = [np.median(F_y, axis=0) for F_y in F_byclass]
            topk_byclass = [topk(F_y, self.k, axis=1) for F_y in F_median]
            self.freq_idx = [np.unique(np.concatenate([topk[i,:] for topk in topk_byclass])) for i in range(24)]
        elif self.aggregate == 'union':
            topk_sample = topk(F, self.k, axis=2)
            self.freq_idx = [np.unique(topk_sample[:,i,:].flatten()) for i in range(24)]
        elif self.aggregate == 'intersection' or self.aggregate == 'exclusive':
            topk_sample = topk(F, self.k, axis=2)
            topk_byclass = self.get_perclass_values(topk_sample, y)
            topk_y = [[set(topk[:,i,:].flatten()) for topk in topk_byclass] for i in range(24)]

            self.freq_idx = [topk_y[i][0].intersection(*topk_y[i][1:]) for i in range(24)]
            if self.aggregate == 'exclusive':
                union = [set(topk_sample[:,i,:].flatten()) for i in range(24)]
                self.freq_idx = [union[i] - self.freq_idx[i] for i in range(24)]
            
        self.freq_idx = [list(idx) for idx in self.freq_idx]
        return self
    
    def get_perclass_values(self, tensor, y):
        assert y is not None, 'y shall not be None'
        self.yvalues = np.unique(y)
        y_sep = [tensor[y==y0,:,:] for y0 in self.yvalues]
        return y_sep
    
    def transform(self, X, y=None):
        assert self.freq_idx is not None, 'The transformer must be fitted first!'

        X = X.reshape(X.shape[0], 24, 2048)
        M = FourierTransformer.fourier_magnitude(X, axis=2)
        return np.hstack([M[:,i,idx] for i,idx in enumerate(self.freq_idx)])
    
    @staticmethod
    def power(F):
        return  2*(np.abs(F/2048))**2 
    
    @staticmethod
    def fourier_magnitude(ts, axis=2):
        return FourierTransformer.power(fft(ts, axis=axis))
    
    @staticmethod
    def fourier_approximate(ts, k, axis=0):            
        F = fft(ts, axis=axis)
        M = FourierTransformer.power(F)
        if type(k) is list:
            topk_freq = k
        else:
            topk_freq = topk(M, k, axis=axis)

        F_zeroed = np.zeros(F.shape, dtype=np.complex_)
        F_zeroed[topk_freq] = F[topk_freq]

        return np.real(ifft(F_zeroed, ts.shape[0])), topk_freq
    



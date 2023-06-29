from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from numpy.fft import fft, ifft, fftfreq




class FourierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, k=10, aggregate='union'):
        self.k = k
        self.aggregate = aggregate

    def fit(self, X, y=None):
        X = X.reshape(X.shape[0], 24, 2048)
        F = fft(X, axis=2)
        if self.aggregate == 'median':
            F_byclass = self.get_perclass_values(F, y)
            topk_byclass = [np.median(F_y, axis=0).argsort(axis=1)[:,-self.k:] for F_y in F_byclass]
            self.freq_idx = [np.unique(np.concatenate([topk[i,:] for topk in topk_byclass])) for i in range(24)]
        elif self.aggregate == 'union':
            topk_sample = F.argsort(axis=2)[:,:,-self.k:]
            self.freq_idx = [np.unique(topk_sample[:,i,:].flatten()) for i in range(24)]
        elif self.aggregate == 'intersection' or self.aggregate == 'exclusive':
            topk_sample = F.argsort(axis=2)[:,:,-self.k:]
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
        X = X.reshape(X.shape[0], 24, 2048)
        F = fft(X, axis=2)

        return np.hstack([FourierTransformer.power(F[:,i,idx]) for i,idx in enumerate(self.freq_idx)])
    
    @staticmethod
    def power(F):
        return  2*(np.abs(F/2048))**2 


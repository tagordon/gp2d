import numpy as np
from scipy.optimize import minimize 
import emcee
from mygp import gp, kernel

class model:
    '''Defines a gaussian process model'''
    
    def __init__(self, k1_params, k2_params, white_noise, mean_params, offsets, mean_func, k1_func, k2_func, n):
        self.mean_func = mean_func
        self.n = n
        self.k1_params = k1_params
        self.k2_params = k2_params
        self.white_noise = white_noise
        self.mean_params = mean_params
        self.offsets = offsets
        self.k1_func = k1_func
        self.k2_func = k2_func
        
        mean_arr = self._set_mean(self.offsets, self.mean_params)
        k1 = self.k1_func(*self.k1_params)
        k2 = self.k2_func(self.k2_params)
        self.gp = gp(mean_arr, k1, kernel2=k2, wn=self.white_noise, dim=(self.n > 1)+1)
            
    def _set_mean(self, offsets, mean_params):
        
        def make_mean(off, params): return (lambda x: self.mean_func(x, off, *params))
        
        if self.n > 1:
            return [make_mean(o, mean_params) for o in offsets]
        else: 
            return [make_mean(mean_params)]
        
    def update_gp(self, k1_params=None, k2_params=None, mean_params=None, white_noise=None, offsets=None):
        
        if mean_params is not None:
            self.mean_params = mean_params
        if k1_params is not None:
            self.k1_params = k1_params
        if k2_params is not None:
            self.k2_params = k2_params
        if white_noise is not None:
            self.white_noise = white_noise
        if offsets is not None:
            self.offsets = offsets
        
        mean_arr = self._set_mean(self.offsets, self.mean_params)
        k1 = self.k1_func(*self.k1_params)
        k2 = self.k2_func(self.k2_params)        
        self.gp = gp(mean_arr, k1, kernel2=k2, wn=self.white_noise, dim=(self.n > 1)+1)
        
    def get_params(self):
        return np.concatenate((self.k1_params, self.k2_params, 
                               self.mean_params, self.white_noise, self.offsets))
    
    def _break_params(self, params): 
        lengths = (len(self.k1_params), len(self.k2_params), 
                          np.prod(np.shape(self.mean_params)), 
                   len(self.white_noise), len(self.offsets))
        breaks = [int(np.sum(lengths[:i])) for i in range(len(lengths)+1)]
        k1_params, k2_params, mean_params, white_noise, offsets = [params[breaks[i]:breaks[i+1]] for i in range(len(breaks)-1)]
        return k1_params, k2_params, mean_params, white_noise, offsets
    
    def set_params(self, params):      
        k1_params, k2_params, mean_params, white_noise, offsets = self._break_params(params)
        self._update_gp(k1_params=k1_params, k2_params=k2_params, 
                        white_noise=white_noise, mean_params=mean_params, offsets=offsets)
    
    def neg_log_like(self, data, t):  
        self.gp.compute(t)
        return -self.gp.log_likelihood(data, x=t)
    
    def sample(self, t):
        self.gp.compute(t)
        return self.gp.sample()
    
    # todo: allow offset on means to float while other parameters are locked
    def fit(self, data, t, method='L-BFGS-B', lock_means=True):  
        p0 = self.get_params()
        
        def func(p):
            k1_params, k2_params, mean_params, white_noise, offsets = self._break_params(p)
            self.update_gp(k1_params=k1_params, 
                           k2_params=k2_params, mean_params=mean_params, 
                           white_noise=white_noise, offsets=offsets)
            return self.neg_log_like(data, t)
        
        min_params = minimize(func, x0=p0, method=method)
        BIC = self.neg_log_like(data, t) + len(min_params.x)*np.log(len(t)*self.n)
        
        return min_params.x, BIC
        
        
        

    
    
    
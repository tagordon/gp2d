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
        
    def set_bounds(self, k1_params_bounds=None, params_bounds=None, mean_params_bounds=None, white_noise_bounds=None, offsets_bounds=None):
        
        if k1_params_bounds is None:
            k1_params_bounds = [(0, np.inf)]*len(self.k1_params)
        if k2_params_bounds is None:
            k2_params_bounds = [(0, np.inf)]*(len(self.k2_params)-1)
        if mean_params_bounds is None:
            mean_params_bounds = [(-np.inf, np.inf)]*len(self.mean_params)
        if white_noise_bounds is None:
            white_noise_bounds = [(0, np.inf)]*len(self.white_noise)
        if offsets_bounds is None:
            offsets_bounds = [(0, np.inf)]*len(self.offsets)
            
        self.param_bounds = np.concatenate((k1_params_bounds, k2_params_bounds, 
                                      mean_params_bounds, white_noise_bounds, 
                                      offsets_bounds))
        
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
        
    def print_params(self):
        print('k1_params: ', self.k1_params)
        print('k2_params: ', self.k2_params)
        print('mean_params: ', self.mean_params)
        print('white_noise: ', self.white_noise)
        print('offsets: ', self.offsets)
        
    def get_params_string(self):
        string = 'k1_params: ' + str(self.k1_params) + '\n'
        string += 'k2_params: ' + str(self.k2_params) + '\n'
        string += 'mean_params: ' + str(self.mean_params) + '\n'
        string += 'white_noise: ' + str(self.white_noise) + '\n'
        string += 'offsets: ' + str(self.offsets) + '\n'
        return string
    
    def neg_log_like(self, data, t):  
        self.gp.compute(t)
        return -self.gp.log_likelihood(data, x=t)
    
    def sample(self, t):
        self.gp.compute(t)
        return self.gp.sample()
    
    # todo: allow offset on means to float while other parameters are locked
    def fit(self, data, t, method='L-BFGS-B', lock_means=True):  
        p0 = np.log(self.get_params())
        
        # delete the first element of the scale parameters 
        p0 = np.delete(p0, len(self.k1_params))
        
        def func(p):
            p = np.insert(p, len(self.k1_params), 0)
            p = np.exp(p)
            k1_params, k2_params, mean_params, white_noise, offsets = self._break_params(p)
            self.update_gp(k1_params=k1_params, 
                           k2_params=k2_params, mean_params=mean_params, 
                           white_noise=white_noise, offsets=offsets)
            return self.neg_log_like(data, t)
        
        min_params = minimize(func, x0=p0, method=method, bounds=[(-25, np.inf)]*len(p0))
        BIC = self.neg_log_like(data, t) + len(min_params.x)*np.log(len(t)*self.n)
        
        return min_params.x, BIC
    
    def BIC(self, data, t):
        
        # the -1 is because the first k1_param is always set to 1. 
        return self.neg_log_like(data, t) + (len(self.get_params())-1)*np.log(len(t)*self.n)
    
    # just fit the mean - hold other parameters constant
    def fit_mean(self, data, t, method='L-BFGS-B', bounds=None):
        p0 = self.mean_params
        
        def func(mp):
            self.update_gp(mean_params=mp)
            return self.neg_log_like(data, t)
        
        min_params = minimize(func, x0=p0, method=method, bounds=bounds)
        return min_params.x, 0
    
    def do_mcmc(self, n_steps, n_walkers, data, t, progress=False):
        
        p0 = np.log(self.get_params())
        p0 = np.delete(p0, len(self.k1_params))
        
        def log_like(p):
            p = np.insert(p, len(self.k1_params), 0)
            p = np.exp(p)
            self.set_params(p)
            return -self.neg_log_like(data, t)
        
        n_dim = len(p0)
        p0 = [p0 + 1e-4*np.random.randn(len(p0)) for i in range(n_walkers)]
        
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_like)
        
        if progress:
            for i, result in enumerate(sampler.sample(p0, iterations=n_steps)):
                if (i+1) % 10 == 0:
                    print("{0:5.1%}".format(i/n_steps))
        else: 
            sampler.run_mcmc(p0, n_steps)
        
        samples = sampler.chain.reshape((-1, n_dim))
        return samples        
        
        

    
    
    
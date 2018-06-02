import numpy as np
import solver
from numpy.linalg import cholesky

class kernel:
    '''Defines the kernel function of a gaussian process
    
    '''
    
    def __init__(self):
        return
     
    # apparently with zero white noise the covariance matrix may not be positive definite...  
    # square exponential kernel with white noise
    def exp_sq_kernel(l, sig, wn=1e-12):
        '''Defines the square exponential (gaussian) kernel'''
        
        def k(r):
            return (sig**2)*np.exp(-(r**2)/(2*(l**2))) + (r == 0)*wn
        return k
    
    def matern32_kernel(l, A, wn=1e-12):
        '''Defines the Matérn 3/2 kernel'''
        
        def k(r):
            p = np.sqrt(3)*r/l
            return A*(1 + p)*np.exp(-p) + (r == 0)*wn
        return k
    
    def matern52_kernel(l, A, wn=1e-12):
        '''Defines the Matérn 5/2 kernel'''
        
        def k(r):
            p = np.sqrt(5)*r/l
            return A*(1 + p + (p**2)/3)*np.exp(-p) + (r == 0)*wn
        return k
    
    def celerite_root2_kernel(l, S0, wn=1e-12):
        '''Defines the Q=1/sqrt(2) version of the celerite kernel'''
        w0 = 1/l
        
        def k(r):
            p = w0*r/np.sqrt(2)
            return S0*w0*np.exp(-p)*np.cos(p - np.pi/4.) + (r == 0)*wn
        return k
    
    def two_celerite_root2_kernel(l1, S01, l2, S02, wn=1e-12):
        '''Defines a sum of celerite kernels'''
        
        w01 = 1/l1
        w02 = 1/l2
        
        def k(r):
            p1 = w01*r/np.sqrt(2)
            p2 = w02*r/np.sqrt(2)
            comp1 = S01*w01*np.exp(-p1)*np.cos(p1 - np.pi/4.)
            comp2 = S02*w02*np.exp(-p2)*np.cos(p2 - np.pi/4.)
            return comp1 + comp2 + (r == 0)*wn
        return k
    
    def two_exp_sq_kernel(l1, sig1, l2, sig2, wn=1e-12):
        '''Defines a sum of square exponential kernels'''
        
        def k(r):
            comp1 = (sig1**2)*np.exp(-(r**2)/(2*(l1**2)))
            comp2 = (sig2**2)*np.exp(-(r**2)/(2*(l2**2)))
            return comp1 + comp2 + (r == 0)*wn
        return k
    
    def two_matern32_kernel(l1, A1, l2, A2, wn=1e-12):
        '''Defines a sum of matérn 3/2 kernels'''
        
        def k(r):
            p1, p2 =  np.sqrt(3)*r/l1, np.sqrt(3)*r/l2
            comp1 = A1*(1 + p1)*np.exp(-p1)
            comp2 = A2*(1 + p2)*np.exp(-p2)
            return comp1 + comp2 + (r == 0)*wn
        return k
    
    def scale_kernel(scale):
        '''Defines the scale kernel, representing scaling by an integer between samples'''
        
        scale = np.concatenate(([1], scale))
        return np.outer(scale, scale)
    
    def white_noise_kernel(self, white_noise):
        '''Defines a white noise kernel, with the variance along the diagonal'''
        
        def k(r):
            if r == 0:
                return white_noise
        return k


class gp:
    
    # mean should be an array of vectorized functions
    def __init__(self, mean, kt, jitter=1e-12):
    
        self.mean = mean
        self.kt = kt
        self.jitter = jitter
        
        self.L = None
    
    def _make_cov_t(self, x):
        
        n = len(x)
        return np.matrix([[self.kt(np.abs(x[i]-x[j])) for i in range(n)] for j in range(n)])
    
    def _make_cov_w(self, c):
        
        return np.matrix(np.outer(c, c))
    
    def _make_cov(self, x, c):
        
        n = len(c)*len(x)
        return solver.kronprod(self._make_cov_w(c), self._make_cov_t(x)).to_matrix() + np.matrix(np.diag([self.jitter]*n))
    
    def _compute(self, x, c):
        '''
        Computes the covariance matrix and cholesky decomposition for sampling the GP.
        
        Args: 
            x: A vector containing the sampling points
        '''    
        
        covariance = self._make_cov(x, c)
        self.L = cholesky(covariance)
        
    def sample(self, x, c, n_samples=1):
                
        if len(c) != len(self.mean):
            raise ValueError('Length of mean array must match number of samples')
        
        self._compute(x, c)
        n = len(c)*len(x)
        mean_arr = np.concatenate([m(x) for m in self.mean])
        
        split_samp = []
        for i in range(n_samples):
            u = np.matrix(np.random.randn(n))
            samp = np.array(mean_arr + (self.L*u.transpose()).transpose())[0]
            split_samp.append([samp[i*len(x):(i+1)*len(x)] for i in range(len(c))])
        return split_samp
    
    def log_likelihood(self, data, x, c):
        
        if len(c) != len(self.mean):
            raise ValueError('Length of mean array must match number of samples')
            
        n = len(c)*len(x)
        mean_arr = np.concatenate([m(x) for m in self.mean])
        
        data = np.concatenate(data)
        r = np.matrix((data - mean_arr)).transpose()
        K_inv, log_detK = solver.invert_kron(self._make_cov_w(c), self._make_cov_t(x), self.jitter)
        return -0.5*((r.transpose())*(K_inv*r))[0, 0] - 0.5*log_detK - 0.5*n*np.log(2*np.pi)
        
    
    
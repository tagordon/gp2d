import numpy as np
from numpy.linalg import cholesky
from numpy.linalg import eigvals
from numpy.linalg import inv
from numpy.linalg import det
from scipy.optimize import minimize

# defines a block matrix - useful for constructing 4-d correlation matrix 
class block_matrix:
    
    # initialize zero matrix
    def __init__(self, n, m, p, q):
        self.matrix = np.zeros((n*p, m*q))
     
    # set element i, j with matrix a
    def set_element(self, i, j, a):
        p, q = np.shape(a)
        for k in range(p):
            for l in range(q):
                self.matrix[p*i+k, q*j+l] = a[k, l]
    
    # print the matrix nicely
    def __str__(self):
        return_string = ''
        format_string = '{:.2e}'
        m, n = np.shape(self.matrix)
        for i in range(m):
            for j in range(n):
                return_string += format_string.format(self.matrix[i, j]) + ' '
            return_string += '\n'
        return return_string

#--------------------------------------------------------------------------------------------------------------------------------
# defines the kernel for the gaussian process
class kernel:
    
    # empty init function
    def __init__(self):
        return
     
    # apparently with zero white noise the covariance matrix may not be positive definite...  
    # square exponential kernel with white noise
    def exp_sq_kernel(l, sig, wn=1e-12): 
        def k(r):
            return (sig**2)*np.exp(-(r**2)/(2*(l**2))) + (r == 0)*wn
        return k
    
    def scale_kernel(scale):
        return np.outer(scale, scale)
    
    # white noise adds sigma to diagonal
    def white_noise_kernel(self, white_noise):
        def k(r):
            if r == 0:
                return white_noise
        return k

#--------------------------------------------------------------------------------------------------------------------------------
# defines a gaussian process
class gp:
    
    # initialize with a kernel and mean value (not a function, yet...)
    def __init__(self, mean, kernel1, kernel2=None, wn=1e-6, dim=2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.mean = mean
        self.wn = wn
        self.dim = dim
        
        self.computed = False
        self.log_detK = None
        self.x = None
        self.covariance = None
        self.L = None
        self.K_inv = None
            
    # generate the covariance matrix in the 2d case. subcov is the covariance matrix for the second dimension. 
    def _make_covariance_matrix(self, x):
        
        # two dimensional covariance matrix
        if self.dim is 2:
            n = len(x)
            p = len(self.kernel2)
            sigma = block_matrix(n, n, p, p)
            cov1d = np.zeros((n, n))
            
            # define the covariance matrix in the first dimension:
            for i in range(n):
                for j in range(i+1):
                    cov1d[i, j] = self.kernel1(x[i]-x[j])
                    cov1d[j, i] = cov1d[i, j]
                
            for i in range(p):
                for j in range(i+1):
                    if i is j:
                        sigma.set_element(i, j, cov1d*self.kernel2[i, j] + np.diag((self.wn[i]**2.)*np.ones(len(cov1d))))
                    else:
                        sigma.set_element(i, j, cov1d*self.kernel2[i, j])
                        sigma.set_element(j, i, cov1d*self.kernel2[i, j])
                    
            self.covariance = sigma.matrix
            return sigma.matrix
        
        # one dimensional covariance matrix
        elif self.dim is 1:
            n = len(x)
            sigma = np.matrix(np.zeros((n, n)))
            for i in range(n):
                for j in range(i+1):
                    sigma[i, j] = self.kernel1(x[i] - x[j]) + (i == j)*(self.wn**2.)
                    sigma[j, i] = sigma[i, j]
            return sigma
        else: 
            print('bad dimension argument.')
     
    # pre-compute the covariance matrix, its inverse, the cholesky decomposition, etc.
    def compute(self, x):
        if self.x is not x:            
            self.covariance = self._make_covariance_matrix(x)
            self.L = cholesky(self.covariance)
            self.K_inv = inv(self.covariance)
            self.x = x
            
            self.detK = np.prod(np.diag(self.L))**2
            self.log_detK = 2*np.sum(np.diag(self.L))
            
            self.computed = True
    
    # sample the gaussian process in 1 or 2 dimensions. 
    def sample(self, x=None):
        
        if not self.computed:
            if x is not None:
                self.compute(x)
            else:
                print('Covariance matrix has not been computed. Pass in an array of coordinates.')
        elif x is not None:
            self.compute(x)
    
        sigma = self.covariance
        
        if self.dim is 2:
            n_samp = len(self.kernel2)
            n = n_samp*len(self.x)
            u = np.matrix(np.random.randn(n))
            samp = np.array(self.mean*np.ones(n) + (self.L*u.transpose()).transpose())[0]
            ret = []
            for i in range(n_samp):
                ret.append(samp[i*len(self.x):(i+1)*len(self.x)])
            return ret
        elif self.dim is 1:
            n = len(self.x)
            u = np.matrix(np.random.randn(n))
            return np.array(self.mean*np.ones(n) + (self.L*u.transpose()).transpose())[0]
    
    # compute the log-likelihood of the GP model given the data
    def log_likelihood(self, data, x=None):
        
        # make a two-dimensional array of datapoints into the 1-d array corresponding to the gp sample
        def flatten(arr):
            if len(np.shape(arr)) is 1:
                return arr
            else:
                ret = []
                for r in arr:
                    ret = np.concatenate((ret, r))
                return ret
        
        if self.computed is False:
            if x is not None:
                self.compute(x)
            else: 
                print('Covariance matrix has not been computed. Pass in an array of coordinates.')
        
        # see above
        data = flatten(data)
        r = np.matrix((data - self.mean)).transpose()
        
        if self.dim is 1:
            N = len(self.x)  
        elif self.dim is 2:
            n_samp = len(self.kernel2)
            N = n_samp*len(self.x)
        
        # log likelihood computation 
        return -0.5*((r.transpose())*(self.K_inv*r))[0, 0] - 0.5*self.log_detK - 0.5*N*np.log(2*np.pi)
           
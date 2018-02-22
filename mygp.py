import numpy as np
from numpy.linalg import cholesky
from numpy.linalg import eigvals
from numpy.linalg import inv
from numpy.linalg import det
from scipy.optimize import minimize

# some useful functions...

def flatten(arr):
    '''Concatenates the rows of a 2d array onto the first row
    
    arr is a 2d array 
    '''
    
    if len(np.shape(arr)) is 1:
        return arr
    else:
        ret = []
        for r in arr:
            ret = np.concatenate((ret, r))
        return ret
    
def unflatten(arr, n, m):
    '''Cuts a 1d array into n subarrays of length m and returns a 2d array
    with each row a subarray.
    
    '''
    ret = []
    for i in range(n):
        ret.append(arr[i*m:(i+1)*m])
    return ret
    

def tile(arr, n):
    '''Produces a 1d array which is n copies of arr 
    
    arr is a 1d array
    n is the number of copies to be made 
    '''
    
    a = []
    for i in range(n):
        a = np.concatenate((a, arr))
    return a

class block_matrix:
    '''Defines a block matrix, useful for constructing (n x n) x (m x m) correlation matrix'''
    
    def __init__(self, n, m, p, q):
        '''initialize as a zero matrix with correct dimensions'''
        self.matrix = np.zeros((n*p, m*q))
     
    # set element i, j with matrix a
    def set_element(self, i, j, a):
        '''Set element i, j to a matrix a
        
        a is an m x m matrix
        i, and j are integers with i, j < n
        '''
        
        p, q = np.shape(a)
        for k in range(p):
            for l in range(q):
                self.matrix[p*i+k, q*j+l] = a[k, l]
    
    def __str__(self):
        '''returns a string representation of the block matrix'''
        
        return_string = ''
        format_string = '{:.2e}'
        m, n = np.shape(self.matrix)
        for i in range(m):
            for j in range(n):
                return_string += format_string.format(self.matrix[i, j]) + ' '
            return_string += '\n'
        return return_string

#----------------------------------------------------------------------------------------------------------
# defines the kernel for the gaussian process
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
    
    def scale_kernel(scale):
        '''Defines the scale kernel, representing scaling by an integer between samples'''
        
        return np.outer(scale, scale)
    
    def white_noise_kernel(self, white_noise):
        '''Defines a white noise kernel, with the variance along the diagonal'''
        
        def k(r):
            if r == 0:
                return white_noise
        return k

#----------------------------------------------------------------------------------------------------------
class gp:
    '''Defines a gaussian process'''
    
    def __init__(self, mean, kernel1, kernel2=None, wn=1e-6, dim=2):
        '''Initialize the gp
        
        mean is the mean of the gp and should be a function of the sample points
        kernel1 defines the covariance of the gp in the first dimension
        kernel2 defines the covariance of the gp in the second dimension
        wn is the white noise. wn can be a float for a 1d gp, and should be an array of floats for a 2d gp
            with length equal to the number of samples in the second dimension. 
        dim should be 1 for a 1d gp or 2 for a 2d gp
        '''
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
             
    def _make_covariance_matrix(self, x):
        '''compute the covariance matrix'''
        
        # 2d covariance matrix
        if self.dim is 2:
            n = len(x)
            p = len(self.kernel2)
            sigma = block_matrix(n, n, p, p)
            cov1d = np.zeros((n, n))
            
            # define the covariance matrix in the first dimension
            for i in range(n):
                for j in range(i+1):
                    cov1d[i, j] = self.kernel1(x[i]-x[j])
                    cov1d[j, i] = cov1d[i, j]
            
            # set each entry to the matrix for the second dimension
            for i in range(p):
                for j in range(i+1):
                    if i is j:
                        sigma.set_element(i, j, cov1d*self.kernel2[i, j] + 
                                          np.diag((self.wn[i]**2.)*np.ones(len(cov1d))))
                    else:
                        sigma.set_element(i, j, cov1d*self.kernel2[i, j])
                        sigma.set_element(j, i, cov1d*self.kernel2[i, j])
                    
            self.covariance = sigma.matrix
            return sigma.matrix
        
        # 1d covariance matrix
        elif self.dim is 1:
            n = len(x)
            sigma = np.matrix(np.zeros((n, n)))
            for i in range(n):
                for j in range(i+1):
                    sigma[i, j] = self.kernel1(x[i] - x[j]) + (i == j)*(self.wn[0]**2.)
                    sigma[j, i] = sigma[i, j]
            return sigma
        else: 
            print('bad dimension argument.')
     
    def compute(self, x):
        '''pre-compute the properties of the gp
        x is the list of sample points in the time dimension
        
        '''
        
        if self.x is not x:
            self.computed = False
        
        # only recompute if not already computed for this x
        if not self.computed:            
            self.covariance = self._make_covariance_matrix(x)
            self.L = cholesky(self.covariance)
            self.K_inv = inv(self.covariance)
            self.x = x
            self.detK = np.prod(np.diag(self.L))**2
            self.log_detK = 2*np.sum(np.log(np.diag(self.L)))
            self.computed = True
    
    def sample(self):
        '''compute a sample from the gaussian process'''
        
        if self.computed is False:
            print('Must call compute() before sampling the GP')
        
        # prepare mean array from the mean function
        mean_array = [self.mean(x) for x in self.x]
        
        if self.dim is 2:
            
            mean_array = tile(mean_array, len(self.kernel2))
            n_samp = len(self.kernel2)
            n = n_samp*len(self.x)
            u = np.matrix(np.random.randn(n))
            samp = np.array(mean_array + (self.L*u.transpose()).transpose())[0]
            return unflatten(samp, n_samp, len(self.x))
        
        elif self.dim is 1:
            n = len(self.x)
            u = np.matrix(np.random.randn(n))
            return np.array(mean_array + (self.L*u.transpose()).transpose())[0]
    
    def log_likelihood(self, data, x=None):
        '''Computes the log likelihood for the GP'''
        
        if self.computed is False:
            print('Must call compute() before computing log likelihood')
        
        # prepare the mean array from the mean function
        mean_array = [self.mean(x) for x in self.x]
        if self.dim > 1:
            mean_array = tile(mean_array, len(self.kernel2))
        
        # fix the data to have correction dimensions
        data = flatten(data)
        
        r = np.matrix((data - mean_array)).transpose()
        
        if self.dim is 1:
            N = len(self.x)  
        elif self.dim is 2:
            n_samp = len(self.kernel2)
            N = n_samp*len(self.x)
        
        # log likelihood computation 
        return -0.5*((r.transpose())*(self.K_inv*r))[0, 0] - 0.5*self.log_detK - 0.5*N*np.log(2*np.pi)
           

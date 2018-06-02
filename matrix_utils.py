import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from numpy.linalg import cholesky
from scipy.linalg.lapack import dtrtri
import celerite

class blockmatrix:
    '''
    Representation of a block matrix as a list of numpy matrices. 
    '''
    
    def __init__(self):
        
        pass
        
    def set_blocks(self, m):
        '''
        Sets the blockmatrix 
        
        Args:
            m: A 2D list of matrices
            
        Returns: 
            None
        '''
        
        self.blocks = m
        
    def set_element(self, m, i, j):
        '''
        Sets an element of the block matrix
        
        Args:
            m: The sub-matrix to set an element to
            i, j: Location of element to set to m
        '''
        
        self.blocks[i][j] = m
        
    def to_matrix(self):
        '''
        Flattens the block matrix into a single matrix object
        '''
        
        return np.matrix(np.vstack([np.hstack([m.tolist() for m in row]) for row in self.blocks]))
    
def zeros(self, dim1, dim2):
    '''
    Instantiates a blockmatrix with dimension dim1 filled with 
    zero matrices of dimension dim2
    
    Args:
        dim1: tuple
        dim2: tuple
    
    Return: 
        A blockmatrix object
    '''
        
    n, m = dim1
    p, q = dim2
    ret = blockmatrix()
    ret.set_blocks([[np.matrix(np.zeros((p, q))) for i in range(m)] for j in range(n)])
    return ret
    
def kronprod(A, B):
    '''
    Computes the Kronecker product of two matrices A and B. 
    
    Args: 
        A: Matrix
        B: Matrix
        
    Return: 
        A 2D list with each element a matrix
    '''
    A = A.tolist()
    ret = blockmatrix()
    ret.set_blocks([[a*B for a in row] for row in A])
    return ret

def inv_lower(L):
    '''
    Inverts a lower triangular matrix.
    
    Args:
        L: A lower triangular matrix
    '''
    return np.matrix(dtrtri(L, lower=1)[0])

def det_lower(L):
    '''
    Computes the determinant of a triangular matrix.
    
    Args:
        L: A triangular matrix
    '''
    return np.prod(np.diag(L))**2

def log_det_lower(L):
    '''
    Computes the log determinant of a triangular matrix. 
    
    Args: 
        L: A triangular matrix
    '''
    return 2*np.sum(np.log(np.diag(L)))

def cofactor(m, i, j):
    '''
    computes the cofactor C_ij of matrix m
    
    Args:
        m: A matrix
        i, j: which cofactor to compute
    '''
    
    ncol, nrow = np.shape(m)
    m = m.tolist()
    submatrix = np.matrix([[m[l][k] for k in range(ncol) if k is not i] for l in range(nrow) if l is not j])
    return det(submatrix)*((-1)**(i+j))

def cofactor_matrix(m):
    '''
    computes the cofactor matrix of m
    
    Args: 
        m: A matrix
    '''
    ncol, nrow = np.shape(m)
    return np.matrix([[cofactor(m, i, j) for j in range(nrow)] for i in range(ncol)])

def outer(u, v):
    
    return np.matrix([[u*v for u in u] for v in v])

def LDL(m):
    '''
    computes the LDL decomposition of a matrix m
    
    Args:
        m: A matrix
    '''
    Lchol = cholesky(m)
    S_inv = np.diag(1./np.diag(Lchol))
    D = S*S
    return Lchol*S_inv

    
    
import numpy as np
from numpy.linalg import det
from numpy.linalg import eig
from itertools import product

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

def apply_inv_kron(Kw, Kt, jitter, r):
    '''
    Computes (A x B)^-1 z 
    
    Args:
        Kw: Positive-definite square matrix (length m)
        Kt: Celerite kernel covariance matrix (length n)
        r: Vector of length (m*n)
        
    Returns:
        y: Vector of length (m*n) solving the equation (Kw x Kt) y = r
    '''
    
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

def eig_kron(A, B):
    '''
    Computes the eigenvalues of the kronecker product of matrices A and B
    
    Args:
        A: A matrix
        B: A matrix
        
    Return: 
        values: An array containing the eigenvalues of A x B
        vectors: A matrix containing the corresponding eigenvectors of A x B
    '''
    
    v1, U1 = eig(A)
    v2, U2 = eig(B)
    values = [a*b for a, b in product(v1, v2)]
    vectors = np.stack([kronprod(a, b).to_matrix() for a, b in product(U1, U2)])
    return values, vectors
    
def invert_kron(A, B, j, eps=1e-15):
    '''
    inverts (A x B) + j where j is a diagonal matrix representing the jitter term. 
    
    Args: 
        A: A square matrix (n x n)
        B: A square matrix (m x m)
        r: A 1D array of length n x m
        eps: Minimum eigenvalue such that any smaller eigenvalues will correspond to an entry of 0 in 1/(1/eigenvalue + 1/j)
    '''
    
    n, m = len(A), len(B)
    lam, Q = eig_kron(A, B)
    log_det = np.sum([np.log(1 + l/j) for l in lam]) + n*m*np.log(j)
    Q_inv = Q.T
    D = np.matrix(np.diag([1/(1/l + 1/j) if l > eps else 0 for l in lam]))
    j_inv = np.matrix(np.diag([1/j]*(n*m)))
    j_inv2 = np.matrix(np.diag([1/(j**2)]*(n*m)))
    return j_inv - j_inv2*Q*D*Q_inv, log_det
    
    
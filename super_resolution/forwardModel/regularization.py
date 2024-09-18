
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

def calculateRegularizationMatrix(resolutionXY, lambdaReg):
    
    nRows       = resolutionXY*resolutionXY
    nCols       = resolutionXY*resolutionXY
    rows        = np.linspace(0,nRows-1, nRows)
    cols        = np.linspace(0,nCols-1, nCols)
    
    matrixVal   = np.ones((nRows,))*lambdaReg
    regMatrix   = csc_matrix((matrixVal, (rows, cols)), shape=(nRows, nCols))
    
    return regMatrix
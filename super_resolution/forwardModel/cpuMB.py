
# cpu version of model based reconstruction

import numpy as np
from scipy.sparse.linalg import lsqr

def sigMatReconMB(sigMat, reconMatrix, iterationNum, resolutionXY):
    
    sigMatVec   = np.expand_dims(np.transpose(sigMat).reshape(-1),axis=1)

    bVec        = np.concatenate((sigMatVec, np.zeros((resolutionXY*resolutionXY, 1)) ))
    
    recon, reasonTerm, iterNum, normR = lsqr(reconMatrix, bVec, iter_lim=iterationNum)[:4]

    imageRecon  = np.reshape(recon, (resolutionXY, resolutionXY))

    return imageRecon
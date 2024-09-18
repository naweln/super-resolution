
# function to crop time points and sigMat

import numpy as np

def cropMatrix(matrixIn, xSensor, ySensor, reconDimsXY, fSampling, speedOfSound, delayInSamples, nSamples):
    
    r1      = ySensor[119]
    r2      = np.sqrt(xSensor[0]**2+ySensor[0]**2)
    
    limits  = [r1-(reconDimsXY)/np.sqrt(2), r2+(reconDimsXY)/np.sqrt(2)]
    
    # extract delay
    conversionFactor    = fSampling/speedOfSound
    limitsInSamples     = [np.floor(limits[0]*conversionFactor - delayInSamples), np.ceil(limits[1]                                 *conversionFactor - delayInSamples)]
    
    limitsInSamples[0]  = max(0, int(limitsInSamples[0])-1)
    limitsInSamples[1]  = min(nSamples, int(limitsInSamples[1]))
    
    matrixOut           = matrixIn[limitsInSamples[0]:limitsInSamples[1]]

    return matrixOut


# function to filter sigMat

import numpy as np
from scipy.signal import butter, lfilter, filtfilt

def sigMatFilter(sigMat, lowCutOff, highCutOff, fSampling, fOrder, conRatio):

    sigMatF         = np.zeros(np.shape(sigMat))
    
    nyquistRatio    = conRatio * fSampling
    lowCutOff       = lowCutOff / nyquistRatio
    highCutOff      = highCutOff / nyquistRatio
    lowF, highF     = butter(fOrder, [lowCutOff, highCutOff], btype='bandpass')
    
    for i in range(np.shape(sigMat)[2]):
        for j in range(np.shape(sigMat)[1]):
            sigMatF[:,j,i] = filtfilt(lowF, highF, sigMat[:,j,i])

    return sigMatF
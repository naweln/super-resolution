
import numpy as np

def sigMatNormalize(sigMatIn):

    sigMatOut = np.zeros(np.shape(sigMatIn))

    for i in range(np.shape(sigMatIn)[2]):
        singleF             = sigMatIn[:,:,i]
        meanF               = np.mean(singleF, axis=0)
        sigMatOut[:,:,i]    = singleF - np.tile(meanF, (np.shape(singleF)[0], 1))

    return sigMatOut
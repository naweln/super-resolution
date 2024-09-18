
import numpy as np

# custom libraries
from projection import calculateProjection

# Function to calculate model matrix
def calculateModelMatrix(speedOfSound, resolutionXY, reconDimsXY, timePoints, rSensor, angleSensor, nAngles):
    
    nCols       = resolutionXY*resolutionXY                 # number of columns of the matrix
    nRows       = len(timePoints)*len(angleSensor)            # number of rows of the matrix
    pixelSize   = reconDimsXY/(resolutionXY-1)              # one pixel size
    dt          = 1e-15                                     # diferential of time to perform derivation
    tPlusdt     = timePoints+dt                             # time instants for t+dt
    tMinusdt    = timePoints-dt                             # time instants for t-dt
    
    # max angle required to cover all grid for each of the transducers
    angleMax = np.arcsin(((reconDimsXY+2*pixelSize)*np.sqrt(2))/(2*np.amin(rSensor)))
    
    minusDistSensor     = speedOfSound*tMinusdt
    plusDistSensor      = speedOfSound*tPlusdt

    angles              = np.transpose(np.expand_dims(np.linspace(-angleMax,angleMax,nAngles),axis=0))*np.ones((1,len(timePoints)))
    
    for i in range(0,len(angleSensor)):
        
        print('Projection Number: {}'.format(i+1))
        
        theta               = angleSensor[i]                        # angle to (0,0) point
        
        rMinus              = np.ones((nAngles,1))*minusDistSensor  # -t distance from sensor to curve
        rPlus               = np.ones((nAngles,1))*plusDistSensor   # +t distance from sensor to curve
        
        xMinust             = rSensor[i]-(rMinus)*np.cos(angles)   # x distance at -t based on (0,0) to transducer coordinate system
        yMinust             = (rMinus)*np.sin(angles)              # y distance at +t based on (0,0) to transducer coordinate system
        
        xPlust              = rSensor[i]-(rPlus)*np.cos(angles)    # x distance at +t based on (0,0) to transducer coordinate system
        yPlust              = (rPlus)*np.sin(angles)               # y distance at +t based on (0,0) to transducer coordinate system

        projectionMinust    = calculateProjection(xMinust, yMinust, rMinus, theta, reconDimsXY, resolutionXY, nRows, i)
        projectionPlust     = calculateProjection(xPlust, yPlust, rPlus, theta, reconDimsXY, resolutionXY, nRows, i)

        if i > 0:
            modelMatrix     = modelMatrix + (1/(2*dt))*(projectionPlust - projectionMinust)
        else:
            modelMatrix     = (1/(2*dt))*(projectionPlust - projectionMinust)
    
    # clear variables
    del xMinust, yMinust, rMinus, xPlust, yPlust, rPlus
    
    return modelMatrix
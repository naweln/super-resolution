
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

def calculateProjection(xPoint,yPoint,rPoint,theta,reconDimsXY,resolutionXY,nRows,i):
    
    nCols               = resolutionXY*resolutionXY
    lt                  = np.shape(xPoint)[1]                 # length of the time vector
    nAngles             = np.shape(xPoint)[0]                 # number of points on the curve
    pixelSize           = reconDimsXY/(resolutionXY-1)        # sampling distance in x and y
    
    ##### map points to original grid #####
    xPointUnrotated     = xPoint*np.cos(theta) - yPoint*np.sin(theta); # horizontal position of the points of the curve in the original grid (not rotated)
    yPointUnrotated     = xPoint*np.sin(theta) + yPoint*np.cos(theta); # vertical position of the points of the curve in the original grid (not rotated)
    
    # pad zeros to x and y positions
    xPaddedRight    = np.concatenate((xPointUnrotated, np.zeros((1,lt))))
    yPaddedRight    = np.concatenate((yPointUnrotated, np.zeros((1,lt))))
    xPaddedLeft     = np.concatenate((np.zeros((1,lt)), xPointUnrotated))
    yPaddedLeft     = np.concatenate((np.zeros((1,lt)), yPointUnrotated))
    
    distToOrigin    = np.sqrt(np.power((xPaddedRight - xPaddedLeft),2) + np.power((yPaddedRight - yPaddedLeft),2))
    distToOrigin    = distToOrigin[1:nAngles,:]
    del xPaddedRight, xPaddedLeft, yPaddedRight, yPaddedLeft
    
    vecIntegral = (1/2)*np.divide((np.concatenate((distToOrigin, np.zeros((1,lt))))+np.concatenate((np.zeros((1,lt)), distToOrigin))),rPoint) # vector for calculating the integral
    del distToOrigin
    
    xNorm = (xPointUnrotated + (reconDimsXY/2))/pixelSize+1 # horizontal position of the points of the curve in normalized coordinates
    del xPointUnrotated
    
    yNorm = (yPointUnrotated + (reconDimsXY/2))/pixelSize+1 # vertical position of the points of the curve in normalized coordinates
    del yPointUnrotated
    
    xBefore = np.floor(xNorm)   # horizontal position of the point of the grid at the left of the point (normalized coordinates)
    xAfter  = np.floor(xNorm+1) # horizontal position of the point of the grid at the right of the point (normalized coordinates)
    xDiff   = xNorm-xBefore
    del xNorm

    yBefore = np.floor(yNorm)      # vertical position of the point of the grid below of the point (normalized coordinates)
    yAfter  = np.floor(yNorm+1)    # vertical position of the point of the grid above of the point (normalized coordinates)
    yDiff   = yNorm - yBefore
    del yNorm
    
    ##### define square #####

    # position of the first point of the square
    xSquare1 = xBefore.astype(int)
    ySquare1 = yBefore.astype(int)
    
    # position of the second point of the square
    xSquare2 = xAfter.astype(int)
    ySquare2 = yBefore.astype(int)
    
    # position of the third point of the square
    xSquare3 = xBefore.astype(int)
    ySquare3 = yAfter.astype(int)
    
    # position of the fourth point of the square
    xSquare4 = xAfter.astype(int)
    ySquare4 = yAfter.astype(int)
    
    ##### decide points are inside or outside of the rectangle #####
    inPoint1 = (xSquare1>0) & (xSquare1<=resolutionXY) & (ySquare1>0) & (ySquare1<=resolutionXY) # boolean to decide 1. point of the square is inside the grid
    inPoint2 = (xSquare2>0) & (xSquare2<=resolutionXY) & (ySquare2>0) & (ySquare2<=resolutionXY) # boolean to decide 2. point of the square is inside the grid
    inPoint3 = (xSquare3>0) & (xSquare3<=resolutionXY) & (ySquare3>0) & (ySquare3<=resolutionXY) # boolean to decide 3. point of the square is inside the grid
    inPoint4 = (xSquare4>0) & (xSquare4<=resolutionXY) & (ySquare4>0) & (ySquare4<=resolutionXY) # boolean to decide 4. point of the square is inside the grid

    inVec1 = np.transpose(inPoint1).reshape(1, -1) # convert to vector
    inVec2 = np.transpose(inPoint2).reshape(1, -1) # convert to vector
    inVec3 = np.transpose(inPoint3).reshape(1, -1) # convert to vector
    inVec4 = np.transpose(inPoint4).reshape(1, -1) # convert to vector
    
    ##### define points on grid #####
    pos1 = resolutionXY*(xSquare1-1)+ySquare1 # one dimensional position of the first points of the squares in the grid
    pos2 = resolutionXY*(xSquare2-1)+ySquare2 # one dimensional position of the first points of the squares in the grid
    pos3 = resolutionXY*(xSquare3-1)+ySquare3 # one dimensional position of the first points of the squares in the grid
    pos4 = resolutionXY*(xSquare4-1)+ySquare4 # one dimensional position of the first points of the squares in the grid
    del xSquare1, xSquare2, xSquare3, xSquare4, ySquare1, ySquare2, ySquare3, ySquare4
    
    ##### convert to vector format #####
    posVec1 = np.transpose(pos1).reshape(1, -1) # Pos_triang_1_t in vector form
    posVec2 = np.transpose(pos2).reshape(1, -1) # Pos_triang_1_t in vector form
    posVec3 = np.transpose(pos3).reshape(1, -1) # Pos_triang_1_t in vector form
    posVec4 = np.transpose(pos4).reshape(1, -1) # Pos_triang_1_t in vector form
    del pos1, pos2, pos3, pos4
    
    weight1 = (1-xDiff)*(1-yDiff)*vecIntegral # weight of the first point of the triangle
    weight2 = (xDiff)*(1-yDiff)*vecIntegral # weight of the second point of the triangle
    weight3 = (1-xDiff)*(yDiff)*vecIntegral # weight of the third point of the triangle
    weight4 = (xDiff)*(yDiff)*vecIntegral # weight of the fourth point of the triangle
    
    weightVec1 = np.transpose(weight1).reshape(1, -1) # weight_sq_1 in vector form
    weightVec2 = np.transpose(weight2).reshape(1, -1) # weight_sq_1 in vector form
    weightVec3 = np.transpose(weight3).reshape(1, -1) # weight_sq_1 in vector form
    weightVec4 = np.transpose(weight4).reshape(1, -1) # weight_sq_1 in vector form
    del weight1, weight2, weight3, weight4
    
    rowMatrix       = np.transpose( np.transpose(np.expand_dims(np.linspace(1,lt,lt, dtype=int),axis=0)) * np.ones((1,nAngles), dtype=int) ) # rows of the sparse matrix
    rowMatrixVec    = np.transpose(np.transpose(rowMatrix).reshape(-1, 1))-1 # rows of the sparse matrix in vector form
    del rowMatrix

    rowMat              = np.concatenate((rowMatrixVec[inVec1]+(i*lt), rowMatrixVec[inVec2]+(i*lt),                                                 rowMatrixVec[inVec3]+(i*lt), rowMatrixVec[inVec4]+(i*lt)))

    posMat              = np.concatenate((posVec1[inVec1]-1, posVec2[inVec2]-1, posVec3[inVec3]-1,                                                  posVec4[inVec4]-1))

    weightMat           = np.concatenate((weightVec1[inVec1], weightVec2[inVec2], weightVec3[inVec3],                                               weightVec4[inVec4]))

    projectionMatrix    = csc_matrix((weightMat, (rowMat, posMat)), shape=(nRows, nCols))
    
    return projectionMatrix
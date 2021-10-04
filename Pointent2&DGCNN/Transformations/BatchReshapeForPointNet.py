
import numpy as np

def reshapeBatchInput(pointsInBatch, pointsPerCloud):
    '''
    This funtion will reshape the points coming from the loader so that they can be processed by this implementation of pointnet

    input:
        pointsInBatch type torch tensor of float (pointsInOneCloud*batchSize x 3) 
        pointsPerCloud type float
    
    output
        reshapedBatch (batches x pointsPerCloud x 3)

    '''
    
    PointCloudsInBatch = pointsInBatch.size(0) // pointsPerCloud

    return np.reshape(pointsInBatch, (PointCloudsInBatch,-1,3))
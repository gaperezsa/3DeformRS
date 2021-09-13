
import os

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

class RandomRotateZ(object):

    def __call__(self, data, theta):
        """Transformation layer to be composed within pytorch geometric import of datasets  

        parameters :    data (tuple)(points coordinates tensor, faces information tensor(if exists), array of correspodning labels) 
                        theta (float)(angle to rotate around Z axis)

        returns : tuple (rotated points coordinates tensor, faces information tensor(if exists), array of correspodning labels)
        
        """
        points, faces, label = data
        assert len(theta) == 1
        rotationRepresentation: Rotation = Rotation.from_euler('z', theta)
        rotationMatrix = rotationRepresentation.as_matrix()
        points = points.dot(rotationMatrix.T)
        if faces is not None:
            faces = faces.dot(rotationMatrix.T)
        return points, faces, label

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RotateSO3(object):

    def __call__(self, data,theta):
        """Transformation layer to be composed within pytorch geometric import of datasets  

        parameters :    data (tuple)(points coordinates tensor, faces information tensor(if exists), array of correspodning labels) 
                        theta (tuple)(angles to rotate around XYZ axis)

        returns : tuple (rotated points coordinates tensor, faces information tensor(if exists), array of correspodning labels)
        
        """
        points, faces, label = data
        assert len(theta) == 3
        rotationRepresentation: Rotation = Rotation.from_euler('xyz', theta)
        rotationMatrix = rotationRepresentation.as_matrix()
        points = points.dot(rotationMatrix.T)
        if faces is not None:
            faces = faces.dot(rotationMatrix.T)
        return points, faces, label

    def __repr__(self):
        return self.__class__.__name__ + '()'



class TestDummy(object):

    def __call__(self, data):
        print(data)
        assert False

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
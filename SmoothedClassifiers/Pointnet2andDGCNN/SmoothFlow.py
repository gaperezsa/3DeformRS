#adapted from https://github.com/MotasemAlfarra/DeformRS/blob/main/deform_smooth.py

import math
import torch
import torch.nn.functional as F
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from scipy.spatial.transform import Rotation
from plyfile import PlyElement, PlyData
import copy


class SmoothFlow(object):
    """A smoothed classifier g specifically for pointnet2 or dgcnn implementation"""

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, certify_method : str, sigma: float, exp_name ='debugging' ,device='cuda'):
        """
        :param base_classifier: maps from Batch(batch=[1024*batch_sz], pos=[1024*batch_sz, 3], ptr=[batch_sz+1], y=[batch_sz]) to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.certify_method = certify_method
        self.sigma = sigma
        self.device = device
        self.exp_name = exp_name
        self.plywritten = False
        self.num_bases = 2

    def _GenCloudGaussianNoise(self, x, N,counter):
        ''' This function returns N gaussian noised versions of the pointcloud x...
            meaning, every point will be randomy translated by some vector, l2 norm of said vector will be the certified radius.
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 

        #Gaussian distribution with std deviation sigma
        gaussianNoise = (torch.randn((N*pointCloudShape,3))*self.sigma).float().to(self.device)

        #keep the original pointcloud as the first example
        if counter==0:
            gaussianNoise[0:pointCloudShape] = torch.tensor([0,0,0]).repeat(pointCloudShape,1).float().to(self.device)
        
        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)

        #apply Gaussian noise and return
        hardcopy.pos = hardcopy.pos.repeat(N,1).float().to(self.device)
        hardcopy.pos = hardcopy.pos + gaussianNoise
        return hardcopy

    def _GenCloudRotationX(self, x, N,counter):
        ''' This function returns N rotated versions of the pointcloud x
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0]
        
        #Uniform between [-sigma, sigma]
        theta = (-2 * np.random.rand(N,3) + 1) *self.sigma 

        #null Y and Z rotations
        theta[:,1] = 0
        theta[:,2] = 0

        #keep the original pointcloud as the first example
        if counter==0:
            theta[0] = [0,0,0]

        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)

        #build the rotation matrixes needed
        builder = [torch.from_numpy(Rotation.from_euler('xyz', angle).as_matrix()).float().to(self.device) for angle in theta]

        #rotate to get flow
        allRotations = torch.cat([x.unsqueeze(0) for x in builder])
        stackedPointcloud = hardcopy.pos.repeat(N,1,1)
        flow = (torch.bmm(allRotations,stackedPointcloud.permute(0,2,1)) - stackedPointcloud.permute(0,2,1)).permute(0,2,1)
        flow = torch.reshape(flow,(-1,3))

        #apply rotation and return
        hardcopy.pos = hardcopy.pos.repeat(N,1).float().to(self.device)
        hardcopy.pos = hardcopy.pos + flow
        return hardcopy

    def _GenCloudRotationY(self, x, N,counter):
        ''' This function returns N rotated versions of the pointcloud x
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0]
        
        #Uniform between [-sigma, sigma]
        theta = (-2 * np.random.rand(N,3) + 1) *self.sigma 

        #null X and Z rotations
        theta[:,0] = 0
        theta[:,2] = 0

        #keep the original pointcloud as the first example
        if counter==0:
            theta[0] = [0,0,0]

        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)

        #build the rotation matrixes needed
        builder = [torch.from_numpy(Rotation.from_euler('xyz', angle).as_matrix()).float().to(self.device) for angle in theta]

        #rotate to get flow
        allRotations = torch.cat([x.unsqueeze(0) for x in builder])
        stackedPointcloud = hardcopy.pos.repeat(N,1,1)
        flow = (torch.bmm(allRotations,stackedPointcloud.permute(0,2,1)) - stackedPointcloud.permute(0,2,1)).permute(0,2,1)
        flow = torch.reshape(flow,(-1,3))

        #apply rotation and return
        hardcopy.pos = hardcopy.pos.repeat(N,1).float().to(self.device)
        hardcopy.pos = hardcopy.pos + flow
        return hardcopy

    def _GenCloudRotationZ(self, x, N,counter):
        ''' This function returns N rotated versions of the pointcloud x
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0]
        
        #Uniform between [-sigma, sigma]
        theta = (-2 * np.random.rand(N,3) + 1) *self.sigma 

        #null X and Y rotations
        theta[:,0:2] = 0

        #keep the original pointcloud as the first example
        if counter==0:
            theta[0] = [0,0,0]

        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)

        #build the rotation matrixes needed
        builder = [torch.from_numpy(Rotation.from_euler('xyz', angle).as_matrix()).float().to(self.device) for angle in theta]

        #rotate to get flow
        allRotations = torch.cat([x.unsqueeze(0) for x in builder])
        stackedPointcloud = hardcopy.pos.repeat(N,1,1)
        flow = (torch.bmm(allRotations,stackedPointcloud.permute(0,2,1)) - stackedPointcloud.permute(0,2,1)).permute(0,2,1)
        flow = torch.reshape(flow,(-1,3))

        #apply rotation and return
        hardcopy.pos = hardcopy.pos.repeat(N,1).float().to(self.device)
        hardcopy.pos = hardcopy.pos + flow
        return hardcopy

    def _GenCloudRotationXZ(self, x, N,counter):
        ''' This function returns N rotated versions of the pointcloud x
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0]
        
        #Uniform between [-sigma, sigma]
        theta = (-2 * np.random.rand(N,3) + 1) *self.sigma 

        #null all Y rotations
        theta[:,1] = 0

        #keep the original pointcloud as the first example
        if counter==0:
            theta[0] = [0,0,0]

        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)

        #build the rotation matrixes needed
        builder = [torch.from_numpy(Rotation.from_euler('xyz', angle).as_matrix()).float().to(self.device) for angle in theta]

        #rotate to get flow
        allRotations = torch.cat([x.unsqueeze(0) for x in builder])
        stackedPointcloud = hardcopy.pos.repeat(N,1,1)
        flow = (torch.bmm(allRotations,stackedPointcloud.permute(0,2,1)) - stackedPointcloud.permute(0,2,1)).permute(0,2,1)
        flow = torch.reshape(flow,(-1,3))

        #apply rotation and return
        hardcopy.pos = hardcopy.pos.repeat(N,1).float().to(self.device)
        hardcopy.pos = hardcopy.pos + flow
        return hardcopy


    def _GenCloudRotationXYZ(self, x, N,counter):
        ''' This function returns N rotated versions of the pointcloud x
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0]

        #Uniform between [-sigma, sigma]
        theta = (-2 * np.random.rand(N,3) + 1) *self.sigma 

        #keep the original pointcloud as the first example
        if counter==0:
            theta[0] = [0,0,0]

        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)

        #build the rotation matrixes needed
        builder = [torch.from_numpy(Rotation.from_euler('xyz', angle).as_matrix()).float().to(self.device) for angle in theta]

        #rotate to get flow
        allRotations = torch.cat([x.unsqueeze(0) for x in builder])
        stackedPointcloud = hardcopy.pos.repeat(N,1,1)
        flow = (torch.bmm(allRotations,stackedPointcloud.permute(0,2,1)) - stackedPointcloud.permute(0,2,1)).permute(0,2,1)
        flow = torch.reshape(flow,(-1,3))

        #apply rotation and return
        hardcopy.pos = hardcopy.pos.repeat(N,1).float().to(self.device)
        hardcopy.pos = hardcopy.pos + flow
        return hardcopy
        
    def _GenCloudTranslation(self, x, N,counter):
        ''' This function returns N translated versions of the pointcloud x
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 
        
        #Gaussian distribution with std deviation sigma
        translations = (torch.randn((N, 3))*self.sigma).float().to(self.device)
        
        #keep the original pointcloud as the first example
        if counter==0:
            translations[0] = torch.tensor([0,0,0]).float().to(self.device)

        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)

        #expand translations accordingly, one translation vector for every pointcloud
        builder = [x.repeat(pointCloudShape,1) for x in translations]
        flow = torch.cat([x for x in builder]).float().to(self.device)
        
        #apply translations and return
        hardcopy.pos = hardcopy.pos.repeat(N,1).float().to(self.device)
        hardcopy.pos = hardcopy.pos + flow
        return hardcopy


    def _GenCloudShearing(self, x, N,counter):
        ''' This function returns N sheared versions of the pointcloud x
            shearing will be apllied on the x and y coordinate keeping z coordinate intact
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 
        
        #Gaussian distribution with std deviation sigma
        shearingCoeff = torch.randn((N,1,3))*self.sigma #although 3 values generated, the third wont be used

        #keep the original pointcloud as the first example
        if counter==0:
            shearingCoeff[0] = torch.tensor([[0,0,0]]).float().to(self.device) #keep the original pointcloud as the first example

        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)

        #shearing is introducing the coefficients in the last column of the identity matrix and not changing the diagonal
        shearingMatrixs = torch.eye(3).unsqueeze(0).repeat(N,1,1)
        shearingMatrixs[:,2,:2] = shearingCoeff[:,0,:2]
        shearingMatrixs = shearingMatrixs.permute(0,2,1).float().to(self.device)

        '''                     [[1         0       CoefA   ],
            shearingMatrix =     [0         1       CoefB   ],
                                 [0         0       1       ]]
        '''
        
        #apply to get flow
        stackedPointcloud = hardcopy.pos.repeat(N,1,1)
        flow = (torch.bmm(shearingMatrixs,stackedPointcloud.permute(0,2,1)) - stackedPointcloud.permute(0,2,1)).permute(0,2,1)
        flow = torch.reshape(flow,(-1,3))
        

        #apply shearing and return
        hardcopy.pos = hardcopy.pos.repeat(N,1).float().to(self.device)
        hardcopy.pos = hardcopy.pos + flow
        return hardcopy

    def _GenCloudTapering(self, x, N,counter):

        ''' This function returns N tapered versions of the pointcloud x
            tapering will be apllied on the x and y coordinate keeping z coordinate intact
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved


            in particular, for tapering, the trick will be having one transformation PER POINT rather than per pointcloud.
            this is because the tapering applied in the 3d Certify paper was a function of each Z coordinate of each point.
            Meaning the transformation matrix is higher order and not static given that it depends on the very point its going to be multiplied with.
            this can be noticed here: https://github.com/eth-sri/3dcertify/blob/master/transformations/tapering.py

            thsi funtion has been vectorized in order to compute the perturbed point faster, hence the use of one extra dimension
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 

        #Gaussian distribution with std deviation sigma
        TaperingCoeff = (torch.randn((N, 2))*self.sigma).to(self.device)

        #keep the original pointcloud as the first example
        if counter==0:
            TaperingCoeff[0] = torch.tensor([0,0]).float().to(self.device) #keep the original pointcloud as the first example

        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)

        #have all points that are going to be altered , needed in order to compute the tapering matrixs
        hardcopy.pos = hardcopy.pos.repeat(N,1)

        #preparing a,b and z to fit with the mask
        #same a,b for every 2*amountOfPointPerCloud because two positions in the diagonal of each matrix are gonna change
        #same z for every two positions
        z = hardcopy.pos[:, 2].repeat(2,1).T.flatten()
        a = TaperingCoeff[:,0].repeat(2*pointCloudShape,1).T.flatten()
        b = TaperingCoeff[:,1].repeat(2*pointCloudShape,1).T.flatten()


        boolMask = torch.tensor([[1,0,0],[0,1,0],[0,0,0]]).bool().repeat(N*pointCloudShape,1,1).to(self.device)
        TransformationMatrixs = torch.eye(3).repeat(N*pointCloudShape,1,1).to(self.device)
        TransformationMatrixs[boolMask] = torch.mul(torch.mul(0.5,torch.square(a)),z)+ torch.mul(b,z) + 1
        
        '''This gives matrixes that look something like this

            [[0.5*a^2*z+b*z+1 ,0                 ,0],
             [0               ,0.5*a^2*z+b*z+1   ,0]
             [0               ,0                 ,1]]
        
        one per point, meaning shape of TransformationMatrixs is N*pointCloudShape X 3 X 3
        '''
        
        #use transformation to get flow
        StackedPointcloud = torch.reshape(hardcopy.pos,(N*pointCloudShape,1,3)).permute(0,2,1)
        taperedPoints = torch.bmm(TransformationMatrixs,StackedPointcloud)
        flow = torch.reshape(taperedPoints,(-1,3))-hardcopy.pos

        #apply shearing and return
        hardcopy.pos = hardcopy.pos + flow
        return hardcopy

    def _GenCloudTwisting(self, x, N,counter):
        ''' This function returns N sheared versions of the pointcloud x
            twisting will be apllied on the x and y coordinate keeping z coordinate intact
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved

            in particular, for twisting, the trick will be having one transformation PER POINT rather than per pointcloud.
            this is because the twisting applied in the 3d Certify paper was a function of each Z coordinate of each point.
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 

        #Gaussian distribution with std deviation sigma
        twistingCoeff = (torch.randn((N,1))*self.sigma).float().to(self.device)

        #keep the original pointcloud as the first example
        if counter==0:
            twistingCoeff[0] = torch.tensor([0]).float().to(self.device) #keep the original pointcloud as the first example

        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)
        
        #have all points that are going to be altered , needed in order to compute the twisting matrixs
        hardcopy.pos = hardcopy.pos.repeat(N,1)

        #preparing alpha and z to fit with the mask
        #same alpha for every 4*amountOfPointPerCloud because 4 positions in the identity matrix are gonna change
        #same z for every 4 positions (4 position in this mask means a single point in the point cloud
        z = hardcopy.pos[:, 2].repeat(4,1).T.flatten()
        alpha = twistingCoeff[:,0].repeat(4*pointCloudShape,1).T.flatten()

        #create transformation matrixes
        boolMask = torch.tensor([[1,1,0],[1,1,0],[0,0,0]]).bool().repeat(N*pointCloudShape,1,1).to(self.device)
        TransformationMatrixs = torch.eye(3).repeat(N*pointCloudShape,1,1).to(self.device)
        angles = torch.mul(alpha,z)
        ''' remember 
            sin(alpha*z) = cos(alpha*z + pi/2)
            -sin(alpha*z) = cos(alpha*z - pi/2)
            
            meaning

            [[cos(alpha*z)    ,sin(alpha*z)      ,0],              [[cos(alpha*z)       ,cos(alpha*z + pi/2), 0]
             [-sin(alpha*z)   ,cos(alpha*z)      ,0]        =       [cos(alpha*z - pi/2),cos(alpha*z)       , 0] 
             [0               ,0                 ,1]]               [0                  ,0                  , 1]]
            
            '''
        transformer = torch.tensor([0,math.pi/2,-math.pi/2,0]).repeat(N*pointCloudShape).float().to(self.device)
        angles += transformer
        TransformationMatrixs[boolMask] = torch.cos(angles)
        
        #use transformation to get flow
        StackedPointcloud = torch.reshape(hardcopy.pos,(N*pointCloudShape,1,3)).permute(0,2,1)
        twistedPoints = torch.bmm(TransformationMatrixs,StackedPointcloud)
        flow = torch.reshape(twistedPoints,(-1,3))-hardcopy.pos

        #apply twisting and return
        hardcopy.pos = hardcopy.pos + flow
        return hardcopy
    
    def _GenCloudSqueezing(self, x, N,counter):
        ''' This function returns N squeezed versions of the pointcloud x
            squeezing will be apllied by compressing the x coordinate and stretching the y and z coordinate accordingly
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved

            x will be stretched by a factor K, so, Y and Z will be compressed by 1/sqrt(K)
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 

        #Uniform between [-sigma, sigma]
        Kbar =(-2 * np.random.rand(N,1) + 1) *self.sigma

        #transforming uniform distributed variable
        #Kbar=0 -> compressingCoeffK = 1     Identity transform
        #Kbar=1 -> compressingCoeffK = 1/2   all x coordinates cut to half
        #Kbar=-1 -> compressingCoeffK = 1/2   all x coordinates cut to half
        #Kbar=2 -> compressingCoeffK = 1/3   all x coordinates cut to a third
        #Kbar=-2 -> compressingCoeffK = 1/3   all x coordinates cut to a third
        #compressing on the x coordinate by a 1/(|Kbar|+1) ratio and stretching y,z accordingly so that barycenter and volume is preserved
        compressingCoeffK = (1/(1+torch.abs(torch.from_numpy(Kbar)))).float().to(self.device)

        #keep the original pointcloud as the first example
        if counter==0:
            compressingCoeffK[0] = torch.tensor([1]).float().to(self.device)

        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)


        #preparing K and z to fit with the mask
        #same K for every three positions meaning one matrix during mask asignation, one matrix per one point cloud
        K = compressingCoeffK[:,0].repeat(3,1).T.flatten()
        divisor = (1 / torch.mul( compressingCoeffK[:,0] , torch.sqrt(compressingCoeffK[:,0]) ) ).repeat(3,1).T
        divisor[:,0] = 1
        divisor = divisor.flatten()

        '''K        = [k1              ,k1              ,k1             ,k2             ,k2             ,k2             ,...]
           divisor  = [1               ,1/k1*sqrt(k1)   ,1/k1*sqrt(k1)  ,1              ,1/k2*sqrt(k2)  ,1/k2*sqrt(k2)  ,...]
        
        '''
        #create transformation matrixes
        boolMask = torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).bool().repeat(N,1,1).to(self.device)
        TransformationMatrixs = torch.eye(3).repeat(N,1,1).to(self.device)
        TransformationMatrixs[boolMask] = torch.mul(K,divisor)
        
        '''This gives matrixes that look something like this

            [[k               ,0                 ,0         ],
             [0               ,1/sqrt(k)         ,0         ],
             [0               ,0                 ,1/sqrt(k)]]
        
        '''
        #apply to get flow
        stackedPointcloud = hardcopy.pos.repeat(N,1,1)
        flow = (torch.bmm(TransformationMatrixs,stackedPointcloud.permute(0,2,1)) - stackedPointcloud.permute(0,2,1)).permute(0,2,1)
        flow = torch.reshape(flow,(-1,3))

        #apply twisting and return
        hardcopy.pos = hardcopy.pos.repeat(N,1).float().to(self.device)
        hardcopy.pos = hardcopy.pos + flow
        return hardcopy

    def _GenCloudStretching(self, x, N,counter):
        ''' This function returns N squeezed versions of the pointcloud x
            stretching will be apllied by stretching the x coordinate and compressing the y and z coordinate accordingly
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved

            x will be stretched by a factor K, so, Y and Z will be compressed by 1/sqrt(K)
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 

        #Uniform between [-sigma, sigma]
        Kbar =(-2 * np.random.rand(N,1) + 1) *self.sigma

        #transforming uniform distributed variable
        #Kbar=0 -> stretchingCoeffK = 1     Identity transform
        #Kbar=1 -> stretchingCoeffK = 2   all x coordinates doubled
        #Kbar=-1 -> stretchingCoeffK = 2   all x coordinates doubled
        #Kbar=2 -> stretchingCoeffK = 3   all x coordinates tripled
        #Kbar=-2 -> stretchingCoeffK = 3   all x coordinates tripled
        #stretching on the x coordinate by a |Kbar|+1 ratio and compressing y,z accordingly so that barycenter and volume is preserved
        stretchingCoeffK = (torch.abs(torch.from_numpy(Kbar))+1).float().to(self.device)

        #keep the original pointcloud as the first example
        if counter==0:
            stretchingCoeffK[0] = torch.tensor([1]).float().to(self.device)

        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)


        #preparing K and z to fit with the mask
        #same K for every three positions meaning one matrix during mask asignation, one matrix per one point cloud
        K = stretchingCoeffK[:,0].repeat(3,1).T.flatten()
        divisor = (1 / torch.mul( stretchingCoeffK[:,0] , torch.sqrt(stretchingCoeffK[:,0]) ) ).repeat(3,1).T
        divisor[:,0] = 1
        divisor = divisor.flatten()

        '''K        = [k1              ,k1              ,k1             ,k2             ,k2             ,k2             ,...]
           divisor  = [1               ,1/k1*sqrt(k1)   ,1/k1*sqrt(k1)  ,1              ,1/k2*sqrt(k2)  ,1/k2*sqrt(k2)  ,...]
        
        '''
        #create transformation matrixes
        boolMask = torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).bool().repeat(N,1,1).to(self.device)
        TransformationMatrixs = torch.eye(3).repeat(N,1,1).to(self.device)
        TransformationMatrixs[boolMask] = torch.mul(K,divisor)
        
        '''This gives matrixes that look something like this

            [[k               ,0                 ,0         ],
             [0               ,1/sqrt(k)         ,0         ],
             [0               ,0                 ,1/sqrt(k)]]
        
        '''
        #apply to get flow
        stackedPointcloud = hardcopy.pos.repeat(N,1,1)
        flow = (torch.bmm(TransformationMatrixs,stackedPointcloud.permute(0,2,1)) - stackedPointcloud.permute(0,2,1)).permute(0,2,1)
        flow = torch.reshape(flow,(-1,3))

        #apply twisting and return
        hardcopy.pos = hardcopy.pos.repeat(N,1).float().to(self.device)
        hardcopy.pos = hardcopy.pos + flow
        return hardcopy

    def _GenCloudAffineNoTranslation(self, x, N,counter):
        ''' This function returns N affine transformed versions of the pointcloud x with no translations
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 

        #Uniform between [-sigma, sigma]
        affineCoeffs = (torch.randn((N,3,3))*self.sigma).float().to(self.device) 

        #keep the original pointcloud as the first example
        if counter==0:
            affineCoeffs[0] = torch.tensor([[0,0,0],[0,0,0],[0,0,0]]).float().to(self.device)

        boolMask = torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).bool().repeat(N,1,1).to(self.device)
        affineCoeffs[boolMask] = 1+affineCoeffs[boolMask]
        '''This gives affineCoeffs that look something like this

            [[1+a               ,b                 ,c           ],
             [d                 ,1+e               ,f           ],
             [g                 ,h                 ,1+i         ]]
        
        '''
        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)

        #apply to get flow
        stackedPointcloud = hardcopy.pos.repeat(N,1,1)
        flow = (torch.bmm(affineCoeffs,stackedPointcloud.permute(0,2,1)) - stackedPointcloud.permute(0,2,1)).permute(0,2,1)
        flow = torch.reshape(flow,(-1,3))

        #apply affine with no translation and return
        hardcopy.pos = hardcopy.pos.repeat(N,1).float().to(self.device)
        hardcopy.pos = hardcopy.pos + flow
        return hardcopy

    def _GenCloudAffine(self, x, N,counter):
        ''' This function returns N affine transformed versions of the pointcloud x
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
            counter: int just to cehck if the original pointcloud should be conserved

        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 

        #Uniform between [-sigma, sigma]
        affineCoeffs = (torch.randn((N,3,4))*self.sigma).float().to(self.device) 

        #keep the original pointcloud as the first example
        if counter==0:
            affineCoeffs[0] = torch.tensor([[0,0,0,0],[0,0,0,0],[0,0,0,0]]).float().to(self.device)

        #copy x by value to not affect it by reference
        hardcopy = copy.deepcopy(x)

        #expand labels,pointer,batch indexes
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)

        boolMask = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).bool().repeat(N,1,1).to(self.device)
        affineCoeffs[boolMask] = 1+affineCoeffs[boolMask]
        '''This gives affineCoeffs that look something like this

            [[1+a               ,b                 ,c           j],
             [d                 ,1+e               ,f           k],
             [g                 ,h                 ,1+i         l]]]
        
        '''

        StackedPointcloud = torch.cat((hardcopy.pos,torch.ones(pointCloudShape).unsqueeze(0).T.float().to(self.device)),1)

        '''This gives StackedPointcloud that look something like this

            [[[x1               ,y1                 ,z1                  ,1       ],
             [x2               ,y2                 ,z2                   ,1       ],
             [x3               ,y3                 ,z3                   ,1       ],
             [x4               ,y4                 ,z4                   ,1       ],
             ... 
             xpointCloudShape  ,ypointCloudShape,  ,zpointCloudShape     ,1       ]]]
        
        '''

        #batch multiply and get flow
        StackedPointcloud = StackedPointcloud.repeat(N,1,1)
        flow = (torch.bmm(affineCoeffs,StackedPointcloud.permute(0,2,1)) - hardcopy.pos.repeat(N,1,1).permute(0,2,1)).permute(0,2,1)
        flow = torch.reshape(flow,(-1,3))

        #apply affine and return
        hardcopy.pos = hardcopy.pos.repeat(N,1).float().to(self.device)
        hardcopy.pos = hardcopy.pos + flow
        return hardcopy


    def certify(self, x, n0: int, n: int, alpha: float, batch_size: int, plywrite=False) -> tuple((int, float)):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size,plywrite)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size,plywrite)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return SmoothFlow.ABSTAIN, 0.0, 0.5#0.5 for the radius to be zero
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius, pABar

    def predict(self, x, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return SmoothFlow.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x, num: int, batch_size,plywrite) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input (in this case, torch geometric Batch object containing 1 pointcloud )
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        pointcloudsize = x.ptr[1]
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for cert_batch_num in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                if self.certify_method == 'gaussianNoise':
                    batch = self._GenCloudGaussianNoise(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/gaussianNoise/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/gaussianNoise/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'rotationX':
                    batch = self._GenCloudRotationX(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/rotation/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/rotation/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'rotationY':
                    batch = self._GenCloudRotationY(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/rotation/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/rotation/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'rotationZ':
                    batch = self._GenCloudRotationZ(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/rotation/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/rotation/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'rotationXZ':
                    batch = self._GenCloudRotationXZ(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/rotation/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/rotation/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'rotation' or self.certify_method == 'rotationXYZ':
                    batch = self._GenCloudRotationXYZ(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/rotation/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/rotation/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'translation':
                    batch = self._GenCloudTranslation(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/translation/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/translation/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'shearing':
                    batch = self._GenCloudShearing(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/shearing/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/shearing/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'tapering':
                    batch = self._GenCloudTapering(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/tapering/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/tapering/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'twisting':
                    batch = self._GenCloudTwisting(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/twisting/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/twisting/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'squeezing':
                    batch = self._GenCloudSqueezing(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/squeezing/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/squeezing/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'stretching':
                    batch = self._GenCloudStretching(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/stretching/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/stretching/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True
                
                elif self.certify_method == 'affineNoTranslation':
                    batch = self._GenCloudAffineNoTranslation (x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/affineNoTranslation/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/affineNoTranslation/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'affine':
                    batch = self._GenCloudAffine (x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and plywrite and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/affine/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/affine/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                else:
                    raise Exception('Undefined certify_method!')
                

                predictions = self.base_classifier(batch).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

def write_ply(points, filename, text=True):
        """ input: Nx3, write points to filename as PLY format. """
        points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=text).write(filename)   
    
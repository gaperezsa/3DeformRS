import torch
import math
import numpy as np
from scipy.spatial.transform import Rotation

class DeformSample():
    r"""transform each pointcloud into a deformed version. wich deformation and wich sigma defined when initialized. 

    This version of the transformation is compliant with torch geometric transformations for the dataset and designed to receive a
    Torch geometric Data type object rather than a Batch object. Also, 

    Args:
        deformation (string): which deformation is going to be used. (default:RotationZ)
        sigma (float):noise hyperparameter for the normal / uniform sampling to be done. (default: 0.1)
    """
    def __init__(self, deformation="RotationZ", sigma=0.1):
        self.deformation = deformation
        self.sigma = sigma

    def __call__(self, data):
        if (self.deformation != 'none'):
            data = self.Transformer(deformation=self.deformation, x=data)

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(deform={self.deformation}, '
                f'sigma={self.sigma})')

    def Transformer(self,deformation,x):
        switcher = {
            "GaussianNoise"     : self._GenCloudGaussianNoise,
            "RotationX"         : self._GenCloudRotationX,
            "RotationY"         : self._GenCloudRotationY,
            "RotationZ"         : self._GenCloudRotationZ,
            "RotationXZ"        : self._GenCloudRotationXZ,
            "RotationXYZ"       : self._GenCloudRotationXYZ,
            "Translation"       : self._GenCloudTranslation,
            "Shearing"          : self._GenCloudShearing,
            "Tapering"          : self._GenCloudTapering,
            "Twisting"          : self._GenCloudTwisting,
            "Squeezing"         : self._GenCloudSqueezing,
            "Stretching"        : self._GenCloudStretching,
            "AffineNoTranslation": self._GenCloudAffineNoTranslation,
            "Affine"            : self._GenCloudAffine,
        }
        return switcher.get(deformation,"not a valid deformation")(x)

    def _GenCloudGaussianNoise(self,x):
        ''' This function returns a gaussian noised versions of the pointcloud x...
            meaning, every point will be randomy translated by some vector, l2 norm of said vector will be the certified radius.
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
            
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 

        #Gaussian distribution with std deviation sigma
        gaussianNoise = (torch.randn((pointCloudShape,3))*self.sigma).float()

        x.pos = x.pos + gaussianNoise
        return x

    def _GenCloudRotationX(self,x):
        ''' This function returns a rotated versions of the pointcloud x
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
            
        '''
        
        #Uniform between [-sigma, sigma]
        theta = (-2 * np.random.rand(3) + 1) *self.sigma  

        #null Y and Z rotations
        theta[1] = 0
        theta[2] = 0

        #rotate to get flow
        allRotations = torch.from_numpy(Rotation.from_euler('xyz', theta).as_matrix()).float()
        stackedPointcloud = x.pos
        flow = (torch.matmul(allRotations,stackedPointcloud.T) - stackedPointcloud.T).T
        flow = torch.reshape(flow,(-1,3))

        #apply rotation and return
        x.pos = x.pos + flow
        return x

    def _GenCloudRotationY(self,x):
        ''' This function returns a rotated versions of the pointcloud x
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
             
            
        '''
        #Uniform between [-sigma, sigma]
        theta = (-2 * np.random.rand(3) + 1) *self.sigma 

        #null X and Z rotations
        theta[0] = 0
        theta[2] = 0

        #rotate to get flow
        allRotations = torch.from_numpy(Rotation.from_euler('xyz', theta).as_matrix()).float()
        stackedPointcloud = x.pos
        flow = (torch.matmul(allRotations,stackedPointcloud.T) - stackedPointcloud.T).T
        flow = torch.reshape(flow,(-1,3))

        #apply rotation and return
        x.pos = x.pos + flow
        return x

    def _GenCloudRotationZ(self,x):
        ''' This function returns a rotated versions of the pointcloud x
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
             
            
        '''
        #Uniform between [-sigma, sigma]
        theta = (-2 * np.random.rand(3) + 1) *self.sigma 

        #null X and Y rotations
        theta[0] = 0
        theta[1] = 0

        #rotate to get flow
        allRotations = torch.from_numpy(Rotation.from_euler('xyz', theta).as_matrix()).float()
        stackedPointcloud = x.pos
        flow = (torch.matmul(allRotations,stackedPointcloud.T) - stackedPointcloud.T).T
        flow = torch.reshape(flow,(-1,3))

        #apply rotation and return
        x.pos = x.pos + flow
        return x

    def _GenCloudRotationXZ(self,x):
        ''' This function returns a rotated versions of the pointcloud x
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
             
            
        '''
        
        #Uniform between [-sigma, sigma]
        theta = (-2 * np.random.rand(3) + 1) *self.sigma 

        #null all Y rotations
        theta[1] = 0

        #rotate to get flow
        allRotations = torch.from_numpy(Rotation.from_euler('xyz', theta).as_matrix()).float()
        stackedPointcloud = x.pos
        flow = (torch.matmul(allRotations,stackedPointcloud.T) - stackedPointcloud.T).T
        flow = torch.reshape(flow,(-1,3))

        #apply rotation and return
        x.pos = x.pos + flow
        return x


    def _GenCloudRotationXYZ(self,x):
        ''' This function returns a rotated versions of the pointcloud x
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
             
            
        '''

        #Uniform between [-sigma, sigma]
        theta = (-2 * np.random.rand(3) + 1) *self.sigma 

        #rotate to get flow
        allRotations = torch.from_numpy(Rotation.from_euler('xyz', theta).as_matrix()).float()
        stackedPointcloud = x.pos
        flow = (torch.matmul(allRotations,stackedPointcloud.T) - stackedPointcloud.T).T
        flow = torch.reshape(flow,(-1,3))

        #apply rotation and return
        x.pos = x.pos + flow
        return x
        
    def _GenCloudTranslation(self,x):
        ''' This function returns a translated versions of the pointcloud x
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
             
            
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 
        
        #Gaussian distribution with std deviation sigma
        translations = (torch.randn(3)*self.sigma).float()

        #expand translations accordingly, one translation vector for every pointcloud
        flow = torch.tensor(translations).repeat(pointCloudShape,1).float()
        
        #apply translations and return
        x.pos = x.pos + flow
        return x


    def _GenCloudShearing(self,x):
        ''' This function returns a sheared versions of the pointcloud x
            shearing will be apllied on the x and y coordinate keeping z coordinate intact
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
             
            
        '''
        
        #Gaussian distribution with std deviation sigma
        #although 3 values generated, the third wont be used

        shearingCoeff = torch.randn(3)*self.sigma 
        #shearing is introducing the coefficients in the last column of the identity matrix and not changing the diagonal
        shearingMatrixs = torch.eye(3)
        shearingMatrixs[2,:2] = shearingCoeff[0:2]
        shearingMatrixs = shearingMatrixs.T.float()

        '''                     [[1        0       CoefA   ],
            shearingMatrix =    [0         1       CoefB   ],
                                [0         0       1       ]]
        '''
        
        #apply to get flow
        stackedPointcloud = x.pos
        flow = (torch.matmul(shearingMatrixs,stackedPointcloud.T) - stackedPointcloud.T).T
        flow = torch.reshape(flow,(-1,3))
        

        #apply shearing and return
        x.pos = x.pos + flow
        return x

    def _GenCloudTapering(self,x):

        ''' This function returns a tapered versions of the pointcloud x
            tapering will be apllied on the x and y coordinate keeping z coordinate intact
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
             
            


            in particular, for tapering, the trick will be having one transformation PER POINT rather than per pointcloud.
            this is because the tapering applied in the 3d Certify paper was a function of each Z coordinate of each point.
            Meaning the transformation matrix is higher order and not static given that it depends on the very point its going to be multiplied with.
            this can be noticed here: https://github.com/eth-sri/3dcertify/blob/master/transformations/tapering.py

            thsi funtion has been vectorized in order to compute the perturbed point faster, hence the use of one extra dimension
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 

        #Gaussian distribution with std deviation sigma
        TaperingCoeff = (torch.randn(2)*self.sigma)
        
        #preparing a,b and z to fit with the mask
        #same a,b for every 2*amountOfPointPerCloud because two positions in the diagonal of each matrix are gonna change
        #same z for every two positions
        z = x.pos[:, 2].repeat(2,1).T.flatten()
        a = TaperingCoeff[0].repeat(2*pointCloudShape,1).T.flatten()
        b = TaperingCoeff[1].repeat(2*pointCloudShape,1).T.flatten()


        boolMask = torch.tensor([[1,0,0],[0,1,0],[0,0,0]]).bool().repeat(pointCloudShape,1,1)
        TransformationMatrixs = torch.eye(3).repeat(pointCloudShape,1,1)
        TransformationMatrixs[boolMask] = torch.mul(torch.mul(0.5,torch.square(a)),z)+ torch.mul(b,z) + 1
        
        '''This gives matrixes that look something like this

            [[0.5*a^2*z+b*z+1 ,0                 ,0],
                [0               ,0.5*a^2*z+b*z+1   ,0]
                [0               ,0                 ,1]]
        
        one per point, meaning shape of TransformationMatrixs is N*pointCloudShape X 3 X 3
        '''
        
        #use transformation to get flow
        StackedPointcloud = torch.reshape(x.pos,(pointCloudShape,1,3)).permute(0,2,1)
        taperedPoints = torch.bmm(TransformationMatrixs,StackedPointcloud)
        flow = torch.reshape(taperedPoints,(-1,3))-x.pos

        #apply shearing and return
        x.pos = x.pos + flow
        return x

    def _GenCloudTwisting(self,x):
        ''' This function returns a twisted versions of the pointcloud x
            shearing will be apllied on the x and y coordinate keeping z coordinate intact
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
             
            

            in particular, for twisting, the trick will be having one transformation PER POINT rather than per pointcloud.
            this is because the twisting applied in the 3d Certify paper was a function of each Z coordinate of each point.
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 

        #Gaussian distribution with std deviation sigma
        twistingCoeff = (torch.randn(1)*self.sigma).float()
        

        #preparing alpha and z to fit with the mask
        #same alpha for every 4*amountOfPointPerCloud because 4 positions in the identity matrix are gonna change
        #same z for every 4 positions (4 position in this mask means a single point in the point cloud
        z = x.pos[:, 2].repeat(4,1).T.flatten()
        alpha = twistingCoeff[0].repeat(4*pointCloudShape,1).T.flatten()

        #create transformation matrixes
        boolMask = torch.tensor([[1,1,0],[1,1,0],[0,0,0]]).bool().repeat(pointCloudShape,1,1)
        TransformationMatrixs = torch.eye(3).repeat(pointCloudShape,1,1)
        angles = torch.mul(alpha,z)
        ''' remember 
            sin(alpha*z) = cos(alpha*z + pi/2)
            -sin(alpha*z) = cos(alpha*z - pi/2)
            
            meaning

            [[cos(alpha*z)    ,sin(alpha*z)      ,0],              [[cos(alpha*z)       ,cos(alpha*z + pi/2), 0]
                [-sin(alpha*z)   ,cos(alpha*z)      ,0]        =       [cos(alpha*z - pi/2),cos(alpha*z)       , 0] 
                [0               ,0                 ,1]]               [0                  ,0                  , 1]]
            
            '''
        transformer = torch.tensor([0,math.pi/2,-math.pi/2,0]).repeat(pointCloudShape).float()
        angles += transformer
        TransformationMatrixs[boolMask] = torch.cos(angles)
        
        #use transformation to get flow
        StackedPointcloud = torch.reshape(x.pos,(pointCloudShape,1,3)).permute(0,2,1)
        twistedPoints = torch.bmm(TransformationMatrixs,StackedPointcloud)
        flow = torch.reshape(twistedPoints,(-1,3))-x.pos

        #apply twisting and return
        x.pos = x.pos + flow
        return x

    def _GenCloudSqueezing(self,x):
        ''' This function returns a squeezed versions of the pointcloud x
            squeezing will be apllied by compressing the x coordinate and stretching the y and z coordinate accordingly
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
             
            

            x will be stretched by a factor K, so, Y and Z will be compressed by 1/sqrt(K)
        '''

        #Uniform between [-sigma, sigma]
        Kbar = (-2 * np.random.rand(1) + 1) *self.sigma

        #transforming uniform distributed variable
        #Kbar=0 -> compressingCoeffK = 1     Identity transform
        #Kbar=1 -> compressingCoeffK = 1/2   all x coordinates cut to half
        #Kbar=-1 -> compressingCoeffK = 1/2   all x coordinates cut to half
        #Kbar=2 -> compressingCoeffK = 1/3   all x coordinates cut to a third
        #Kbar=-2 -> compressingCoeffK = 1/3   all x coordinates cut to a third
        #compressing on the x coordinate by a 1/(|Kbar|+1) ratio and stretching y,z accordingly so that barycenter and volume is preserved
        compressingCoeffK = (1/(1+torch.abs(Kbar))).float()

        #preparing K and z to fit with the mask
        #same K for every three positions meaning one matrix during mask asignation, one matrix per one point cloud
        K = compressingCoeffK[0].repeat(3,1).T.flatten()
        divisor = (1 / torch.mul( compressingCoeffK[0] , torch.sqrt(compressingCoeffK[0]) ) ).repeat(3,1).T
        divisor[:,0] = 1
        divisor = divisor.flatten()

        '''K        = [k1              ,k1              ,k1             ,k2             ,k2             ,k2             ,...]
            divisor  = [1               ,1/k1*sqrt(k1)   ,1/k1*sqrt(k1)  ,1              ,1/k2*sqrt(k2)  ,1/k2*sqrt(k2)  ,...]
        
        '''
        #create transformation matrixes
        boolMask = torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).bool()
        TransformationMatrixs = torch.eye(3)
        TransformationMatrixs[boolMask] = torch.mul(K,divisor)
        
        '''This gives matrixes that look something like this

            [[k              ,0                 ,0        ],
            [0               ,1/sqrt(k)         ,0         ],
            [0               ,0                 ,1/sqrt(k)]]
        
        '''
        #apply to get flow
        stackedPointcloud = x.pos
        flow = (torch.matmul(TransformationMatrixs,stackedPointcloud.T) - stackedPointcloud.T).T
        flow = torch.reshape(flow,(-1,3))

        #apply twisting and return
        x.pos = x.pos + flow
        return x

    def _GenCloudStretching(self,x):
        ''' This function returns a stretched versions of the pointcloud x
            stretching will be apllied by stretching the x coordinate and compressing the y and z coordinate accordingly
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
             
            

            x will be stretched by a factor K, so, Y and Z will be compressed by 1/sqrt(K)
        '''

        #Uniform between [-sigma, sigma]
        Kbar = (-2 * np.random.rand(1) + 1) *self.sigma

        #transforming uniform distributed variable
        #Kbar=0 -> stretchingCoeffK = 1     Identity transform
        #Kbar=1 -> stretchingCoeffK = 2   all x coordinates doubled
        #Kbar=-1 -> stretchingCoeffK = 2   all x coordinates doubled
        #Kbar=2 -> stretchingCoeffK = 3   all x coordinates tripled
        #Kbar=-2 -> stretchingCoeffK = 3   all x coordinates tripled
        #stretching on the x coordinate by a |Kbar|+1 ratio and compressing y,z accordingly so that barycenter and volume is preserved
        stretchingCoeffK = (torch.abs(torch.from_numpy(Kbar))+1).float()

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
        boolMask = torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).bool()
        TransformationMatrixs = torch.eye(3)
        TransformationMatrixs[boolMask] = torch.mul(K,divisor)
        
        '''This gives matrixes that look something like this

            [[k              ,0                 ,0         ],
            [0               ,1/sqrt(k)         ,0         ],
            [0               ,0                 ,1/sqrt(k)]]
        
        '''
        #apply to get flow
        stackedPointcloud = x.pos
        flow = (torch.matmul(TransformationMatrixs,stackedPointcloud.T) - stackedPointcloud.T).T
        flow = torch.reshape(flow,(-1,3))

        #apply twisting and return
        x.pos = x.pos + flow
        return x

    def _GenCloudAffineNoTranslation(self,x):
        ''' This function returns a affine transformed versions of the pointcloud x with no translations
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
             
            
        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 

        #Uniform between [-sigma, sigma]
        affineCoeffs = (torch.randn((3,3))*self.sigma).float()

        boolMask = torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).bool()
        affineCoeffs[boolMask] = 1+affineCoeffs[boolMask]
        '''This gives affineCoeffs that look something like this

            [[1+a              ,b                 ,c           ],
            [d                 ,1+e               ,f           ],
            [g                 ,h                 ,1+i         ]]
        
        '''
        #apply to get flow
        stackedPointcloud = x.pos
        flow = (torch.matmul(affineCoeffs,stackedPointcloud.T) - stackedPointcloud.T).T
        flow = torch.reshape(flow,(-1,3))

        #apply affine with no translation and return
        x.pos = x.pos + flow
        return x

    def _GenCloudAffine(self,x):
        ''' This function returns a affine transformed versions of the pointcloud x
            x: pytorch geometric Data type object containing the info of a single point_cloud_shape
             
            

        '''
        #amount of points in one point cloud
        pointCloudShape = x.pos.shape[0] 

        #Uniform between [-sigma, sigma]
        affineCoeffs = (torch.randn((3,4))*self.sigma).float()

        boolMask = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).bool()
        affineCoeffs[boolMask] = 1+affineCoeffs[boolMask]
        '''This gives affineCoeffs that look something like this

            [[1+a              ,b                 ,c           j],
            [d                 ,1+e               ,f           k],
            [g                 ,h                 ,1+i         l]]]
        
        '''

        StackedPointcloud = torch.cat((x.pos,torch.ones(pointCloudShape).unsqueeze(0).T.float() ),1)

        '''This gives StackedPointcloud that look something like this

            [[[x1               ,y1                 ,z1                  ,1       ],
                [x2               ,y2                 ,z2                   ,1       ],
                [x3               ,y3                 ,z3                   ,1       ],
                [x4               ,y4                 ,z4                   ,1       ],
                ... 
                xpointCloudShape  ,ypointCloudShape,  ,zpointCloudShape     ,1       ]]]
        
        '''

        #multiply and get flow
        flow = (torch.matmul(affineCoeffs,StackedPointcloud.T) - x.pos.T).T
        flow = torch.reshape(flow,(-1,3))

        #apply affine and return
        x.pos = x.pos + flow
        return x
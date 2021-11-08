import argparse
import os.path as osp
import os
import numpy as np
import math
import torch
from scipy.spatial.transform import Rotation
from plyfile import PlyElement, PlyData

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def _GenCloudGaussianNoise(x,perturbation):
    ''' This function returns a gaussian noised versions of the pointcloud x...
        meaning, every point will be randomy translated by some vector, l2 norm of said vector will be the certified radius.
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved
    '''
    #amount of points in one point cloud
    pointCloudShape = x.pos.shape[0] 

    #Gaussian distribution with std deviation sigma
    gaussianNoise = (torch.randn((pointCloudShape,3))*perturbation).float()

    x.pos = x.pos + gaussianNoise
    return x

def _GenCloudRotationX(x,perturbation):
    ''' This function returns a rotated versions of the pointcloud x
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved
    '''
    
    #Uniform between [-sigma, sigma]
    theta = perturbation 

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

def _GenCloudRotationY(x,perturbation):
    ''' This function returns a rotated versions of the pointcloud x
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved
    '''
    #Uniform between [-sigma, sigma]
    theta = perturbation

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

def _GenCloudRotationZ(x,perturbation):
    ''' This function returns a rotated versions of the pointcloud x
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved
    '''
    #Uniform between [-sigma, sigma]
    theta = perturbation

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

def _GenCloudRotationXZ(x,perturbation):
    ''' This function returns a rotated versions of the pointcloud x
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved
    '''
    
    #Uniform between [-sigma, sigma]
    theta = perturbation

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


def _GenCloudRotationXYZ(x,perturbation):
    ''' This function returns a rotated versions of the pointcloud x
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved
    '''

    #Uniform between [-sigma, sigma]
    theta = perturbation

    #rotate to get flow
    allRotations = torch.from_numpy(Rotation.from_euler('xyz', theta).as_matrix()).float()
    stackedPointcloud = x.pos
    flow = (torch.matmul(allRotations,stackedPointcloud.T) - stackedPointcloud.T).T
    flow = torch.reshape(flow,(-1,3))

    #apply rotation and return
    x.pos = x.pos + flow
    return x
    
def _GenCloudTranslation(x,perturbation):
    ''' This function returns a translated versions of the pointcloud x
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved
    '''
    #amount of points in one point cloud
    pointCloudShape = x.pos.shape[0] 
    
    #Gaussian distribution with std deviation sigma
    translations = perturbation

    #expand translations accordingly, one translation vector for every pointcloud
    flow = torch.tensor(translations).repeat(pointCloudShape,1).float()
    
    #apply translations and return
    x.pos = x.pos + flow
    return x


def _GenCloudShearing(x,perturbation):
    ''' This function returns a sheared versions of the pointcloud x
        shearing will be apllied on the x and y coordinate keeping z coordinate intact
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved
    '''
    
    #Gaussian distribution with std deviation sigma
    shearingCoeff = perturbation

    #shearing is introducing the coefficients in the last column of the identity matrix and not changing the diagonal
    shearingMatrixs = torch.eye(3)
    shearingMatrixs[2,:2] = shearingCoeff[0,:2]
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

def _GenCloudTapering(x,perturbation):

    ''' This function returns a tapered versions of the pointcloud x
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
    TaperingCoeff = perturbation
    
    #preparing a,b and z to fit with the mask
    #same a,b for every 2*amountOfPointPerCloud because two positions in the diagonal of each matrix are gonna change
    #same z for every two positions
    z = x.pos[:, 2].repeat(2,1).T.flatten()
    a = TaperingCoeff[:,0].repeat(2*pointCloudShape,1).T.flatten()
    b = TaperingCoeff[:,1].repeat(2*pointCloudShape,1).T.flatten()


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

def _GenCloudTwisting(x,perturbation):
    ''' This function returns a twisted versions of the pointcloud x
        shearing will be apllied on the x and y coordinate keeping z coordinate intact
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved

        in particular, for twisting, the trick will be having one transformation PER POINT rather than per pointcloud.
        this is because the twisting applied in the 3d Certify paper was a function of each Z coordinate of each point.
    '''
    #amount of points in one point cloud
    pointCloudShape = x.pos.shape[0] 

    #Gaussian distribution with std deviation sigma
    twistingCoeff = perturbation
    

    #preparing alpha and z to fit with the mask
    #same alpha for every 4*amountOfPointPerCloud because 4 positions in the identity matrix are gonna change
    #same z for every 4 positions (4 position in this mask means a single point in the point cloud
    z = x.pos[:, 2].repeat(4,1).T.flatten()
    alpha = twistingCoeff[:,0].repeat(4*pointCloudShape,1).T.flatten()

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

def _GenCloudSqueezing(x,perturbation):
    ''' This function returns a squeezed versions of the pointcloud x
        squeezing will be apllied by compressing the x coordinate and stretching the y and z coordinate accordingly
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved

        x will be stretched by a factor K, so, Y and Z will be compressed by 1/sqrt(K)
    '''

    #Uniform between [-sigma, sigma]
    Kbar = perturbation

    #transforming uniform distributed variable
    #Kbar=0 -> compressingCoeffK = 1     Identity transform
    #Kbar=1 -> compressingCoeffK = 1/2   all x coordinates cut to half
    #Kbar=-1 -> compressingCoeffK = 1/2   all x coordinates cut to half
    #Kbar=2 -> compressingCoeffK = 1/3   all x coordinates cut to a third
    #Kbar=-2 -> compressingCoeffK = 1/3   all x coordinates cut to a third
    #compressing on the x coordinate by a 1/(|Kbar|+1) ratio and stretching y,z accordingly so that barycenter and volume is preserved
    compressingCoeffK = (1/(1+torch.abs(torch.from_numpy(Kbar)))).float()

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

def _GenCloudStretching(x,perturbation):
    ''' This function returns a stretched versions of the pointcloud x
        stretching will be apllied by stretching the x coordinate and compressing the y and z coordinate accordingly
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved

        x will be stretched by a factor K, so, Y and Z will be compressed by 1/sqrt(K)
    '''
    #amount of points in one point cloud
    pointCloudShape = x.pos.shape[0] 

    #Uniform between [-sigma, sigma]
    Kbar = perturbation

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

def _GenCloudAffineNoTranslation(x,perturbation):
    ''' This function returns a affine transformed versions of the pointcloud x with no translations
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved
    '''
    #amount of points in one point cloud
    pointCloudShape = x.pos.shape[0] 

    #Uniform between [-sigma, sigma]
    affineCoeffs = perturbation

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

def _GenCloudAffine(x,perturbation):
    ''' This function returns a affine transformed versions of the pointcloud x
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved

    '''
    #amount of points in one point cloud
    pointCloudShape = x.pos.shape[0] 

    #Uniform between [-sigma, sigma]
    affineCoeffs = perturbation

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

    #batch multiply and get flow
    flow = (torch.matmul(affineCoeffs,StackedPointcloud.T) - x.pos.T).T
    flow = torch.reshape(flow,(-1,3))

    #apply affine and return
    x.pos = x.pos + flow
    return x


certification_method_choices = ['rotationX','rotationY','rotationZ','rotationXZ','rotationXYZ','translation','shearing','tapering','twisting','squeezing','stretching','gaussianNoise','affine','affineNoTranslation'] 

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--sample_class", type=float, default=0 , help="0-39, which class to sample")
parser.add_argument('--num_points', type=int, default=1024,help='num of points to use in case of curvenet, default 1024 recommended')
parser.add_argument("--data_path", type=str, default='../Pointnet2andDGCNN/Data/Modelnet40fp',help="path to dataset")
parser.add_argument("--deformation_method", type=str, default='rotationXYZ', required=True, choices=certification_method_choices, help='type of certification for certification')
parser.add_argument("--perturbation_amount", type=float, nargs='+',help="perturbation parameter, RotationX would only use 1, affine will use 12")
args = parser.parse_args()


if not os.path.exists('../output/samples/'+args.deformation_method):
    os.makedirs('../output/samples/'+args.deformation_method, exist_ok=True)

#dataset and loaders
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
path = osp.join(osp.dirname(osp.realpath(__file__)), args.data_path)
pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
print(path)
test_dataset = ModelNet(path, '40', False, transform, pre_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# iterate through the dataset
for iterator in test_loader:
    if iterator.y.item() == args.sample_class:
        sample = iterator
        break;


['rotationX','rotationY','rotationZ','rotationXZ','rotationXYZ','translation','shearing',
'tapering','twisting','squeezing','stretching','gaussianNoise','affine','affineNoTranslation']

def Transformer(deformation,x,perturbation):
    switcher = {
        "gaussianNoise"     : _GenCloudGaussianNoise,
        "rotationX"         : _GenCloudRotationX,
        "rotationY"         : _GenCloudRotationY,
        "rotationZ"         : _GenCloudRotationZ,
        "rotationXZ"        : _GenCloudRotationXZ,
        "rotationXYZ"       : _GenCloudRotationXYZ,
        "translation"       : _GenCloudTranslation,
        "shearing"          : _GenCloudShearing,
        "tapering"          : _GenCloudTapering,
        "twisting"          : _GenCloudTwisting,
        "squeezing"         : _GenCloudSqueezing,
        "stretching"        : _GenCloudStretching,
        "affineNoTranslation": _GenCloudAffineNoTranslation,
        "affine"            : _GenCloudAffine,
    }
    return switcher.get(deformation,"not a valid deforamtion with defined hypervolume")(x,perturbation)

PC = sample.pos[0:args.num_points].cpu().detach().numpy()
write_ply(PC, '../output/samples/'+args.deformation_method+'/Original.ply')
sample = Transformer(args.deformation_method,sample,args.perturbation_amount)
PC = sample.pos[0:args.num_points].cpu().detach().numpy()
write_ply(PC, '../output/samples/'+args.deformation_method+'/Perturbed.ply')


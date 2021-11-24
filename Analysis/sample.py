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
    shearingCoeff = torch.tensor(perturbation).float()

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
    TaperingCoeff = torch.tensor(perturbation).float()
    
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
    twistingCoeff = torch.tensor(perturbation).float()
    

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

def _GenCloudSqueezing(x,perturbation):
    ''' This function returns a squeezed versions of the pointcloud x
        squeezing will be apllied by compressing the x coordinate and stretching the y and z coordinate accordingly
        x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
        N: int 
        counter: int just to cehck if the original pointcloud should be conserved

        x will be stretched by a factor K, so, Y and Z will be compressed by 1/sqrt(K)
    '''

    #Uniform between [-sigma, sigma]
    Kbar = torch.tensor(perturbation).float()

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


certification_method_choices = ['RotationX','RotationY','RotationZ','RotationXZ','RotationXYZ','Translation','Shearing','Tapering','Twisting','Squeezing','Stretching','GaussianNoise','Affine','AffineNoTranslation'] 

saveFolder = '/home/santamgp/Downloads/CVPRGraphics/SupplementaryMaterial/QualitativeResultsPool/'

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--sample_class", type=int, default=0 , help="0-39, which class to sample")
parser.add_argument('--num_points', type=int, default=2048,help='num of points to use, default 1024 recommended')
parser.add_argument("--data_path", type=str, default='../Data/PointNet2andDGCNN/Modelnet40fp',help="path to dataset")
parser.add_argument("--deformation_method", type=str, default='RotationXYZ', required=True, choices=certification_method_choices, help='type of certification for certification')
parser.add_argument("--perturbation_amount", type=float, nargs='+',help="perturbation parameter, RotationX would only use 1, affine will use 12")
parser.add_argument('--test', action='store_true', default=False, help='test what the smoothed versions of each network classifiy this instance as')
parser.add_argument('--search', action='store_true', default=False, help='keep looking for other amount that sum up the same')
parser.add_argument("--sigma", type=float,help="which sigma to smooth the networks with")
parser.add_argument("--certify_betch_sz", type=int,default=64,help="which sigma to smooth the networks with")

args = parser.parse_args()

# For rotaions to transform the angles to [0, pi]
if args.deformation_method[0:8] == 'rotation' or args.deformation_method[0:8] == 'Rotation':
        args.sigma *= math.pi 

if not os.path.exists('../output/samples/'+args.deformation_method):
    os.makedirs('../output/samples/'+args.deformation_method, exist_ok=True)

#dataset and loaders
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader as TorchGeometricDataLoader
path = osp.join(osp.dirname(osp.realpath(__file__)), args.data_path)
pre_transform, transform = T.NormalizeScale(), T.SamplePoints(args.num_points)
print(path)
test_dataset = ModelNet(path, '40', False, transform, pre_transform)
torch_geometric_test_loader = TorchGeometricDataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# iterate through the dataset
for iterator in torch_geometric_test_loader:
    if iterator.y.item() == args.sample_class:
        original_sample = iterator
        break;

def Transformer(deformation,x,perturbation):
    switcher = {
        "GaussianNoise"     : _GenCloudGaussianNoise,
        "RotationX"         : _GenCloudRotationX,
        "RotationY"         : _GenCloudRotationY,
        "RotationZ"         : _GenCloudRotationZ,
        "RotationXZ"        : _GenCloudRotationXZ,
        "RotationXYZ"       : _GenCloudRotationXYZ,
        "Translation"       : _GenCloudTranslation,
        "Shearing"          : _GenCloudShearing,
        "Tapering"          : _GenCloudTapering,
        "Twisting"          : _GenCloudTwisting,
        "Squeezing"         : _GenCloudSqueezing,
        "Stretching"        : _GenCloudStretching,
        "AffineNoTranslation": _GenCloudAffineNoTranslation,
        "Affine"            : _GenCloudAffine,
    }
    return switcher.get(deformation,"not a valid deformation")(x,perturbation)

PC = original_sample.pos[0:args.num_points].cpu().detach().numpy()
write_ply(PC,saveFolder+'sampleClass'+str(args.sample_class)+'Original.ply')
sample = Transformer(args.deformation_method,original_sample,args.perturbation_amount)
PC = sample.pos[0:args.num_points].cpu().detach().numpy()
write_ply(PC,saveFolder+'sampleClass'+str(args.sample_class)+args.deformation_method+str(args.perturbation_amount)+'.ply')

if args.test:

    print("ah shit, here we go again")

    PointNet2_base_classifier_path = "/home/santamgp/Documents/CertifyingAffineTransformationsOnPointClouds/3D-RS-PointCloudCertifying/Pointnet2andDGCNN/trainedModels/pointnetBaseline/FinalModel.pth.tar"
    DGCNN_base_classifier_path = "/home/santamgp/Documents/CertifyingAffineTransformationsOnPointClouds/3D-RS-PointCloudCertifying/Pointnet2andDGCNN/trainedModels/dgcnnBaseline/FinalModel.pth.tar"

    #use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #send sample to device
    sample = sample.to(device)

    #Import Smooth Classifier
    import sys
    sys.path.insert(0, "/home/santamgp/Documents/CertifyingAffineTransformationsOnPointClouds/3D-RS-PointCloudCertifying/")
    from SmoothedClassifiers.Pointnet2andDGCNN.SmoothFlow import SmoothFlow as PN2_DGCNN_SmoothFlow

    num_classes = 40


    #PointNet++:
    from Pointnet2andDGCNN.Trainers.pointnet2Train import Net as PointNet2_Net

    #model and optimizer
    Pointnet2_base_classifier = PointNet2_Net(num_classes).to(device)
    Pointnet2_optimizer = torch.optim.Adam(Pointnet2_base_classifier.parameters(), lr=0.001)

    #loadTrainedModel
    checkpoint = torch.load(PointNet2_base_classifier_path)
    Pointnet2_base_classifier.load_state_dict(checkpoint['model_param'])
    Pointnet2_optimizer.load_state_dict(checkpoint['optimizer'])

    # create the smooothed classifier g
    Pointnet2_smoothed_classifier = PN2_DGCNN_SmoothFlow(Pointnet2_base_classifier, num_classes, args.deformation_method, args.sigma)

    #predicting
    print("Calculating PointNet++ prediction")
    PointNet2Prediction = Pointnet2_smoothed_classifier.predict(sample,1000,0.001,args.certify_betch_sz)
    print(f"PointNet++ prediction: class {PointNet2Prediction}, ground truth: {sample.y.item()}")

    #free cuda memory
    Pointnet2_base_classifier.to("cpu")

    #DGCNN:
    from Pointnet2andDGCNN.Trainers.dgcnnTrain import Net as DGCNN_Net

    #model and optimizer
    DGCNN_base_classifier = DGCNN_Net(num_classes, k=20).to(device)
    DGCNN_optimizer = torch.optim.Adam(DGCNN_base_classifier.parameters(), lr=0.001)
    DGCNN_scheduler = torch.optim.lr_scheduler.StepLR(DGCNN_optimizer, step_size=20, gamma=0.5)

    #loadTrainedModel
    checkpoint = torch.load(DGCNN_base_classifier_path)
    DGCNN_base_classifier.load_state_dict(checkpoint['model_param'])
    DGCNN_optimizer.load_state_dict(checkpoint['optimizer'])
    DGCNN_scheduler.load_state_dict(checkpoint['scheduler'])

    # create the smooothed classifier g
    DGCNN_smoothed_classifier = PN2_DGCNN_SmoothFlow(DGCNN_base_classifier, num_classes, args.deformation_method, args.sigma)

    #predicting
    print("Calculating DGCNN prediction")
    DGCNNPrediction = DGCNN_smoothed_classifier.predict(sample,1000,0.001,args.certify_betch_sz)
    print(f"DGCNN prediction: class {DGCNNPrediction}, ground truth: {sample.y.item()}")

    #free cuda memory
    DGCNN_base_classifier.to("cpu")


    if args.search:
        
        
        while (PointNet2Prediction == DGCNNPrediction):

            #find L1 or L2 norm of perturbation_amount depending on the perturbation_amount
            if args.deformation_method[0:8] == 'rotation' or args.deformation_method[0:8] == 'Rotation':
                targetSum = np.linalg.norm(np.array(args.perturbation_amount),ord=1)
            else:
                targetSum = np.linalg.norm(np.array(args.perturbation_amount),ord=2)

            #dirichlet distribution makes it so that this array sums almost exactly one
            testing_perturbation = np.squeeze(np.random.dirichlet(np.ones(len(args.perturbation_amount)),size=1))

            #fit to desired perturbation sum
            testing_perturbation *= targetSum

            print(f"\nwith perturbation parameters {testing_perturbation} ...")

            #sample with this perturbation
            sample = sample.to("cpu")
            sample = Transformer(args.deformation_method,original_sample,testing_perturbation)
            sample = sample.to(device)

            #predicting with PointNet++
            Pointnet2_base_classifier.to(device)
            print("Calculating PointNet++ prediction")
            PointNet2Prediction = Pointnet2_smoothed_classifier.predict(sample,1000,0.001,args.certify_betch_sz)
            print(f"PointNet++ prediction: class {PointNet2Prediction}, ground truth: {sample.y.item()}")

            #free cuda memory
            Pointnet2_base_classifier.to("cpu")

            #predicting with DGCNN++
            DGCNN_base_classifier.to(device)
            print("Calculating DGCNN prediction")
            DGCNNPrediction = DGCNN_smoothed_classifier.predict(sample,1000,0.001,args.certify_betch_sz)
            print(f"DGCNN prediction: class {DGCNNPrediction}, ground truth: {sample.y.item()}")

            #free cuda memory
            DGCNN_base_classifier.to("cpu")
    
    

    
    


    
    
    



    '''
    #CurveNet
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
    from CurveNet.core.data import ModelNet40
    from CurveNet.core.models.curvenet_cls import CurveNet
    import numpy as np
    from torch.utils.data import DataLoader as CurveNetDataLoader
    from CurveNet.core.util import cal_loss, IOStream
    import sklearn.metrics as metrics

    CurveNet_test_loader = CurveNetDataLoader(ModelNet40(partition='test', num_points=args.num_points),batch_size=1, shuffle=False, drop_last=False)

    #declare and load pretrained model
    base_classifier = CurveNet(num_classes=num_classes).to(device)
    base_classifier = nn.DataParallel(base_classifier)
    base_classifier.load_state_dict(torch.load(args.base_classifier_path))
    base_classifier.eval()

    #PointNet

    import sys
    sys.path.insert(0, osp.join(osp.dirname(osp.realpath(__file__)),'Pointnet'))
    #sys.path.insert(0, "/home/santamgp/Documents/CertifyingAffineTransformationsOnPointClouds/3D-RS-PointCloudCertifying/Pointnet")

    from Pointnet.DataLoaders import datasets
    from torch.utils.data import DataLoader
    from Pointnet.model import PointNet

    test_data = datasets.modelnet40(num_points=args.num_points, split='test', rotate='none')

    PointNet_test_loader = DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    '''


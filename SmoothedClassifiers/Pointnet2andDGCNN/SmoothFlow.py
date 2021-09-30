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

    def _GenDeformGaussian(self, imgs, N, device):
        ''' This function takes an image C x W x H and returns N Gaussianly perturbed coordinates versions
        '''       
        batch = imgs.repeat((N, 1, 1, 1))
        num_channels, rows, cols = imgs.shape
        randomFlow = torch.randn(N, rows, cols, 2, device=device) * self.sigma

        new_ros = torch.linspace(-1, 1, rows)
        new_cols = torch.linspace(-1, 1, cols)

        meshx, meshy = torch.meshgrid((new_ros, new_cols))
        grid = torch.stack((meshy, meshx), 2).unsqueeze(0).expand(N, rows, cols, 2).to(device)

        new_grid = grid + randomFlow

        Iwarp = F.grid_sample(batch, new_grid)
        return Iwarp


    def _GenCloudRotation(self, x, N,counter):
        ''' This function returns N rotated versions of the pointcloud x
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
        '''
        theta = (-2 * np.random.rand(N,3) + 1) *self.sigma #Uniform between [-sigma, sigma]
        if counter==0:
            theta[0] = [0,0,0] #keep the original pointcloud as the first example


        pointCloudShape = x.pos.shape[0] #amount of points in one point cloud
        hardcopy = copy.deepcopy(x)
        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)
        builder = []

        for i in range(N): # for every new perturbed sample desired

            #rotation that will be applied to point cloud
            rotationRepresentation: Rotation = Rotation.from_euler('xyz', theta[i])
            rotationMatrix = rotationRepresentation.as_matrix()
            rotationMatrix = torch.from_numpy(rotationMatrix.T).float().to(self.device)
            builder.append(rotationMatrix)
        
        #apply rotation and return
        allRotations = torch.cat([x.unsqueeze(0) for x in builder])
        StackedPointcloud = hardcopy.pos.repeat(N,1,1)
        rotatedPoints = torch.bmm(StackedPointcloud,allRotations)
        hardcopy.pos = torch.reshape(rotatedPoints,(-1,3))
        

        return hardcopy
        
    def _GenCloudTranslation(self, x, N,counter):
        ''' This function returns N translated versions of the pointcloud x
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
        '''

        translations = torch.randn((N, 3))*self.sigma
        translations = translations.to(self.device)
        pointCloudShape = x.pos.shape[0] #amount of points in one point cloud
        hardcopy = copy.deepcopy(x)
        if counter==0:
            translations[0] = torch.tensor([0,0,0]).float().to(self.device) #keep the original pointcloud as the first example

        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)
        builder = []

        '''
        for translationMatrix in translations: # for every new perturbed sample desired

            #translation that will be applied to point cloud
            builder.append(translationMatrix.repeat(pointCloudShape,1))
        '''

        builder = [x.repeat(pointCloudShape,1) for x in translations]
        
        #apply translation and return
        allTranslations = torch.cat([x for x in builder]).float().to(self.device)
        translatedPoints = hardcopy.pos.repeat(N,1) + allTranslations
        hardcopy.pos = translatedPoints
        

        return hardcopy


    def _GenCloudShearing(self, x, N,counter):
        ''' This function returns N sheared versions of the pointcloud x
            shearing will be apllied on the x and y coordinate keeping z coordinate intact
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 
        '''

        shearingCoeff = torch.randn((N,1,3))*self.sigma #although 3 values generated, the third wont be used

        pointCloudShape = x.pos.shape[0] #amount of points in one point cloud

        hardcopy = copy.deepcopy(x)
        if counter==0:
            shearingCoeff[0] = torch.tensor([[0,0,0]]).float().to(self.device) #keep the original pointcloud as the first example

        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)

        shearingMatrixs = torch.eye(3).unsqueeze(0).repeat(N,1,1)
        shearingMatrixs[:,2,:2] = shearingCoeff[:,0,:2]
        shearingMatrixs = shearingMatrixs.float().to(self.device)
        
        StackedPointcloud = hardcopy.pos.repeat(N,1,1)
        ShearedPoints = torch.bmm(StackedPointcloud,shearingMatrixs)
        hardcopy.pos = torch.reshape(ShearedPoints,(-1,3))
        

        return hardcopy

    def _GenCloudTapering(self, x, N,counter):

        ''' This function returns N tapered versions of the pointcloud x
            tapering will be apllied on the x and y coordinate keeping z coordinate intact
            x: pytorch geometric Batch type object containing the info of a single point_cloud_shape
            N: int 


            in particular, for tapering, the trick will be having one transformation PER POINT rather than per pointcloud.
            this is because the tapering applied in the 3d Certify paper was a function of each Z coordinate of each point.
            Meaning the transformation matrix is higher order and not static given that it depends on the very point its going to be multiplied with.
            this can be noticed here: https://github.com/eth-sri/3dcertify/blob/master/transformations/tapering.py

            thsi funtion has been vectorized in order to compute the perturbed point faster, hence the use of one extra dimension
        '''

        TaperingCoeff = (torch.randn((N, 2))*self.sigma).to(self.device)

        pointCloudShape = x.pos.shape[0] #amount of points in one point cloud

        hardcopy = copy.deepcopy(x)
        if counter==0:
            TaperingCoeff[0] = torch.tensor([0,0]).float().to(self.device) #keep the original pointcloud as the first example

        hardcopy.batch = torch.arange(N).unsqueeze(1).expand(N, pointCloudShape).flatten().type(torch.LongTensor).to(self.device)
        hardcopy.ptr = (torch.arange(N+1) * pointCloudShape).type(torch.LongTensor).to(self.device)
        hardcopy.y = hardcopy.y.expand(N)

        #have all points that are going to be altered for computing the tapering matrix
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

            [[0.5*a^2*z+b*z+1],0                 ,0],
             [0               ,0.5*a^2*z+b*z+1   ,0]
             [0               ,0                 ,1]]
        
        one per point, meaning shape of TransformationMatrixs is N*pointCloudShape X 3 X 3
        '''
            
        StackedPointcloud = torch.reshape(hardcopy.pos,(N*pointCloudShape,1,3))
        taperedPoints = torch.bmm(StackedPointcloud,TransformationMatrixs)
        hardcopy.pos = torch.reshape(taperedPoints,(-1,3))

        return hardcopy
    
    def _GenImageScalingUniform(self, x, N):
        _, rows, cols = x.shape # N is the batch size
        #Scaling here is sampled from uniform distribution between [1-sigma, 1+sigma]
        scale = (-2 * torch.rand((N, 1, 1)) + 1.0) * self.sigma + 1.0
        #Generating the vector field for scaling.
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = X * scale - X
        Yv = Y * scale - Y
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(self.device)
        grid = torch.stack((Y,X), axis=3).to(self.device)
        
        return F.grid_sample(x.repeat((N, 1, 1, 1)), grid+randomFlow)

    def _GenImageAffine(self, x, N):
        _, rows, cols = x.shape # N is the batch size
        
        params = torch.randn((6, N, 1, 1))*self.sigma

        #Generating the vector field for Affine transformation.
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = params[0]*X + params[1]*Y + params[2]
        Yv = params[3]*X + params[4]*Y + params[5]
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(self.device)
        grid = torch.stack((Y,X), axis=3).to(self.device)
        
        return F.grid_sample(x.repeat((N, 1, 1, 1)), grid+randomFlow)

    def _GenImageDCT(self, x, N):

            _, rows, cols = x.shape
            new_ros = torch.linspace(-1, 1, rows)
            new_cols = torch.linspace(-1, 1, cols)
            meshx, meshy = torch.meshgrid((new_ros, new_cols))
            grid = torch.stack((meshy, meshx), 2).unsqueeze(0).expand(N, rows, cols, 2).to(self.device)

            X, Y = torch.meshgrid((new_ros, new_cols))
            X = torch.reshape(X, (1, 1, 1, rows, cols))
            Y = torch.reshape(Y, (1, 1, 1, rows, cols))

            param_ab = torch.randn(N, self.num_bases, self.num_bases, 1, 2) * self.sigma
            a = param_ab[:, :, :, :, 0].unsqueeze(4)
            b = param_ab[:, :, :, :, 1].unsqueeze(4)
            K1 = torch.arange(self.num_bases).view(1, self.num_bases, 1, 1, 1)
            K2 = torch.arange(self.num_bases).view(1, 1, self.num_bases, 1, 1)
            basis_factors  = torch.cos( math.pi* (K1 * (X+0.5/rows) ))*torch.cos( math.pi * (K2 * (Y+0.5/cols)))

            U = torch.squeeze(torch.sum(a * basis_factors, dim=(1, 2)))
            V = torch.squeeze(torch.sum(b * basis_factors, dim=(1, 2)))

            randomFlow = torch.stack((V, U), dim=3).to(self.device)

            return F.grid_sample(x.repeat((N, 1, 1, 1)), grid + randomFlow)

    def certify(self, x, n0: int, n: int, alpha: float, batch_size: int) -> tuple((int, float)):
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
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
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

    def _sample_noise(self, x, num: int, batch_size) -> np.ndarray:
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
                if self.certify_method == 'rotation':
                    batch = self._GenCloudRotation(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/rotation/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/rotation/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'translation':
                    batch = self._GenCloudTranslation(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/translation/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/translation/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'shearing':
                    batch = self._GenCloudShearing(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/shearing/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/shearing/'+self.exp_name+'Perturbed.ply')
                        self.plywritten = True

                elif self.certify_method == 'tapering':
                    batch = self._GenCloudTapering(x, this_batch_size,cert_batch_num)

                    #write as ply the original and a perturbed pointcloud
                    if (not self.plywritten) and this_batch_size > 2:
                        PC = batch.pos[0:pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/tapering/'+self.exp_name+'Original.ply')
                        PC =batch.pos[pointcloudsize:2*pointcloudsize].cpu().detach().numpy()
                        write_ply(PC, 'output/samples/tapering/'+self.exp_name+'Perturbed.ply')
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
    
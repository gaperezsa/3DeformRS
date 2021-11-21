import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.spatial.transform import Rotation

certification_method_choices = ['rotationX','rotationY','rotationZ','rotationXZ','translation','shearing','tapering','twisting','squeezing','stretching','gaussianNoise','affine','affineNoTranslation'] 

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--deformation", type=str, default='rotationXYZ', required=True, choices=certification_method_choices, help='type of deforamtion to base the field upon')
parser.add_argument('--parameters', type=float, default=1,nargs='+',help='arguments to be used as parameter in the deformation')
args = parser.parse_args()

def PaintGaussianNoise():

    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.4))

    # Make the direction data for the arrows

    u = np.random.randn(x.shape[0],x.shape[1],x.shape[2])
    v = np.random.randn(y.shape[0],y.shape[1],y.shape[2])
    w = np.random.randn(z.shape[0],z.shape[1],z.shape[2])
    ax.quiver(x, y, z, u, v, w, length=0.1)

    plt.show()

def PaintXRot(angle):
    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.4),
                        np.arange(-0.8, 1, 0.4),
                        np.arange(-0.8, 1, 0.4))

    # Make the direction data for the arrows
    Matrix = Rotation.from_euler('xyz', (angle,0,0)).as_matrix()

    u = (Matrix[0,0]*x + Matrix[0,1]*y + Matrix[0,2]*z)-x
    v = (Matrix[1,0]*x + Matrix[1,1]*y + Matrix[1,2]*z)-y
    w = (Matrix[2,0]*x + Matrix[2,1]*y + Matrix[2,2]*z)-z

    ax.quiver(x, y, z, u, v, w, length=0.9, normalize=False)

    plt.xticks([])
    plt.yticks([])
    plt.show()

def PaintYRot(angle):
    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.4))

    # Make the direction data for the arrows
    Matrix = Rotation.from_euler('xyz', (0,angle,0)).as_matrix()

    u = (Matrix[0,0]*x + Matrix[0,1]*y + Matrix[0,2]*z)-x
    v = (Matrix[1,0]*x + Matrix[1,1]*y + Matrix[1,2]*z)-y
    w = (Matrix[2,0]*x + Matrix[2,1]*y + Matrix[2,2]*z)-z

    ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

    plt.show()

def PaintZRot(angle):
    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.4))

    # Make the direction data for the arrows
    Matrix = Rotation.from_euler('xyz', (0,0,angle)).as_matrix()

    u = (Matrix[0,0]*x + Matrix[0,1]*y + Matrix[0,2]*z)-x
    v = (Matrix[1,0]*x + Matrix[1,1]*y + Matrix[1,2]*z)-y
    w = (Matrix[2,0]*x + Matrix[2,1]*y + Matrix[2,2]*z)-z

    ax.quiver(x, y, z, u, v, w, length=0.2, normalize=True)

    plt.show()

def PaintXZRot(angleX,angleZ):
    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.4),
                        np.arange(-0.8, 1, 0.4),
                        np.arange(-0.8, 1, 0.4))

    # Make the direction data for the arrows
    Matrix = Rotation.from_euler('xyz', (angleX,0,angleZ)).as_matrix()

    u = (Matrix[0,0]*x + Matrix[0,1]*y + Matrix[0,2]*z)-x
    v = (Matrix[1,0]*x + Matrix[1,1]*y + Matrix[1,2]*z)-y
    w = (Matrix[2,0]*x + Matrix[2,1]*y + Matrix[2,2]*z)-z

    ax.quiver(x, y, z, u, v, w, length=0.2, normalize=True)

    plt.show()

def PaintTranslation(Xoffset,Yoffset,Zoffset):
    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.4),
                        np.arange(-0.8, 1, 0.4),
                        np.arange(-0.8, 1, 0.4))


    u = Xoffset
    v = Yoffset
    w = Zoffset

    ax.quiver(x, y, z, u, v, w, length=0.2, normalize=True)

    plt.xticks([])
    plt.yticks([])
    plt.show()

def PaintShearing(CoefA,CoefB):
    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.4))

    # Make the direction data for the arrows
    shearingMatrix = np.eye(3)
    shearingMatrix[0,2] = CoefA
    shearingMatrix[1,2] = CoefB
    Matrix = shearingMatrix

    u = (Matrix[0,0]*x + Matrix[0,1]*y + Matrix[0,2]*z)-x
    v = (Matrix[1,0]*x + Matrix[1,1]*y + Matrix[1,2]*z)-y
    w = (Matrix[2,0]*x + Matrix[2,1]*y + Matrix[2,2]*z)-z

    ax.quiver(x, y, z, u, v, w, length=0.8)

    plt.xticks([])
    plt.yticks([])
    plt.show()

def PaintTapering(CoefA,CoefB):
    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.4),
                        np.arange(-0.8, 1, 0.4))

    # Make the direction data for the arrows
    u = ((0.5*np.square(CoefA)*z+CoefB*z+1)*x )-x
    v = ((0.5*np.square(CoefA)*z+CoefB*z+1)*y )-y
    w = 0

    ax.quiver(x, y, z, u, v, w, length=0.4)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def PaintTwisting(alpha):
    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.4))

    # Make the direction data for the arrows
    u = (np.cos(alpha*z)*x + np.sin(alpha*z)*y)-x
    v = (-np.sin(alpha*z)*x + np.cos(alpha*z)*y)-y
    w = 0

    ax.quiver(x, y, z, u, v, w, length=0.4)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def PaintSqueezing(Kbar):

    k = 1/(1+np.abs(Kbar))

    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.4),
                        np.arange(-0.8, 1, 0.4))

    # Make the direction data for the arrows
    boolMask = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=bool)
    TransformationMatrixs = np.eye(3)
    TransformationMatrixs[boolMask] = [k,1/np.sqrt(k),1/np.sqrt(k)]
    Matrix = TransformationMatrixs

    u = (Matrix[0,0]*x + Matrix[0,1]*y + Matrix[0,2]*z)-x
    v = (Matrix[1,0]*x + Matrix[1,1]*y + Matrix[1,2]*z)-y
    w = (Matrix[2,0]*x + Matrix[2,1]*y + Matrix[2,2]*z)-z
    ax.quiver(x, y, z, u, v, w, length=0.4)

    plt.xticks([])
    plt.yticks([])
    plt.show()

def PaintStretching(Kbar):

    k = (1+np.abs(Kbar))

    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.4))

    # Make the direction data for the arrows
    boolMask = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=bool)
    TransformationMatrixs = np.eye(3)
    TransformationMatrixs[boolMask] = [k,1/np.sqrt(k),1/np.sqrt(k)]
    Matrix = TransformationMatrixs

    u = (Matrix[0,0]*x + Matrix[0,1]*y + Matrix[0,2]*z)-x
    v = (Matrix[1,0]*x + Matrix[1,1]*y + Matrix[1,2]*z)-y
    w = (Matrix[2,0]*x + Matrix[2,1]*y + Matrix[2,2]*z)-z
    ax.quiver(x, y, z, u, v, w, length=0.2)

    plt.show()


if args.deformation == 'gaussianNoise':
    PaintGaussianNoise()
elif args.deformation == 'rotationX':
    PaintXRot(args.parameters[0])
elif args.deformation == 'rotationY':
    PaintYRot(args.parameters[0])
elif args.deformation == 'rotationZ':
    PaintZRot(args.parameters[0])
elif args.deformation == 'rotationXZ':
    PaintXZRot(args.parameters[0],args.parameters[1])
elif args.deformation == 'translation':
    PaintTranslation(args.parameters[0],args.parameters[1],args.parameters[2])
elif args.deformation == 'shearing':
    PaintShearing(args.parameters[0],args.parameters[1])
elif args.deformation == 'tapering':
    PaintTapering(args.parameters[0],args.parameters[1])
elif args.deformation == 'twisting':
    PaintTwisting(args.parameters[0])
elif args.deformation == 'squeezing':
    PaintSqueezing(args.parameters[0])
elif args.deformation == 'stretching':
    PaintStretching(args.parameters[0])
import os.path as osp
import torch_geometric as tg
#from torch_geometric import datasets
#from torch_geometric import loader
#from torch_geometric import transforms

__DATASET_ROOT = "./Data/"

def modelnet10(split: str = 'train',batch_size_for_loader: int = 15,sampledPoints: int = 4096):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
                    'Data/modelnet10fp')
    assert split in ['train', 'test'], "split must either be 'train' or 'test'"
    train = split == 'train'

    #smaple from the mesh to make it ap oint cloud
    pre_transforms = tg.transforms.Compose([
        tg.transforms.NormalizeScale(),
        tg.transforms.SamplePoints(sampledPoints)
    ])

    #declare the model data and corresponding data loader
    modeldata = tg.datasets.modelnet.ModelNet(root=path, name='10', train=train,pre_transform=pre_transforms, transform=None)
    return tg.data.DataLoader(modeldata,batch_size_for_loader,True,num_workers=0)


def modelnet40(split: str = 'train',batch_size_for_loader: int = 15,sampledPoints: int = 4096):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
                    'Data/modelnet40fp')
    assert split in ['train', 'test'], "split must either be 'train' or 'test'"
    train = split == 'train'

    #smaple from the mesh to make it ap oint cloud
    pre_transforms = tg.transforms.Compose([
        tg.transforms.NormalizeScale(),
        tg.transforms.SamplePoints(sampledPoints)
    ])

    #declare the model data and corresponding data loader
    modeldata = tg.datasets.modelnet.ModelNet(root=path, name='40', train=train,pre_transform=pre_transforms, transform=None)
    return tg.data.DataLoader(modeldata,batch_size_for_loader,True,num_workers=0)
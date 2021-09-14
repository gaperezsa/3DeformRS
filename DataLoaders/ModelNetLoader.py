import os.path
import torch_geometric as tg
#from torch_geometric import datasets
#from torch_geometric import loader
#from torch_geometric import transforms

__DATASET_ROOT = "./Data/"

def modelnet10(split: str = 'train',batch_size_for_loader: int = 10,sampledPoints: int = 4096):
    dataset_root = os.path.join(__DATASET_ROOT, "modelnet10fp")
    assert split in ['train', 'test'], "split must either be 'train' or 'test'"
    train = split == 'train'

    #smaple from the mesh to make it ap oint cloud
    pre_transforms = tg.transforms.Compose([
        tg.transforms.SamplePoints(sampledPoints)
    ])

    #declare the model data and corresponding data loader
    modeldata = tg.datasets.modelnet.ModelNet(root=dataset_root, name='10', train=train,pre_transform=pre_transforms, transform=None)
    return tg.data.DataLoader(modeldata,batch_size_for_loader,True)

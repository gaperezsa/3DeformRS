#taken from https://github.com/eth-sri/3dcertify/blob/master/data_processing/datasets.py
import os.path

from torch_geometric import datasets
from torchvision.transforms import transforms

from DataLoaders import transformers

#change where you have your data
__DATASET_ROOT = "/home/santamgp/Documents/CertifyingAffineTransformationsOnPointClouds/3D-RS-PointCloudCertifying/Pointnet/Data"


def modelnet40(num_points: int = 1024, split: str = 'train', rotate: str = 'z', add_noise: bool = None) -> datasets.modelnet.ModelNet:
    dataset_root = os.path.join(__DATASET_ROOT, "modelnet40fp")
    assert 1 <= num_points <= 1024, "num_points must be between 1 and 1024"
    assert split in ['train', 'test'], "split must either be 'train' or 'test'"
    assert rotate in ['none', 'z', 'so3'], "rotate must be one of 'none', 'z', 'so3'"
    train = split == 'train'

    pre_transforms = transforms.Compose([
        transformers.SamplePoints(num=4096),
        transformers.FarthestPoints(num_points=1024)
    ])

    if add_noise is None:
        add_noise = split == 'train'

    if rotate == 'none':
        random_rotate = transformers.Identity()
    elif rotate == 'z':
        random_rotate = transformers.RandomRotateZ()
    elif rotate == 'so3':
        random_rotate = transformers.RandomRotateSO3()
    else:
        raise Exception(f"Invalid rotation {rotate}")

    if train:
        post_transforms = transforms.Compose([
            transformers.ConvertFromGeometric(),
            transformers.NormalizeUnitSphere(),
            transformers.SelectPoints(num_points),
            random_rotate,
            transformers.RemoveNones()
        ])
    else:
        post_transforms = transforms.Compose([
            transformers.ConvertFromGeometric(),
            transformers.NormalizeUnitSphere(),
            transformers.SelectPoints(num_points),
            random_rotate,
            transformers.RemoveNones()
        ])
    if add_noise:
        post_transforms.transforms.append(transformers.GaussianNoise())

    return datasets.modelnet.ModelNet(root=dataset_root, name='40', train=train,
                                      pre_transform=pre_transforms, transform=post_transforms)


#taken from https://github.com/ajhamdi/MVTN#usage-3d-classification--retrieval

from typing import Dict
import numpy as np
import glob
import h5py
import pandas as pd
from torch.utils.data.dataset import Dataset
import os
import torch
import collections

def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:

            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'pytorch3d.structures.meshes':
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, (int)):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):

        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

def torch_center_and_normalize(points,p="inf"):
    """
    a helper pytorch function that normalize and center 3D points clouds 
    """
    N = points.shape[0]
    center = points.mean(0)
    if p != "fro" and p!= "no":
        scale = torch.max(torch.norm(points - center, p=float(p),dim=1))
    elif p=="fro" :
        scale = torch.norm(points - center, p=p )
    elif p=="no":
        scale = 1.0
    points = points - center.expand(N, 3)
    points = points * (1.0 / float(scale))
    return points


class ScanObjectNN(torch.utils.data.Dataset):
    """
    This class loads ScanObjectNN from a given directory into a Dataset object.
    ScanObjjectNN is a point cloud dataset of realistic shapes of from the ScanNet dataset and can be downloaded from
    https://github.com/hkust-vgd/scanobjectnn .
    """

    def __init__(
        self,
        data_dir,
        split,
        nb_points,
        normals: bool = False,
        suncg: bool = False,
        variant: str = "obj_only",
        dset_norm: str = "inf",

    ):
        """
        Store each object's synset id and models id from data_dir.
        Args:
            data_dir: Path to ShapeNetCore data.
            synsets: List of synset categories to load from ShapeNetCore in the form of
                synset offsets or labels. A combination of both is also accepted.
                When no category is specified, all categories in data_dir are loaded.
            version: (int) version of ShapeNetCore data in data_dir, 1 or 2.
                Default is set to be 1. Version 1 has 57 categories and verions 2 has 55
                categories.
                Note: version 1 has two categories 02858304(boat) and 02992529(cellphone)
                that are hyponyms of categories 04530566(watercraft) and 04401088(telephone)
                respectively. You can combine the categories manually if needed.
                Version 2 doesn't have 02858304(boat) or 02834778(bicycle) compared to
                version 1.
            load_textures: Boolean indicating whether textures should loaded for the model.
                Textures will be of type TexturesAtlas i.e. a texture map per face.
            texture_resolution: Int specifying the resolution of the texture map per face
                created using the textures in the obj file. A
                (texture_resolution, texture_resolution, 3) map is created per face.
        """
        super().__init__()
        self.data_dir = data_dir
        self.nb_points = nb_points
        self.normals = normals
        self.suncg = suncg
        self.variant = variant
        self.dset_norm = dset_norm
        self.split = split
        self.classes = {0: 'bag', 10: 'bed', 1: 'bin', 2: 'box', 3: 'cabinet', 4: 'chair', 5: 'desk', 6: 'display',
                        7: 'door', 11: 'pillow', 8: 'shelf', 12: 'sink', 13: 'sofa', 9: 'table', 14: 'toilet'}

        self.labels_dict = {"train": {}, "test": {}}
        self.objects_paths = {"train": [], "test": []}

        if self.variant != "hardest":
            pcdataset = pd.read_csv(os.path.join(
                data_dir, "split_new.txt"), sep="\t", names=['obj_id', 'label', "split"])
            for ii in range(len(pcdataset)):
                if pcdataset["split"][ii] != "t":
                    self.labels_dict["train"][pcdataset["obj_id"]
                                              [ii]] = pcdataset["label"][ii]
                else:
                    self.labels_dict["test"][pcdataset["obj_id"]
                                             [ii]] = pcdataset["label"][ii]

            all_obj_ids = glob.glob(os.path.join(self.data_dir, "*/*.bin"))
            filtered_ids = list(filter(lambda x: "part" not in os.path.split(
                x)[-1] and "indices" not in os.path.split(x)[-1], all_obj_ids))

            self.objects_paths["train"] = sorted(
                [x for x in filtered_ids if os.path.split(x)[-1] in self.labels_dict["train"].keys()])
            self.objects_paths["test"] = sorted(
                [x for x in filtered_ids if os.path.split(x)[-1] in self.labels_dict["test"].keys()])
        else:
            filename = os.path.join(
                data_dir, "{}_objectdataset_augmentedrot_scale75.h5".format(self.split))
            with h5py.File(filename, "r") as f:
                self.labels_dict[self.split] = np.array(f["label"])
                self.objects_paths[self.split] = np.array(f["data"])

    def __getitem__(self, idx: int) -> Dict:
        """
        Read a model by the given index. no mesh is availble in this dataset so retrun None and correction factor of 1.0
        """
        if self.variant != "hardest":
            obj_path = self.objects_paths[self.split][idx]

            points = self.load_pc_file(obj_path)

            points = points[np.random.randint(
                points.shape[0], size=self.nb_points), :]

            label = self.labels_dict[self.split][os.path.split(obj_path)[-1]]
        else:

            points = self.objects_paths[self.split][idx]
            label = self.labels_dict[self.split][idx]

        points = torch.from_numpy(points).to(torch.float)
        points = torch_center_and_normalize(points, p=self.dset_norm)
        return points, None, label

    def __len__(self):
        return len(self.objects_paths[self.split])

    def load_pc_file(self, filename):

        pc = np.fromfile(filename, dtype=np.float32)

        if(self.suncg):
            pc = pc[1:].reshape((-1, 3))
        else:
            pc = pc[1:].reshape((-1, 11))

        if self.variant == "with_bg":
            pc = np.array(pc[:, 0:3])
            return pc

        else:

            filtered_idx = np.intersect1d(np.intersect1d(np.where(
                pc[:, -1] != 0)[0], np.where(pc[:, -1] != 1)[0]), np.where(pc[:, -1] != 2)[0])
            (values, counts) = np.unique(
                pc[filtered_idx, -1], return_counts=True)
            max_ind = np.argmax(counts)
            idx = np.where(pc[:, -1] == values[max_ind])[0]
            pc = np.array(pc[idx, 0:3])
            return pc


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM

Modified by 
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@Time: 2021/1/21 3:10 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# change this to your data root
DATA_DIR = '/home/santamgp/Documents/CertifyingAffineTransformationsOnPointClouds/3D-RS-PointCloudCertifying/CurveNet/data'

def download_modelnet40():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        os.mkdir(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048'))
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def download_shapenetpart():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        os.mkdir(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data'))
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))


def load_data_normal(partition):
    f = h5py.File(os.path.join(DATA_DIR, 'modelnet40_normal', 'normal_%s.h5'%partition), 'r+')
    data = f['xyz'][:].astype('float32')
    label = f['normal'][:].astype('float32')
    f.close()
    return data, label


def load_data_cls(partition):
    download_modelnet40()
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40*hdf5_2048', '*%s*.h5'%partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_data_partseg(partition):
    download_shapenetpart()
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*%s*.h5'%partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            #pointcloud = rotate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class ModelNetNormal(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_normal(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item][:self.num_points]
        if self.partition == 'train':
            #pointcloud = translate_pointcloud(pointcloud)
            idx = np.arange(0, pointcloud.shape[0], dtype=np.int64)
            np.random.shuffle(idx)
            pointcloud = self.data[item][idx]
            label = self.label[item][idx]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class ShapeNetPart(Dataset):
    def __init__(self, num_points=2048, partition='train', class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]



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
        self.data_dir = DATA_DIR if data_dir=="" else data_dir
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
        return points, label

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


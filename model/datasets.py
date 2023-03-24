import glob
import json
import math
import os
import random
import itertools
itertools.combinations
import numpy as np
import numpy.linalg as LA
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from config import config

class WireframeDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        filelist = glob.glob(f"{rootdir}/{split}/*_label.npz")
        filelist.sort()

        print(f"n{split}:", len(filelist))
        self.split = split
        self.filelist = filelist
        with open(f'{split}.txt', 'w') as f:
            for name in self.filelist:
                f.write(name + '\n')

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        iname = self.filelist[idx][:-10].replace("_a0", "").replace("_a1", "") + ".png"
        image = io.imread(iname).astype(float)[:, :, :3]
        if "a1" in self.filelist[idx]:
            image = image[:, ::-1, :]
        image = (image - config["image"]["mean"]) / config["image"]["stddev"]
        image = np.rollaxis(image, 2).copy()

        # npz["jmap"]: [J, H, W]    Junction heat map # 顶点热图
        # npz["joff"]: [J, 2, H, W] Junction offset within each pixel
        # npz["lmap"]: [H, W]       Line heat map with anti-aliasing 线段热图、
        # npz["junc"]: [Na, 3]      Junction coordinates # 顶点坐标
        # npz["Lpos"]: [M, 2]       Positive lines represented with junction indices # 线段1顶点索引
        # npz["Lneg"]: [M, 2]       Negative lines represented with junction indices # 线段2顶点索引
        # npz["lpos"]: [Np, 2, 3]   Positive lines represented with junction coordinates # 线段1
        # npz["lneg"]: [Nn, 2, 3]   Negative lines represented with junction coordinates # 线段2
        #
        # For junc, lpos, and lneg that stores the junction coordinates, the last
        # dimension is (y, x, t), where t represents the type of that junction.
        with np.load(self.filelist[idx]) as npz:
            target = {
                name: torch.from_numpy(npz[name]).float()
                for name in ["jmap", "joff", "lmap"]
            }
            lpos = np.random.permutation(npz["lpos"])[: config["n_stc_posl"]]
            lneg = np.random.permutation(npz["lneg"])[: config["n_stc_negl"]]
            npos, nneg = len(lpos), len(lneg)
            lpre = np.concatenate([lpos, lneg], 0)
            for i in range(len(lpre)):
                if random.random() > 0.5:
                    lpre[i] = lpre[i, ::-1]
            ldir = lpre[:, 0, :2] - lpre[:, 1, :2]
            ldir /= np.clip(LA.norm(ldir, axis=1, keepdims=True), 1e-6, None)
            feat = [
                lpre[:, :, :2].reshape(-1, 4) / 128 * config["use_cood"],
                ldir * config["use_slop"],
                lpre[:, :, 2],
            ]
            feat = np.concatenate(feat, 1)
            meta = {
                "junc": torch.from_numpy(npz["junc"][:, :2]),
                "jtyp": torch.from_numpy(npz["junc"][:, 2]).byte(), # 顶点label
                "Lpos": self.adjacency_matrix(len(npz["junc"]), npz["Lpos"]),
                "Lneg": self.adjacency_matrix(len(npz["junc"]), npz["Lneg"]),
                "lpre": torch.from_numpy(lpre[:, :, :2]), # lpos和lneg共同组成
                "lpre_label": torch.cat([torch.ones(npos), torch.zeros(nneg)]),
                "lpre_feat": torch.from_numpy(feat),
            }

        return torch.from_numpy(image).float(), meta, target

    def adjacency_matrix(self, n, link):
        mat = torch.zeros(n + 1, n + 1, dtype=torch.uint8)
        link = torch.from_numpy(link)
        if len(link) > 0:
            mat[link[:, 0], link[:, 1]] = 1
            mat[link[:, 1], link[:, 0]] = 1
        return mat


def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        [b[1] for b in batch],
        default_collate([b[2] for b in batch]),
    )

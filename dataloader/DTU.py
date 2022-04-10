import pdb

import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from tqdm import tqdm
import pickle
import cv2
import skimage
from skimage.transform import rescale
from . import base
from utils import camera
from utils.utils import log,debug


class Dataset(base.Dataset):
    def __init__(self,opt,split="train",subset=None):
        if split=="train":
            self.dsam_H,self.dsam_W=opt.data.train_img_size
        else:
            self.dsam_H, self.dsam_W = opt.data.val_img_size
        self.raw_H,self.raw_W = 1200,1600
        downscale=self.raw_H/self.dsam_H

        super().__init__(opt,split)
        self.root = opt.data.root or "datasets/DTU"
        self.path = "{}/{}".format(self.root,opt.data.scene)
        self.path_image = "{0}/{1}/image".format(self.path,opt.data.scene)
        self.path_mask  = "{0}/{1}/mask".format(self.path,opt.data.scene)
        self.path_cam = '{0}/{1}/cameras.npz'.format(self.path,opt.data.scene)
        image_fnames = sorted(os.listdir(self.path_image))
        image_fnames=[os.path.join(self.path_image,img_f_i) for img_f_i in  image_fnames]
        mask_fnames = sorted(os.listdir(self.path_mask))
        mask_fnames = [os.path.join(self.path_mask, mask_f_i) for mask_f_i in mask_fnames]
        self.n_images = len(image_fnames)
        # loading the pose
        camera_dict = np.load(self.path_cam)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.c2w_all = []
        cam_center_norms = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = self.load_K_Rt_from_P(P)
            cam_center_norms.append(np.linalg.norm(pose[:3, 3]))
            # downscale intrinsics
            intrinsics[0, 2] /= downscale
            intrinsics[1, 2] /= downscale
            intrinsics[0, 0] /= downscale
            intrinsics[1, 1] /= downscale
            # intrinsics[0, 1] /= downscale # skew is a ratio, do not scale

            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.c2w_all.append(torch.from_numpy(pose).float())
        max_cam_norm = max(cam_center_norms)
        if opt.VolSDF.obj_bounding_radius > 0:     # normalized the poses into obj_radius
            for i in range(len(self.c2w_all)):
                self.c2w_all[i][:3, 3] *= (opt.VolSDF.obj_bounding_radius / max_cam_norm / 1.1)

        self.rgb_images = []
        self.list = image_fnames
        num_val_split = int(len(self) * opt.data.val_ratio)
        image_fnames = image_fnames[:-num_val_split] if split == "train" else image_fnames[-num_val_split:]
        mask_fnames = mask_fnames[:-num_val_split] if split == "train" else mask_fnames[-num_val_split:]
        self.n_images =len(image_fnames)
        self.c2w_all=self.c2w_all[:-num_val_split] if split == "train" else self.c2w_all[-num_val_split:]
        self.intrinsics_all =self.intrinsics_all[:-num_val_split] if split == "train" else self.intrinsics_all[-num_val_split:]
        self.list = image_fnames
        for path in tqdm(image_fnames, desc='loading images...'):
            rgb = self.load_rgb(path, downscale)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.object_masks = []
        for path in mask_fnames:
            object_mask = self.load_mask(path, downscale)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).to(dtype=torch.bool))
    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])
    def load_rgb(self,path, downscale=1):
        img = imageio.imread(path)
        img = skimage.img_as_float32(img)
        if downscale != 1:
            img = rescale(img, 1. / downscale, anti_aliasing=False, multichannel=True)

        # NOTE: pixel values between [-1,1]
        # img -= 0.5
        # img *= 2.
        img = img.transpose(2, 0, 1)
        return img

    def load_mask(self,path, downscale=1):
        alpha = imageio.imread(path, as_gray=True)
        alpha = skimage.img_as_float32(alpha)
        if downscale != 1:
            alpha = rescale(alpha, 1. / downscale, anti_aliasing=False, multichannel=False)
        object_mask = alpha > 127.5

        return object_mask

    def load_K_Rt_from_P(self,P):
        """
        modified from IDR https://github.com/lioryariv/idr
        """
        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]

        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        return intrinsics, pose

    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=idx)
        sample.update(
            image=self.rgb_images[idx],
            intr=self.intrinsics_all[idx][:3,:3],
            pose=self.c2w_all[idx],
            mask=self.object_masks[idx]
        )
        return sample




import numpy as np
from torch.utils.data import Dataset
import json
import cv2
import torch
import os.path as osp
import glob


class SegmentationDataset(Dataset):
    """Segmentation dataset loader."""

    def __init__(self, json_folder, img_folder, is_train, class_to_id, resolution=None, augmentation=False, transform=None):
        """
        Args:
            json_folder (str): Path to folder that contains the annotations.
            img_folder (str): Path to all images.
            is_train (bool): Is this a training dataset ?
            augmentation (bool): Do dataset augmentation (crete artificial variance) ?
        """

        self.gt_file_list = glob.glob(osp.join(json_folder, '*.json'))

        self.total_samples = len(self.gt_file_list)
        self.img_folder = img_folder
        self.is_train = is_train
        self.transform = transform
        self.augmentation = augmentation
        self.resolution = resolution
        self.class_to_id = class_to_id

        # Mean and std are needed because we start from a pre trained net
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):

        gt_file = self.gt_file_list[idx]
        # print(gt_file)
        img_name = gt_file[:-5]+'.tif'
        # Abre Json
        gt_json = json.load(open(gt_file, 'r'))
        # Abre imagem
        # print(img_name)
        try:
            img_np = cv2.imread(img_name,
                                cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
            img_np.shape
        except:
            img_name = gt_file[:-5]+'.jpg'
            img_np = cv2.imread(img_name,
                                cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
        if not isinstance(img_np, np.ndarray):
            print('Unable to open file %s' %( img_name))
            raise ('Unable to open file %s' %( img_name))
        original_shape = img_np.shape
        if self.resolution is not None:
            img_np = cv2.resize(
                img_np, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_CUBIC)[..., ::-1]
        img_np = np.ascontiguousarray(img_np)
        # Cria imagem zerada
        label_np = np.zeros((img_np.shape[0], img_np.shape[1]))
        for l in gt_json['shapes']:
            cv2.fillPoly( label_np, np.int_([l['points']]), (self.class_to_id[l['label']],))


        # Transforma o GT em inteiro
        label_np = label_np.astype(np.int32)

        if self.is_train and self.augmentation:
            if np.random.rand() > 0.5:
                img_np = np.fliplr(img_np)
                label_np = np.fliplr(label_np)
                img_np = np.ascontiguousarray(img_np)
                label_np = np.ascontiguousarray(label_np)
            # if np.random.rand() > 0.2:
            #     img_np = np.roll(img_np, np.random.randint(
            #         0, 3, size=1)[0], axis=2)
            #     img_np = np.ascontiguousarray(img_np)
            # elif np.random.rand() > 0.2:
            #     gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            #     img_np[..., 0] = gray
            #     img_np[..., 1] = gray
            #     img_np[..., 2] = gray

            # if np.random.rand() < 1:
            #     img_np, label_np = self.distort_transform(
            #         img_np, label_np)

            if np.random.rand() < 1:
                img_np, label_np = self.perspective_transforme(
                    img_np, label_np)

            if np.random.rand() < 0.3:
                img_np, label_np = self.blur_transform(
                    img_np, label_np)

            if np.random.rand() < 0.3:
                img_np, label_np = self.noise_transform(
                    img_np, label_np)

        img_np = np.ascontiguousarray(img_np, dtype=np.float32)
        label_np = np.ascontiguousarray(label_np, dtype=np.float32)

        img_pt = img_np.astype(np.float32) / np.max(img_np)
        for i in range(3):
            img_pt[..., i] -= self.mean[i]
            img_pt[..., i] /= self.std[i]

        img_pt = img_pt.transpose(2, 0, 1)

        img_pt = torch.from_numpy(img_pt).float()
        label_pt = torch.from_numpy(label_np).long()

        sample = {'image': img_pt, 'gt': label_pt, 'image_original': img_np}

        if self.transform:
            sample = self.transform(sample)

        return sample
    def distort_transform(self, img, gt):
        h, w = img.shape[:2]

        # copy parameters to arrays
        K = np.array([[w, 0., w/2],
                    [0, h, h/2],
                    [0, 0, 1]]) 
    
        d = np.array([np.random.random_sample((1,))*0.15
                    , np.random.random_sample((1,))*0.15
                    , (np.random.random_sample((1,))-.5 )*0.005
                    , (np.random.random_sample((1,))-.5 )*0.005,
                    (np.random.random_sample((1,))-.5 )*0.001]) # just use first two terms 
        d = d.reshape(5)

        # undistort
        newcamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0)
        newimg = cv2.undistort(img.astype(np.float32), K, d, None, newcamera)
        newgt = cv2.undistort(gt.astype(np.uint8), K, d, None, newcamera)
        return newimg.astype(np.float32), newgt.astype(np.int32)
    
    
    def perspective_transforme(self, img, gt):
        sz = img.shape[:2][::-1]
        src = np.array([
            [0, 0],
            [sz[0]-1, 0],
            [sz[0]-1, sz[1]-1],
            [0, sz[1]-1]], dtype="float32")
        warp_force = 0.1
        #src += np.array([sz[0]/2,sz[1]/2 ], dtype = "float32")
        rd = (np.random.random_sample(src.shape).astype(np.float32)*2 - 1) * \
            np.array([sz[0]*warp_force, sz[1]*warp_force], dtype="float32")
        dst = src + rd
        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(src, dst)
        return (cv2.warpPerspective(img.astype(np.float32), M, sz, borderMode=cv2.BORDER_CONSTANT, borderValue=0)), cv2.warpPerspective(gt.astype(np.uint8), M, sz, borderMode=cv2.BORDER_CONSTANT,flags= cv2.INTER_NEAREST, borderValue=self.class_to_id['Background']).astype(np.int32)
    def blur_transform(self, img, gt):
        k = np.random.randint(3, 6)
        img = cv2.blur(img, (k, k))
        return img, gt

    def noise_transform(self, img, gt):
        prob = np.random.random_sample(img.shape)
        prob[prob >= 0.5] = (np.random.random_sample((1,))[0]*2-1)*50
        prob[prob < 0.5] = 0
        data32 = img + prob
        np.clip(data32, 0, 255, out=data32)
        img = data32.astype(np.float32)
        return img, gt


def mixup_batch(inp_batch, alpha):
    """
    Applies mixup augementation to a batch
    :param input_batch: tensor with batchsize as first dim
    :param alpha: lamda drawn from beta(alpha+1, alpha)
    """
    inp_clone = inp_batch.clone()
    # getting batch size
    batchsize = inp_batch.size()[0]

    # permute a clone
    perm = np.random.permutation(batchsize)
    for i in range(batchsize):
        inp_clone[i] = inp_batch[perm[i]]
    # generating different lambda for each sample
    # Refernced from http://www.inference.vc/mixup-data-dependent-data-augmentation/
    lam = torch.Tensor(np.random.beta(alpha+1, alpha, batchsize))
    lam = lam.view(-1, 1, 1, 1)
    inp_mixup = lam * inp_batch + (1 - lam) * inp_clone
    return inp_mixup

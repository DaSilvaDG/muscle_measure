import concurrent.futures
import functools
import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from numba import njit
from scipy.optimize import differential_evolution
import sys
import tkinter as tk
from tkinter import filedialog as fd 
import vgg_unet
import xdialog
from statistics import mode
import crop


_DEBUG = 0

transforms = []


def find_instr(func, keyword, sig=0, limit=5):
    count = 0
    for l in func.inspect_asm(func.signatures[sig]).split("\n"):
        if keyword in l:
            count += 1
            print(l)
            if count >= limit:
                break
    if count == 0:
        print("No instructions found")


@njit(fastmath=True, error_model="numpy")
def sad(dst1, dst2):
    d1 = dst1.ravel()
    d2 = dst2.ravel()
    sz = d1.size
    sum_ = 0.0
    for i in range(sz):
        sum_ = np.float32(sum_) + abs(d1[i] - d2[i])
    return sum_


def call_back_ext(x, convergence, t_index):
    global transforms
    print(t_index, convergence, x)
    transforms[t_index] = transforms[t_index]*0.7  + x*0.3

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Registrator:
    def __init__(self):
        self.work_resolution = (512, 384)
        self.resolution = (451, 288)

        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        self.model_name = resource_path("segm.pth")

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.nbClasses = 6

        self.viridis = cm.get_cmap("viridis", self.nbClasses)

        self.images = None
        self.output_path = None

    def seg_images(self):
        net = vgg_unet.UNetVgg(self.nbClasses).to(self.device)

        net.load_state_dict(torch.load(self.model_name, map_location=lambda storage, loc: storage))

        net.eval()

        self.segmentations = []
        for x in self.images:
            img_np = np.ascontiguousarray(x)

            img_pt = np.copy(img_np).astype(np.float32) / 255.0
            for i in range(3):
                img_pt[..., i] -= self.mean[i]
                img_pt[..., i] /= self.std[i]

            img_pt = img_pt.transpose(2, 0, 1)

            img_pt = torch.from_numpy(img_pt[None, ...]).float().to(self.device)

            label_out = net(img_pt)

            label_out = label_out.cpu().detach().numpy()

            label_out = np.squeeze(label_out)

            labels = np.argmax(label_out, axis=0).astype(np.uint8)
            # from matplotlib import pyplot as plt
            # plt.imshow(labels)
            # plt.show()
            self.segmentations.append(labels)

        for k, (seg, img) in enumerate(zip(self.segmentations, self.images)):
            t = img.copy()
            for i in range(self.nbClasses):
                contours, _ = cv2.findContours(
                    (seg == i).astype("u1"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
                )[-2:]
                contour = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                if contour:
                    contour = contour[0]
                    if i != 0 and i != 1:
                        epsilon = 0.01 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        cv2.drawContours(t, [approx], 0, self.viridis.colors[i, :3] * 255, 1)
            print(self.output_path[:-9] + f"seg_{k}.png")
            cv2.imwrite(self.output_path[:-9] + f"seg_{k}.png", t)

    def setInputImages(self, dirname, images_list):
        self.images = []
        self.output_path = dirname + "/mount.png"
        try:
            for i in sorted(images_list):
                with tifffile.TiffFile(i) as tif:
                    data = tif.asarray()
                    # metadata = tif.image_description
                    # print(metadata)
                self.images.append(data)
        except:
            for i in sorted(images_list):
                img = cv2.imread(i)
                self.images.append(img)

    def getTransform(self, i1, i2):
        shape = self.images[i1].shape
        res1 = self.segmentations[i1]

        kernel = np.ones((5, 5), np.uint8)

        imgs1 = []
        for i in range(2, self.nbClasses):
            im1 = self.images[i1].copy()

            im1[cv2.dilate((res1 != i).astype("u1"), kernel, iterations=1).astype(bool)] = 0
            imgs1.append(im1)

        res2 = self.segmentations[i2]
        im2 = self.images[i2].copy()
        imgs2 = []
        for i in range(2, self.nbClasses):
            im2 = self.images[i2].copy()
            im2[cv2.dilate((res2 != i).astype("u1"), kernel, iterations=1).astype(bool)] = 0
            imgs2.append(im2)

        M = np.float32([[1, 0, 0], [0, 1, shape[0] // 2]])
        show_img1 = cv2.warpAffine(self.images[i1].copy(), M, (shape[1] * 2, shape[0] * 2))
        dst1 = []
        for i in range(2, self.nbClasses):
            dst1.append(cv2.warpAffine(imgs1[i - 2], M, (shape[1] * 2, shape[0] * 2)))

        if _DEBUG and 0:
            plt.imshow(np.hstack((im1, im2)))
            plt.show()

        def call_back(x, convergence=0.0):
            print(list(x), convergence)
            m2 = M.copy()
            m2[0, 2] += x[0]
            m2[1, 2] += x[1]
            dst2 = cv2.warpAffine(self.images[i2].copy(), m2, (shape[1] * 2, shape[0] * 2))

            R = cv2.getRotationMatrix2D((M[0, 2], M[1, 2]), (x[2]), 1)
            dst2 = cv2.warpAffine(dst2, R, (shape[1] * 2, shape[0] * 2))

            cv2.imshow("get_transform", np.max(np.array([show_img1, dst2]), axis=0))
            cv2.waitKey(20)

        def func(x):
            ori = np.array([0, shape[0] // 2])
            # if np.linalg.norm(x[:2] - ori) != shape[1]//2:
            #     return np.inf
            m2 = M.copy()
            m2[0, 2] += x[0]
            m2[1, 2] += x[1]

            ret = 0.0
            for i in range(2, self.nbClasses):
                dst2 = cv2.warpAffine(imgs2[i - 2], m2, (shape[1] * 2, shape[0] * 2))
                R = cv2.getRotationMatrix2D((M[0, 2], M[1, 2]), (x[2]), 1)
                dst2 = cv2.warpAffine(dst2, R, (shape[1] * 2, shape[0] * 2))
                ret += sad(np.float32(dst1[i - 2]), np.float32(dst2))

            return ret

        bounds = [
            (shape[1] // 2 - 1, shape[1] // 2 + 1),
            (-shape[0] // 4, shape[0] // 4),
            (-30, 30),
        ]

        result = differential_evolution(
            func,
            bounds,
            popsize=15,
            maxiter=5000,
            updating="immediate",
            mutation=(0.5, 1.9),
            recombination=0.5,
            tol=0.001,
            callback=lambda x, convergence: call_back_ext(x, convergence, t_index=i2),
        )

        return result.x, result.fun

    def __call__(self, t_index, x, convergence=0.0):
        print(t_index, convergence)
        self.transforms[t_index] = x

    def run(self):
        global transforms
        shape = self.images[0].shape
        transforms = [np.array([0, shape[0] // 2, 0] if i == 0 else [shape[1] // 2, shape[0] // 2, 0], dtype='f4') for i in range(len(self.images))]

        self.seg_images()

        for k, im in enumerate(self.images):
            gray = np.ascontiguousarray(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            self.images[k] = cv2.normalize(gray, None, 0.0, 255, cv2.NORM_MINMAX)
        print("Start Recorder")
        cv2.waitKey(200)
        """Multiprocessing Block"""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.images) - 1)

        futures = [
            executor.submit(self.getTransform, i, i + 1) for i in range(len(self.images) - 1)
        ]

        # for f in futures:
        #     t, n = f.result()
        #     transfoms.append(t)
        #     print(n)

        sz = 4
        M = np.float32([[1, 0, 0], [0, 1, shape[0] // 2]])


        while True:
            blended_images = []
            rimages = self.images[::-1]
            for i in range(0, len(self.images) - 1):
                dst2 = rimages[i].copy()
                rtransfoms = transforms[::-1]
                for k in range(i, len(transforms) - 1):
                    T = rtransfoms[k]
                    M2 = M.copy()
                    if k == i:
                        M2[0, 2] += T[0]
                        M2[1, 2] += T[1]
                    else:
                        M2[0, 2] = T[0]
                        M2[1, 2] = T[1]
                    dst2 = cv2.warpAffine(dst2, M2, (shape[1] * sz, shape[0] * sz))
                    # cv2.imshow("MountView", np.max(np.array([*blended_images, dst2.astype('uint8')]), axis=0))
                    # cv2.waitKey(500)
                    R = cv2.getRotationMatrix2D((M[0, 2], M[1, 2]), (T[2]), 1)
                    dst2 = cv2.warpAffine(dst2, R, (shape[1] * sz, shape[0] * sz))
                    # cv2.imshow(
                    #     "MountView", np.max(np.array([*blended_images, dst2.astype("uint8")]), axis=0)
                    # )
                    # cv2.waitKey(700)

                blended_images.append(dst2)
                final_image = np.max(np.array(blended_images), axis=0)
                # cv2.imshow("MountView", final_image.astype("uint8"))
                # cv2.waitKey(2000)

            dst2 = rimages[-1].copy()
            M2 = M.copy()
            dst2 = cv2.warpAffine(dst2, M2, (shape[1] * sz, shape[0] * sz))
            blended_images.append(dst2)

            final_image = np.max(np.array(blended_images), axis=0)

            final_image = cv2.normalize(final_image, None, 0, 255, cv2.NORM_MINMAX)

            final_image = final_image.astype("uint8")
            final_image = self.crop_image(final_image)

            print(transforms)
            cv2.imshow("MountView", final_image)
            cv2.waitKey(100)
            if all([fut.done() for fut in futures]):
                _ = [fut.result()[0] for fut in futures]
                break

        final_image = self.crop_image(final_image)
        cv2.imshow("MountView", final_image)
        cv2.waitKey(1000)
        cv2.imwrite(self.output_path, final_image)
        print(self.output_path)

    def crop_image(self, img, tol=0):
        # img is image data
        # tol  is tolerance
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]


def main(mout_dir, exts):
    cv2.namedWindow("MountView", cv2.WINDOW_NORMAL)

    # cv2.namedWindow('debug', cv2.WINDOW_NORMAL)
    # list_files = glob.glob('./MONTAGEM/**/croped_*.png', recursive=True)
    crop.crop_images(f"{mout_dir}/**/*.{exts}")

    list_files = glob.glob(
        f"{mout_dir}/**/*_cropped.png",
        recursive=True,
    )
    # list_files = glob.glob('/home/diegogomes/dev/deivid/Baseline/João - Baseline/VLD/*.jpg', recursive=True)

    list_files = list(sorted(list_files))
    dirs_names = set()

    [dirs_names.add(os.path.dirname(d)) for d in list_files]

    image_name_per_dir = {k: [] for k in dirs_names}

    [image_name_per_dir[os.path.dirname(d)].append(d) for d in list_files]

    image_name_per_dir = {k: list(sorted(v)) for k, v in image_name_per_dir.items()}
    for k, v in image_name_per_dir.items():
        print(k)
        for vv in v:
            print("\t" + vv)

    model = Registrator()
    for k, v in image_name_per_dir.items():
        model.setInputImages(k, v)
        model.run()


if __name__ == "__main__":
    dir_name = xdialog.directory("Images to mount")
    exts = mode([f.split(".")[-1] for f in glob.glob(f"{dir_name}/**/*", recursive=True)if "cropped" not in f and "seg" not in f and "mount" not in f])
    
    main(os.path.normpath(dir_name), exts)

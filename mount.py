import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.optimize import differential_evolution
import tifffile
import concurrent.futures
from numba import njit
import glob
import os 
import vgg_unet
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


_DEBUG = 1
def find_instr(func, keyword, sig=0, limit=5):
    count = 0
    for l in func.inspect_asm(func.signatures[sig]).split('\n'):
        if keyword in l:
            count += 1
            print(l)
            if count >= limit:
                break
    if count == 0:
        print('No instructions found')


@njit(fastmath=True, error_model='numpy')
def sad(dst1, dst2):
    d1 = dst1.ravel()
    d2 = dst2.ravel()
    sz = d1.size
    sum_ = 0.
    for i in range(sz):
        sum_ = np.float32(sum_) + abs(d1[i] - d2[i]) 
    return sum_ 

class Registrator:
    def __init__(self):
        self.work_resolution = (512, 384)
        self.resolution = (451, 288)

        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


        self.model_name = 'segm.pth'
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.nbClasses = 6 
        
        self.viridis = cm.get_cmap('viridis', self.nbClasses)

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

            self.segmentations.append(labels)
        
        for k, (seg, img) in enumerate(zip(self.segmentations, self.images)):
            t = img.copy()
            for i in range(self.nbClasses):
                contours, _ = cv2.findContours((seg == i).astype('u1'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[-2:]
                contour = sorted(contours, key=lambda x: cv2.contourArea(x), reverse = True)
                if contour:
                    contour = contour[0]
                    if i != 0 and i != 1:
                        epsilon = 0.01*cv2.arcLength(contour,True)
                        approx = cv2.approxPolyDP(contour,epsilon,True) 
                        cv2.drawContours(t, [approx], 0, self.viridis.colors[i,:3] * 255, 1)
            print(self.output_path[:-9] + f"seg_{k}.png")
            cv2.imwrite(self.output_path[:-9] + f"seg_{k}.png", t)
        
    def setInputImages(self, dirname, images_list):
        self.images = []
        self.output_path = dirname + '/mount.png'
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
        
        kernel = np.ones((5,5), np.uint8) 

        imgs1 = []
        for i in range(2, self.nbClasses): 
            im1 = self.images[i1].copy()
            
            
            im1[cv2.dilate((res1 != i).astype('u1'), kernel, iterations=1).astype(np.bool) ] = 0
            imgs1.append(im1)

        
        res2 = self.segmentations[i2]
        im2 = self.images[i2].copy()
        imgs2 = []
        for i in range(2, self.nbClasses): 
            im2 = self.images[i2].copy()
            im2[cv2.dilate((res2 != i).astype('u1'), kernel, iterations=1).astype(np.bool) ] = 0
            imgs2.append(im2)
        
        
        M = np.float32([[1,0,0],[0,1,shape[0]//2]])
        show_img1 = cv2.warpAffine(self.images[i1].copy(),M,(shape[1] * 2,shape[0] * 2 ))
        dst1 = []
        for i in range(2, self.nbClasses):
            dst1.append(cv2.warpAffine(imgs1[i - 2],M,(shape[1] * 2,shape[0] * 2 )))

        if _DEBUG and 0:
            plt.imshow(np.hstack((im1, im2)))
            plt.show()
        def call_back(x, convergence=0. ):
            print(list(x), convergence)
            m2 = M.copy()
            m2[0, 2] += x[0]
            m2[1, 2] += x[1]
            dst2 = cv2.warpAffine(self.images[i2].copy(),m2,(shape[1] * 2,shape[0] * 2 ))
            
            R = cv2.getRotationMatrix2D((M[0, 2],M[1, 2]),(x[2]),1)
            dst2 = cv2.warpAffine(dst2,R,(shape[1] * 2,shape[0] * 2 ))
            
            cv2.imshow('get_transform', np.max(np.array([show_img1,dst2]), axis=0))
            cv2.waitKey(20)


        def func(x):
            ori = np.array([0, shape[0]//2])
            # if np.linalg.norm(x[:2] - ori) != shape[1]//2:
            #     return np.inf
            m2 = M.copy()
            m2[0, 2] += x[0]
            m2[1, 2] += x[1]

            ret = 0.0
            for i in range(2, self.nbClasses): 
                dst2 = cv2.warpAffine(imgs2[i - 2],m2,(shape[1] * 2,shape[0] * 2 ))            
                R = cv2.getRotationMatrix2D((M[0, 2],M[1, 2]),(x[2]),1)
                dst2 = cv2.warpAffine(dst2,R,(shape[1] * 2,shape[0] * 2 ))
                ret += sad(np.float32(dst1[i - 2]) , np.float32(dst2))
            
            return ret
        
        
        bounds = [(shape[1]//2 -1, shape[1]//2 +1), (-shape[0]//4 , shape[0]//4), (-30,30)]
        if _DEBUG:
            result = differential_evolution(func, bounds, popsize=100, maxiter=100, callback=call_back)
        else:
            result = differential_evolution(func, bounds, popsize=100, maxiter=100)


        return result.x, result.fun
            
    def run(self):
        transfoms = []
        shape = self.images[0].shape
        transfoms.append([0, shape[0]//2, 0])
        
        self.seg_images()

        for k, im in enumerate(self.images):
            gray = np.ascontiguousarray(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            self.images[k] = cv2.normalize(gray,None,0.,255,cv2.NORM_MINMAX)
        if not _DEBUG and 1:
            """Multiprocessing Block"""
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=len(self.images) - 1) 
            
            futures = [executor.submit(self.getTransform, i, i + 1) for i in range(len(self.images) - 1)]
            
            for f in futures:
                t, n = f.result()
                transfoms.append(t)
                print(n)
        else:
            """Single Process"""
            for f in [self.getTransform(i, i + 1) for i in range(len(self.images) - 1)]:
                t, n = f
                transfoms.append(t)
                print(n)
        sz = 4
        M = np.float32([[1,0,0],[0,1,shape[0]//2]])
        
        print("Start Recorder")
        cv2.waitKey(0)

        blended_images = []
        rimages = self.images[::-1]
        for i in range(0, len(self.images)-1):
            dst2 = rimages[i].copy()
            rtransfoms = transfoms[::-1]
            for k in range(i,len(transfoms)-1):
                T = np.array(rtransfoms[k])
                M2 = M.copy()
                if k == i:
                    M2[0, 2] += T[0]
                    M2[1, 2] += T[1]
                else:
                    M2[0, 2] = T[0]
                    M2[1, 2] = T[1]
                dst2 = cv2.warpAffine(dst2,M2,(shape[1] * sz,shape[0] * sz ))
                # cv2.imshow("MountView", np.max(np.array([*blended_images, dst2.astype('uint8')]), axis=0))
                # cv2.waitKey(500)
                R = cv2.getRotationMatrix2D((M[0, 2],M[1, 2]),(T[2]),1)
                dst2 = cv2.warpAffine(dst2,R,(shape[1] * sz,shape[0] * sz ))
                cv2.imshow("MountView", np.max(np.array([*blended_images, dst2.astype('uint8')]), axis=0))
                cv2.waitKey(700)
                
            blended_images.append(dst2)
            final_image = np.max(np.array(blended_images), axis=0)
            cv2.imshow("MountView", final_image.astype('uint8'))
            cv2.waitKey(2000)
        dst2 = rimages[-1].copy()
        M2 = M.copy()
        dst2 = cv2.warpAffine(dst2,M2,(shape[1] * sz,shape[0] * sz ))
        blended_images.append(dst2)

        final_image = np.max(np.array(blended_images), axis=0)

        final_image = cv2.normalize(final_image,None,0,255,cv2.NORM_MINMAX)

        final_image = final_image.astype('uint8')

        final_image = self.crop_image(final_image)
        cv2.imshow("MountView", final_image)
        cv2.waitKey(0)
        cv2.imwrite(self.output_path, final_image)
        print(self.output_path)

    def crop_image(self, img,tol=0):
        # img is image data
        # tol  is tolerance
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]


def main():
    cv2.namedWindow('MountView', cv2.WINDOW_NORMAL)

    cv2.namedWindow('get_transform', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('debug', cv2.WINDOW_NORMAL)
    list_files = glob.glob('./MONTAGEM/**/*.tif*', recursive=True)
    # list_files = glob.glob('/home/diegogomes/dev/deivid/MODELO MONTAGEM TESTE/TAKATA/VLD POS/IMAGEM CORTADA VLD/*.tif', recursive=True)
    # list_files = glob.glob('/home/diegogomes/dev/deivid/Baseline/João - Baseline/VLD/*.jpg', recursive=True)

    list_files = list(sorted(list_files))
    dirs_names = set()

    [dirs_names.add(os.path.dirname(d)) for d in  list_files ]

    image_name_per_dir = {k:[] for k in dirs_names}

    [image_name_per_dir[os.path.dirname(d)].append(d) for d in  list_files ]

    image_name_per_dir = {k:list(sorted(v)) for k,v in image_name_per_dir.items()}
    for k,v in image_name_per_dir.items():
        print(k)
        for vv in v:
            print('\t' + vv) 
    
    
    model = Registrator()
    for k,v in image_name_per_dir.items():
        model.setInputImages(k,v)
        model.run()


if __name__ == "__main__":
    main()


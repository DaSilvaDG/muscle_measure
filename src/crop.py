# %% [markdown]
# Primeiro fazemos os imports

# %%
import os
import shutil
import cv2
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def crop_images(search_path):
    
    files = [img for img in glob.glob(search_path, recursive=True)]
    print(files)
    images = [np.asarray(Image.open(img)) for img in files ]


    masks = [(img > 0).astype('f4') for img in images]

   
    final_mask = np.clip(np.sum(np.array(masks, dtype='f4'),axis=0), 0, 1)
    plt.imshow(final_mask)
    cnts, _ = cv2.findContours(
        cv2.cvtColor((final_mask * 255).astype("u1"), cv2.COLOR_BGR2GRAY),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )[-2:]



    draw = (final_mask * 255).astype("u1")
    cv2.drawContours(draw, cnts, -1, (0,255,0), 3)
    plt.imshow(draw)


    c = max(cnts, key=cv2.contourArea)


    draw = (final_mask * 255).astype("u1")
    cv2.drawContours(draw, [c], -1, (0,255,0), 5)
    # plt.imshow(draw)


    x, y, w, h = cv2.boundingRect(c)
    x, y, w, h


    croped = [img[y : y + h, x : x + w] for img in images]


    for i, (img, f) in enumerate(zip(croped, files)):
        cv2.imwrite(f"{f}_cropped.png", img)


# crop_images("./imagens/**/*.TIF")
# # %%

import os
import cv2
import numpy as np
import torch
from PIL import Image

def get_pics():
    path = './Distorted/Real'
    files = os.listdir(path)

    newpath = './EPIs'
    if not os.path.exists(newpath):
        os.mkdir(newpath)

    for file in files:
        pic_path = os.path.join(path, file)
        pic = cv2.imread(pic_path, cv2.IMREAD_COLOR)
        pic = np.reshape(pic, (434, 9, 625, 9, 3))
        pic = torch.from_numpy(pic).permute(1, 3, 0, 2, 4).numpy()
        print(pic.shape)

        for i in range(9):
            for j in range(434):
                img = pic[i, :, j, :, :]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # print(type(pic))
                pic_name = file.split('.')[0] + '_' + str(j + 1) + '.bmp'
                save_path = os.path.join(newpath, 'line' + str(i + 1))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = os.path.join(save_path, pic_name)
                print(save_path)
                cv2.imwrite(save_path, pic)

get_pics()
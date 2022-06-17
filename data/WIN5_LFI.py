import torch
import numpy as np
import cv2
import os
from einops import rearrange

class IQADataset(torch.utils.data.Dataset):
    def __init__(self, db_path, txt_filename, transform, scene_list, train_mode, train_size=0.8):
        super(IQADataset, self).__init__()

        self.db_path = db_path
        self.txt_filename = txt_filename
        self.transform = transform
        self.train_mode = train_mode
        # self.patch_size = patch_size
        self.scene_list = scene_list
        self.train_size = train_size

        self.data_dict = IQAdatalist(
            txt_filename=self.txt_filename,
            train_mode=self.train_mode,
            scene_list=self.scene_list,
            train_size=self.train_size
        ).load_data_dict()

        self.n_images = len(self.data_dict['d_img_list'])
        # print("len=>", self.n_images)


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # print("idx", idx % 132)
        d_img_name = self.data_dict['d_img_list'][idx % self.n_images]

        # LFI read => 3906 * 5625 * 3
        d_img_org = cv2.imread(os.path.join((self.db_path + '/Distorted/Real'), d_img_name), cv2.IMREAD_COLOR)
        # divide into 9 * 9 SAIs
        d_img_org = np.reshape(d_img_org, (434, 9, 625, 9, 3))
        d_img_org = torch.from_numpy(d_img_org).permute(1, 3, 0, 2, 4)

        # concat to 64 * 64 * 243 matrix
        d_out_img = np.zeros([64, 64, 3 * 81])
        rank = idx // self.n_images
        line = rank // 9
        row = rank % 9
        for i in range(9):
            for j in range(9):
                SAI = torch.squeeze(d_img_org[i, j, :, :, :]).numpy()
                d_out_img[:, :, i * 9 + j:i * 9 + j + 3] = SAI[64 * line:64 + 64 * line, 64 * row:64 * row + 64, :]

        d_out_img = rearrange(d_out_img, 'p1 p2 c -> c p1 p2')
        d_out_img = np.array(d_out_img).astype('float32') / 255

        score = self.data_dict['score_list'][idx % self.n_images]
        # d_out_img => 64, 64, 3 * 81
        sample = {'d_img_org': d_out_img, 'score': score}
        # print(d_out_img.dtype)

        return sample


class IQAdatalist():
    def __init__(self, txt_filename, train_mode, scene_list, train_size=0.8):
        self.txt_filename = txt_filename
        self.train_mode = train_mode
        self.train_size = train_size
        self.scene_list = scene_list

    def load_data_dict(self):
        scn_idx_list, d_img_list, score_list = [], [], []

        with open(self.txt_filename, 'r') as f:
            for line in f:
                scn_idx, dis, score = line.split()
                scn_idx = int(scn_idx)
                score = float(score)

                scene_list = self.scene_list

                if scn_idx in scene_list:
                    scn_idx_list.append(scn_idx)
                    d_img_list.append(dis)
                    score_list.append(score)

        # reshape score_list (1xn -> nx1)
        score_list = np.array(score_list)
        score_list = score_list.astype('float').reshape(-1, 1)

        data_dict = {'d_img_list': d_img_list, 'score_list': score_list}

        return data_dict



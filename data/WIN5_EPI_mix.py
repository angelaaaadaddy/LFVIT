import torch
import numpy as np
import cv2
import os

class IQADataset(torch.utils.data.Dataset):
    def __init__(self, db_path, txt_filename, transform, train_mode, scene_list, train_size=0.8):
        super(IQADataset, self).__init__()

        self.db_path = db_path
        self.txt_filename = txt_filename
        self.transform = transform
        self.train_mode = train_mode
        self.scene_list = scene_list
        self.train_size = train_size

        self.data_dict = IQADatalist(
            txt_filename = self.txt_filename,
            train_mode = self.train_mode,
            scene_list = self.scene_list,
            train_size = self.train_size
        ).load_data_dict()

        self.n_images = len(self.data_dict['d_img_list'])

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # d_img_org = 625 x 9 x 3
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img_org = cv2.imread(os.path.join((self.db_path + '/EPIs'), d_img_name), cv2.IMREAD_COLOR)
        # print('d_img_org1', d_img_org.shape)
        d_img_org = cv2.cvtColor(d_img_org, cv2.COLOR_BGR2LAB)
        # print('d_img_org2', d_img_org.shape)
        d_img_org = np.array(d_img_org).astype('float32') / 255
        # print('d_img_org3', d_img_org.shape)

        score = self.data_dict['score_list'][idx]
        sample = {'d_img_org': d_img_org, 'score': score}
        # print("d_img_org", d_img_org.shape)

        if self.transform:
            sample = self.transform(sample)

        return sample

class IQADatalist():
    def __init__(self, txt_filename, train_mode, scene_list, train_size=0.8):
        self.txt_filename = txt_filename
        self.train_mode = train_mode
        self.train_size = train_size
        self.scene_list = scene_list

    def load_data_dict(self):
        scn_idx_list, d_img_list, score_list = [], [], []

        with open(self.txt_filename, 'r') as f:
            for line in f:
                scn_idx, d_img, score = line.split()
                scn_idx = int(scn_idx)
                score = float(score)

                scene_list = self.scene_list

                if scn_idx in scene_list:
                    scn_idx_list.append(scn_idx)
                    score_list.append(score)
                    d_img_list.append(d_img)

        score_list = np.array(score_list)
        score_list = score_list.astype('float').reshape(-1, 1)

        data_dict = {'d_img_list': d_img_list, 'score_list': score_list}

        return data_dict
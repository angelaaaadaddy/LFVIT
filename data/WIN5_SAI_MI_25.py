import torch
import numpy as np
import cv2
import os

class IQADataset(torch.utils.data.Dataset):
    def __init__(self, db_path, txt_filename, transform, train_mode, scene_list):
        super(IQADataset, self).__init__()

        self.db_path = db_path
        self.txt_filename = txt_filename
        self.transform = transform
        self.train_mode = train_mode
        self.scene_list = scene_list

        self.data_dict = IQADatalist(
            txt_filename = self.txt_filename,
            train_mode = self.train_mode,
            scene_list = self.scene_list,
        ).load_data_dict()

        self.n_images = len(self.data_dict['d_img_list'])

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):

        numlist = [18, -9, -2, -1, 0, 1, 2, 9, 18]
        SAI_name = self.data_dict['d_img_list'][idx]
        SAI_no = int(SAI_name.split('_')[-1][:2])
        SAI_name_cut = '_'.join(SAI_name.split('_')[:-1])
        SAI_list = [SAI_name_cut + '_' + str(numlist[i] + SAI_no) + '.bmp' for i in range(len(numlist))]
        for i in range(len(SAI_list)):
            pic_name = os.path.join((self.db_path + '/SAIs'), SAI_list[i])
            pic = cv2.imread(pic_name, cv2.IMREAD_COLOR)
            if pic is None:
                print('个咋几', pic_name)
            pic = np.array(pic).astype('float32') / 255
            if i:
                pics = np.concatenate((pics, pic), axis=2)
            else:
                pics = pic


        score = self.data_dict['score_list'][idx]
        sample = {'d_img_org': pics, 'score': score}

        if self.transform:
            sample = self.transform(sample)

        return sample

class IQADatalist():
    def __init__(self, txt_filename, train_mode, scene_list):
        self.txt_filename = txt_filename
        self.train_mode = train_mode
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


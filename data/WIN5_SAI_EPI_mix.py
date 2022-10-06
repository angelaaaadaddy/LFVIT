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

        SAI_name = self.data_dict['d_img_list'][idx]
        LFI_name = SAI_name.split('_')
        SAI_no = int(LFI_name[-1][:2])
        LFI_name = '_'.join(LFI_name[:-1]) + '.bmp'
        SAI = cv2.imread(os.path.join((self.db_path + '/SAIs'), SAI_name), cv2.IMREAD_COLOR)
        LFI = cv2.imread(os.path.join((self.db_path + '/LFIs'), LFI_name), cv2.IMREAD_COLOR)
        LFI = np.reshape(LFI, (434, 9, 625, 9, 3))
        LFI = torch.from_numpy(LFI).permute(1, 3, 0, 2, 4)

        # EPI => 9 * 625 * 434
        # EPIs = np.zeros(9 * 625 * 434).reshape(9, 625, 434)
        line = int(np.floor(SAI_no / 9))
        # print("line", line)
        # for i in range(434):
        EPI = torch.squeeze(LFI[line, :, 217, :, :]).numpy()
        # EPI = cv2.cvtColor(EPI, cv2.COLOR_RGB2GRAY)
        EPI = np.array(EPI).astype('float32') / 255
        # EPIs[..., i] = img

        SAI = np.array(SAI).astype('float32') / 255


        score = self.data_dict['score_list'][idx]
        sample = {'SAI': SAI, 'EPI': EPI, 'score': score}

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


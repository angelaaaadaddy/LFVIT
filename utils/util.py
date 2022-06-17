import torch
import numpy as np

class MS_RandHorizontalFlip(object):
    def __call__(self, sample):
        d_img_org = sample['d_img_org']
        d_img_scale1 = sample['d_img_scale1']
        d_img_scale2 = sample['d_img_scale2']
        score = sample['score']

        prob_lr = np.random.random()

        if prob_lr > 0.5:
            d_img_org = np.fliplr(d_img_org).copy()
            if d_img_scale1 is not None:
                d_img_scale1 = np.fliplr(d_img_scale1).copy()
            if d_img_scale2 is not None:
                d_img_scale2 = np.fliplr(d_img_scale2).copy()

        sample = {'d_img_org': d_img_org, 'd_img_scale1': d_img_scale1, 'd_img_scale2': d_img_scale2, 'score': score}

        return sample

class NM_RandHorizontalFlip(object):
    def __call__(self, sample):
        d_img_org = sample['d_img_org']
        score = sample['score']

        prob_lr = np.random.random()

        if prob_lr > 0.5:
            d_img_org = np.fliplr(d_img_org).copy()


        sample = {'d_img_org': d_img_org, 'score': score}

        return sample



class MS_Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        d_img_org = sample['d_img_org']
        d_img_scale1 = sample['d_img_scale1']
        d_img_scale2 = sample['d_img_scale2']
        score = sample['score']

        d_img_org[:, :, 0] = (d_img_org[:, :, 0] - self.mean[0]) / self.var[0]
        d_img_org[:, :, 1] = (d_img_org[:, :, 1] - self.mean[1]) / self.var[1]
        d_img_org[:, :, 2] = (d_img_org[:, :, 2] - self.mean[2]) / self.var[2]

        if d_img_scale1 is not None:
            d_img_scale1[:, :, 0] = (d_img_scale1[:, :, 0] - self.mean[0]) / self.var[0]
            d_img_scale1[:, :, 1] = (d_img_scale1[:, :, 1] - self.mean[1]) / self.var[1]
            d_img_scale1[:, :, 2] = (d_img_scale1[:, :, 2] - self.mean[2]) / self.var[2]

        if d_img_scale2 is not None:
            d_img_scale2[:, :, 0] = (d_img_scale2[:, :, 0] - self.mean[0]) / self.var[0]
            d_img_scale2[:, :, 1] = (d_img_scale2[:, :, 1] - self.mean[1]) / self.var[1]
            d_img_scale2[:, :, 2] = (d_img_scale2[:, :, 2] - self.mean[2]) / self.var[2]

        sample = {'d_img_org': d_img_org, 'd_img_scale1': d_img_scale1, 'd_img_scale2': d_img_scale2, 'score': score}

        return sample

class NM_Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        d_img_org = sample['d_img_org']

        score = sample['score']

        d_img_org[:, :, 0] = (d_img_org[:, :, 0] - self.mean[0]) / self.var[0]
        d_img_org[:, :, 1] = (d_img_org[:, :, 1] - self.mean[1]) / self.var[1]
        d_img_org[:, :, 2] = (d_img_org[:, :, 2] - self.mean[2]) / self.var[2]


        sample = {'d_img_org': d_img_org, 'score': score}

        return sample

class MS_ToTensor(object):
    def __call__(self, sample):
        d_img_org = sample['d_img_org']
        if sample['d_img_scale1'] is not None:
            d_img_scale1 = sample['d_img_scale1']
        else:
            d_img_org = None
        if sample['d_img_scale2'] is not None:
            d_img_scale2 = sample['d_img_scale2']
        else:
            d_img_scale2 = None
        score = sample['score']

        d_img_org = np.transpose(d_img_org, (2, 0, 1))
        d_img_org = torch.from_numpy(d_img_org)

        if d_img_scale1 is not None:
            d_img_scale1 = np.transpose(d_img_scale1, (2, 0, 1))
            d_img_scale1 = torch.from_numpy(d_img_scale1)
        if d_img_scale2 is not None:
            d_img_scale2 = np.transpose(d_img_scale2, (2, 0, 1))
            d_img_scale2 = torch.from_numpy(d_img_scale2)

        score = torch.from_numpy(score)

        sample = {'d_img_org' : d_img_org, 'd_img_scale1' : d_img_scale1, 'd_img_scale2': d_img_scale2, 'score': score}

        return sample

class NM_ToTensor(object):
    def __call__(self, sample):
        d_img_org = sample['d_img_org']

        score = sample['score']

        d_img_org = np.transpose(d_img_org, (2, 0, 1))
        d_img_org = torch.from_numpy(d_img_org)

        score = torch.from_numpy(score)

        sample = {'d_img_org' : d_img_org, 'score': score}

        return sample




def RandShuffle(config):
    train_size = config.train_size

    if config.scenes == 'all':
        if config.db_name == 'WIN5-LID':
            scenes = list(range(10692))
        elif config.db_name == 'WIN5-EPI':
            scenes = list(range(1188))
        elif config.db_name == 'WIN5-LFI':
            scenes = list(range(132 * 54))
        elif config.db_name == 'WIN5-SAI-49':
            scenes = list(range(6468))
        elif config.db_name == 'WIN5-SAI-25':
            scenes = list(range(3300))
        elif config.db_name == 'WIN5-SAI-ALL-49':
            scenes = list(range(220 * 49))
    else:
        scenes = config.scenes

    n_scenes = len(scenes)
    n_train_scenes = int(np.floor(n_scenes * train_size))

    seed = np.random.random()
    random_seed = int(seed * 10)
    np.random.seed(random_seed)
    np.random.shuffle(scenes)
    train_scene_list = scenes[:n_train_scenes]
    test_scene_list = scenes[n_train_scenes:]

    return train_scene_list, test_scene_list

def clean_nan(matrix):
    for i in range(len(matrix)):
        if matrix[i] is np.nan:
            matrix[i] = 0

    return matrix
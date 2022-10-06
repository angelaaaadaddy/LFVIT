import torch
import numpy as np
import math

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

class MIX_Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        SAI = sample['SAI']
        EPI = sample['EPI']

        score = sample['score']

        SAI[:, :, 0] = (SAI[:, :, 0] - self.mean[0]) / self.var[0]
        SAI[:, :, 1] = (SAI[:, :, 1] - self.mean[1]) / self.var[1]
        SAI[:, :, 2] = (SAI[:, :, 2] - self.mean[2]) / self.var[2]

        EPI[:, :, 0] = (EPI[:, :, 0] - self.mean[0]) / self.var[0]
        EPI[:, :, 1] = (EPI[:, :, 1] - self.mean[1]) / self.var[1]
        EPI[:, :, 2] = (EPI[:, :, 2] - self.mean[2]) / self.var[2]

        sample = {'SAI': SAI, 'EPI': EPI, 'score': score}

        return sample

class MIX_ToTensor(object):
    def __call__(self, sample):
        SAI = sample['SAI']
        EPI = sample['EPI']

        score = sample['score']

        SAI = np.transpose(SAI, (2, 0, 1))
        SAI = torch.from_numpy(SAI)

        EPI = np.transpose(EPI, (2, 0, 1))
        EPI = torch.from_numpy(EPI)

        score = torch.from_numpy(score)

        sample = {'SAI': SAI, 'EPI': EPI, 'score': score}

        return sample

class split_RandHorizontalFlip(object):
    def __call__(self, sample):
        d_img_org = sample['d_img_org']
        score = sample['score']

        prob_lr = np.random.random()

        if prob_lr > 0.5:
            d_img_org = np.fliplr(d_img_org).copy()

        sample = {'d_img_org': d_img_org, 'score': score}

        return sample

class split_Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        vertical = sample['vertical']
        horizontal = sample['horizontal']
        left = sample['left']
        right = sample['right']

        score = sample['score']

        vertical[:, :, 0] = (vertical[:, :, 0] - self.mean[0]) / self.var[0]
        vertical[:, :, 1] = (vertical[:, :, 1] - self.mean[1]) / self.var[1]
        vertical[:, :, 2] = (vertical[:, :, 2] - self.mean[2]) / self.var[2]

        horizontal[:, :, 0] = (horizontal[:, :, 0] - self.mean[0]) / self.var[0]
        horizontal[:, :, 1] = (horizontal[:, :, 1] - self.mean[1]) / self.var[1]
        horizontal[:, :, 2] = (horizontal[:, :, 2] - self.mean[2]) / self.var[2]

        left[:, :, 0] = (left[:, :, 0] - self.mean[0]) / self.var[0]
        left[:, :, 1] = (left[:, :, 1] - self.mean[1]) / self.var[1]
        left[:, :, 2] = (left[:, :, 2] - self.mean[2]) / self.var[2]

        right[:, :, 0] = (right[:, :, 0] - self.mean[0]) / self.var[0]
        right[:, :, 1] = (right[:, :, 1] - self.mean[1]) / self.var[1]
        right[:, :, 2] = (right[:, :, 2] - self.mean[2]) / self.var[2]

        sample = {'vertical': vertical, 'horizontal': horizontal, 'left': left, 'right':right, 'score': score}

        return sample

class split_ToTensor(object):
    def __call__(self, sample):
        vertical = sample['vertical']
        horizontal = sample['horizontal']
        left = sample['left']
        right = sample['right']

        score = sample['score']

        vertical = np.transpose(vertical, (2, 0, 1))
        vertical = torch.from_numpy(vertical)

        horizontal = np.transpose(horizontal, (2, 0, 1))
        horizontal = torch.from_numpy(horizontal)

        left = np.transpose(left, (2, 0, 1))
        left = torch.from_numpy(left)

        right = np.transpose(right, (2, 0, 1))
        right = torch.from_numpy(right)

        score = torch.from_numpy(score)

        sample = {'vertical': vertical, 'horizontal': horizontal, 'left': left, 'right':right, 'score': score}

        return sample

def RandShuffle(config):
    # train_size = config.train_size

    if config.scenes == 'all':
        if config.db_name == 'WIN5-LID':
            scenes = list(range(10692))
        elif config.db_name == 'WIN5-EPI':
            scenes = list(range(88))
        elif config.db_name == 'WIN5-LFI':
            scenes = list(range(132 * 54))
        elif config.db_name == 'WIN5-SAI-49' or config.db_name == 'WIN5-MIX-49' \
                or config.db_name == 'WIN5-SAI-MI-49' or config.db_name == 'WIN5-SAI-MI-split-49':
            train_scenes = list(range(math.floor(132 * 49 * 0.8) + 1))
            test_scenes = list(range(math.floor(132 * 49 * 0.2)))
        elif config.db_name == 'WIN5-SAI-25' or config.db_name == 'WIN5-SAI-MI-25':
            train_scenes = list(range(math.floor(132 * 25 * 0.8) + 1))
            test_scenes = list(range(math.floor(132 * 25 * 0.2)))
    else:
        scenes = config.scenes

    # train_n_scenes = len(train_scenes)
    # test_n_scenes = len(test_scenes)
    # n_train_scenes = int(np.floor(n_scenes * train_size))

    seed = np.random.random()
    random_seed = int(seed * 10)
    np.random.seed(random_seed)
    np.random.shuffle(train_scenes)
    np.random.shuffle(test_scenes)
    train_scene_list = train_scenes
    test_scene_list = test_scenes

    return train_scene_list, test_scene_list

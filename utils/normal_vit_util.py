import torch
import numpy as np

class RandHorizontalFlip(object):
    def __call__(self, sample):
        d_img_org = sample['d_img_org']
        score = sample['score']

        prob_lr = np.random.random()

        if prob_lr > 0.5:
            d_img_org = np.fliplr(d_img_org).copy()


        sample = {'d_img_org': d_img_org, 'score': score}

        return sample


class Normalize(object):
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

class ToTensor(object):
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

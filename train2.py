import os
, import torch
from model.resnet50 import resnet50_backbone2
from model.vit import NM_IQAregression
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.util import RandShuffle, NM_Normalize, NM_RandHorizontalFlip, NM_ToTensor
from trainer import nm_train_epoch, nm_val_epoch
from option.config import Config
from model.simple_vit import VisionTransformer

def LFI():
    config = Config({
        # device
        'gpu_id': "0",
        'num_workers': 8,

        # dataset
        'db_name': 'WIN5-LFI',
        'txt_filename': './IQA_list/WIN5-LFI.txt',
        'db_path': '/Users/mianmaokuchuanma/DATABASE/Win5-LID/Win5-LID',
        'batch_size': 2,
        'train_size': 0.8,
        'patch_size': 16,
        'scenes': 'all',
        'save_freq': 10,
        'val_freq': 5,

        # optimization & training parameters
        'lr_rate': 1e-4,
        'momentum': 0.9,
        'weight_decay': 0,
        'T_max': 3e4,
        'eta_min': 0,
        'n_epoch': 100,

        # ViT structure
        'n_enc_seq': 16 * 16,
        'n_layer': 14,
        'd_hidn': 384,
        'i_pad': 0,
        'd_ff': 384,
        'd_MLP_head': 1152,
        'attn_head': 6,
        'd_head': 384,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'ln_eps': 1e-12,
        'n_output': 1,
        'Grid': 10,

        # load & save checkpoint
        'snap_path': './LFI_weights',
        'checkpoint': None,
    })

    config.device = torch.device('cuda:%s' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Using GPU %s" % config.gpu_id)
    else:
        print("Using CPU")

    # data selection
    if config.db_name == 'WIN5-LID':
        from data.win5lid import IQADataset
    elif config.db_name == 'WIN5-LFI':
        from data.win5_LFI import IQADataset
    elif config.db_name == 'WIN5-EPI':
        from data.win5epi import IQADataset

    # dataset separation(8:2)
    train_scene_list, test_scene_list = RandShuffle(config)
    print("number of train scenes: %d" % len(train_scene_list))
    print("number of test scenes %d" % len(test_scene_list))

    # data load
    train_dataset = IQADataset(
        db_path=config.db_path,
        txt_filename=config.txt_filename,
        transform=transforms.Compose([NM_Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), NM_ToTensor()]),
        train_mode=True,
        scene_list=train_scene_list,
        train_size=config.train_size
    )

    test_dataset = IQADataset(
        db_path=config.db_path,
        txt_filename=config.txt_filename,
        transform=transforms.Compose([NM_Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), NM_ToTensor()]),
        train_mode=False,
        scene_list=test_scene_list,
        train_size=config.train_size
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                              drop_last=True, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                             drop_last=True, shuffle=True)

    # create model
    model_backbone = resnet50_backbone2().to(config.device)
    model_transformer = VisionTransformer(config).to(config.device)

    # loss function & optimization
    criterion = torch.nn.L1Loss()
    params = list(model_transformer.parameters()) + list(model_backbone.parameters())
    optimizer = torch.optim.SGD(params, lr=config.lr_rate, weight_decay=config.weight_decay, momentum=config.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    # load weights & optimizer
    if config.checkpoint is not None:
        checkpoint = torch.load(config.checkpoint)
        model_backbone.load_state_dict(checkpoint['model_backbone_state_dict'])
        model_transformer.load_state_dict(checkpoint['model_transformer_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        start_epoch = 0

    # make dictionary for saving weights
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    # train & validation
    for epoch in range(start_epoch, config.n_epoch):
        loss, rho_s, rho_p = nm_train_epoch(config, epoch, model_transformer, model_backbone, train_loader,
                                            optimizer, criterion, scheduler)

        if (epoch + 1) % config.val_freq == 0:
            loss, rho_s, rho_p = nm_val_epoch(config, epoch, model_transformer, model_backbone, criterion,
                                              test_loader)

if __name__ == '__main__':
    LFI()

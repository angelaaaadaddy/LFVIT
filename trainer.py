import os
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr
from utils.util import clean_nan

def ms_train_epoch(config, epoch, model_transformer, model_backbone, train_loader, optimizer, criterion, scheduler):
    losses = []

    model_transformer.train()
    model_backbone.train()

    # input mask (batch_size x len_seq+1)
    mask_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device)

    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    save_txt_path = os.path.join(config.snap_path, config.train_name + '.txt')
    f = open(save_txt_path, mode='a')

    for data in tqdm(train_loader):

        d_img_org = data['d_img_org'].to(config.device)
        d_img_scale1 = data['d_img_scale1'].to(config.device)

        d_img_scale2 = data['d_img_scale2'].to(config.device)

        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        feat_dis_org = model_backbone(d_img_org)
        feat_dis_scale1 = model_backbone(d_img_scale1)
        # print("feat_dis_scale1", feat_dis_scale1.size())

        feat_dis_scale2 = model_backbone(d_img_scale2)
        # print("feat_dis_scale2", feat_dis_scale2.size())

        # weight update
        optimizer.zero_grad()

        pred = model_transformer(mask_inputs, feat_dis_org, feat_dis_scale1, feat_dis_scale2)
        loss = criterion(torch.squeeze(pred), labels)
        loss_val = loss.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    save_info = '[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p)
    f.write(save_info + '\n')

    print('[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p))

    # save weights
    if (epoch+1) % config.save_freq == 0:
        weights_filename = "epoch%d.pth" % (epoch+1)
        weights_file = os.path.join(config.snap_path, weights_filename)
        torch.save({
            'epoch': epoch,
            'model_backbone_state_dict': model_backbone.state_dict(),
            'model_transformer_state_dict': model_transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        print('save weights of epoch %d' % (epoch + 1))

    return np.mean(losses), rho_s, rho_p

def ms_val_epoch(config, epoch, model_transformer, model_backbone, criterion, test_loader):
    with torch.no_grad():
        losses = []
        model_transformer.eval()
        model_backbone.eval()

    mask_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device)


    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    save_txt_path = os.path.join(config.snap_path, config.train_name + '.txt')
    f = open(save_txt_path, mode='a')

    for data in tqdm(test_loader):

        d_img_org = data['d_img_org'].to(config.device)
        d_img_scale1 = data['d_img_scale1'].to(config.device)
        d_img_scale2 = data['d_img_scale2'].to(config.device)

        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        feat_dis_org = model_backbone(d_img_org)
        feat_dis_scale1 = model_backbone(d_img_scale1)
        feat_dis_scale2 = model_backbone(d_img_scale2)

        pred = model_transformer(mask_inputs, feat_dis_org, feat_dis_scale1, feat_dis_scale2)

        # compute loss
        loss = criterion(torch.squeeze(pred), labels)
        loss_val = loss.item()
        losses.append(loss_val)

        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    save_info = '[test] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p)
    f.write(save_info + '\n')

    print('test epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p))

    return np.mean(losses), rho_s, rho_p


def nm_train_epoch(config, epoch, model_transformer, model_backbone, train_loader, optimizer, criterion, scheduler):
    losses = []

    model_transformer.train()
    model_backbone.train()

    # input mask (batch_size x len_seq+1)
    mask_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device)

    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    save_txt_path = os.path.join(config.snap_path, config.train_name + '.txt')
    f = open(save_txt_path, mode='a')


    for data in tqdm(train_loader):
        # print("pre_size", data['d_img_org'].shape)
        d_img_org = data['d_img_org'].to(config.device)
        # print("after_size", d_img_org.shape)

        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)
        # if torch.any(torch.isnan(d_img_org)):
        #     print("原始输入数据不对")

        feat_dis_org = model_backbone(d_img_org)
        # if torch.any(torch.isnan(feat_dis_org)):
        #     print("经过backbone的数据不对")

        # weight update
        optimizer.zero_grad()

        pred = model_transformer(mask_inputs, feat_dis_org)
        # print("pred", pred)
        loss = criterion(torch.squeeze(pred), labels)
        # print("loss", loss.item())
        loss_val = loss.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        # print("pred_batch_numpy", pred_batch_numpy)
        # pred_batch_numpy = clean_nan(pred_batch_numpy)
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)


    # print("pred_epoch", pred_epoch)
    # print("labels_epoch", labels_epoch)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    save_info = '[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p)
    f.write(save_info + '\n')

    print('[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p))

    # save weights
    if (epoch+1) % config.save_freq == 0:
        weights_filename = "epoch%d.pth" % (epoch+1)
        weights_file = os.path.join(config.snap_path, weights_filename)
        torch.save({
            'epoch': epoch,
            'model_backbone_state_dict': model_backbone.state_dict(),
            'model_transformer_state_dict': model_transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        print('save weights of epoch %d' % (epoch + 1))

    return np.mean(losses), rho_s, rho_p

def nm_val_epoch(config, epoch, model_transformer, model_backbone, criterion, test_loader):
    with torch.no_grad():
        losses = []
        model_transformer.eval()
        model_backbone.eval()

    mask_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device)


    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    save_txt_path = os.path.join(config.snap_path, config.train_name + '.txt')
    f = open(save_txt_path, mode='a')

    for data in tqdm(test_loader):

        d_img_org = data['d_img_org'].to(config.device)

        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        feat_dis_org = model_backbone(d_img_org)

        pred = model_transformer(mask_inputs, feat_dis_org)

        # compute loss
        loss = criterion(torch.squeeze(pred), labels)
        loss_val = loss.item()
        losses.append(loss_val)

        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    save_info = '[test] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p)
    f.write(save_info + '\n')
    print('test epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p))

    return np.mean(losses), rho_s, rho_p


def smViT_train_epoch(config, epoch, model_transformer, model_backbone, train_loader, optimizer, criterion, scheduler):
    losses = []

    model_transformer.train()
    model_backbone.train()

    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    save_txt_path = os.path.join(config.snap_path, config.train_name + '.txt')
    f = open(save_txt_path, mode='a')

    for data in tqdm(train_loader):

        d_img_org = data['d_img_org'].to(config.device)

        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        feat_dis_org = model_backbone(d_img_org)
        # print("feat_dis_org", feat_dis_org.size())

        # weight update
        optimizer.zero_grad()

        pred = model_transformer(feat_dis_org)
        loss = criterion(torch.squeeze(pred), labels)
        loss_val = loss.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    save_info = '[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p)
    f.write(save_info + '\n')
    print('[train] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p))

    # save weights
    if (epoch+1) % config.save_freq == 0:
        weights_filename = "epoch%d.pth" % (epoch+1)
        weights_file = os.path.join(config.snap_path, weights_filename)
        torch.save({
            'epoch': epoch,
            'model_backbone_state_dict': model_backbone.state_dict(),
            'model_transformer_state_dict': model_transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        print('save weights of epoch %d' % (epoch + 1))

    return np.mean(losses), rho_s, rho_p


def smViT_val_epoch(config, epoch, model_transformer, model_backbone, criterion, test_loader):
    with torch.no_grad():
        losses = []
        model_transformer.eval()
        model_backbone.eval()

    # mask_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device)

    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    save_txt_path = os.path.join(config.snap_path, config.train_name + '.txt')
    f = open(save_txt_path, mode='a')

    for data in tqdm(test_loader):

        d_img_org = data['d_img_org'].to(config.device)

        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        feat_dis_org = model_backbone(d_img_org)

        pred = model_transformer(feat_dis_org)

        # compute loss
        loss = criterion(torch.squeeze(pred), labels)
        loss_val = loss.item()
        losses.append(loss_val)

        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    save_info = '[test] epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p)
    f.write(save_info + '\n')

    print('test epoch:%d / loss:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), rho_s, rho_p))

    return np.mean(losses), rho_s, rho_p
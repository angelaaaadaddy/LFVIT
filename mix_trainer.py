import math
import os
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr
from model.get_epi_feature import EPI_special

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
        SAI = data['SAI'].to(config.device)
        EPI = data['EPI'].to(config.device)


        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        # 20 * 1
        feat_EPI = EPI_special(EPI)
        # 20 * 14
        feat_SAI = model_backbone(SAI)

        feat_dis_org = torch.cat((feat_SAI, feat_EPI), dim=1)


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
        SAI = data['SAI'].to(config.device)
        EPI = data['EPI'].to(config.device)

        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        # 20 * 1
        feat_EPI = EPI_special(EPI)
        # 20 * 14
        feat_SAI = model_backbone(SAI)

        feat_dis_org = torch.cat((feat_SAI, feat_EPI), dim=1)
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


def smViT_train_epoch(config, epoch, model_transformer, model_backbone, model_epi, train_loader, optimizer, criterion, criterion2, scheduler):
    losses = []

    model_transformer.train()
    model_backbone.train()
    model_epi.train()

    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    srocc_cal = dict()

    save_txt_path = os.path.join(config.snap_path, config.train_name + '.txt')
    f = open(save_txt_path, mode='a')

    for data in tqdm(train_loader):

        SAI = data['SAI'].to(config.device)
        EPI = data['EPI'].to(config.device)

        labels = data['score']
        scores = data['score'].numpy()
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        feat_SAI = model_backbone(SAI)
        # 9 * 625 => 20 * 1
        feat_EPI = model_epi(EPI)

        feat_dis_org = torch.cat((feat_SAI, feat_EPI), dim=2)

        # weight update
        optimizer.zero_grad()

        pred = model_transformer(feat_dis_org)
        pred_np = pred.cpu().detach().numpy()
        for i in range(len(scores)):
            lab = scores[i][0]
            score = pred_np[i][0]
            if lab in srocc_cal.keys():
                srocc_cal[lab].append(score)
            else:
                srocc_cal[lab] = [score]

        loss = criterion(torch.squeeze(pred), labels)
        loss2 = criterion2(torch.squeeze(pred), labels)
        loss_val = loss.item()
        loss2_val = math.sqrt(loss2.item())
        losses.append(loss_val)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)


    labels_epoch_srocc = []
    pred_epoch_srocc = []
    for k, v in srocc_cal.items():
        labels_epoch_srocc.append(k)
        pred_epoch_srocc.append(np.mean(v))

    labels_epoch_srocc = np.array(labels_epoch_srocc)
    pred_epoch_srocc = np.array(pred_epoch_srocc)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch_srocc), np.squeeze(labels_epoch_srocc))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    save_info = '[train] epoch:%d / loss:%f / RMSE:%4f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), loss2_val, rho_s, rho_p)
    f.write(save_info + '\n')
    print('[train] epoch:%d / loss:%f / RMSE:%4f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), loss2_val, rho_s, rho_p))

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


def smViT_val_epoch(config, epoch, model_transformer, model_backbone, model_epi, criterion, criterion2, test_loader):
    with torch.no_grad():
        losses = []
        # losses2 = []
        model_transformer.eval()
        model_backbone.eval()
        model_epi.eval()

    # mask_inputs = torch.ones(config.batch_size, config.n_enc_seq+1).to(config.device)

    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    srocc_cal = dict()

    save_txt_path = os.path.join(config.snap_path, config.train_name + '.txt')
    f = open(save_txt_path, mode='a')

    for data in tqdm(test_loader):

        SAI = data['SAI'].to(config.device)
        EPI = data['EPI'].to(config.device)

        labels = data['score']
        scores = data['score'].numpy()
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)

        # 9 * 625 => 20 * 1
        feat_EPI = model_epi(EPI)
        # 20 * 14
        feat_SAI = model_backbone(SAI)

        feat_dis_org = torch.cat((feat_SAI, feat_EPI), dim=2)

        pred = model_transformer(feat_dis_org)
        pred_np = pred.cpu().detach().numpy()
        for i in range(len(scores)):
            lab = scores[i][0]
            score = pred_np[i][0]
            if lab in srocc_cal.keys():
                srocc_cal[lab].append(score)
            else:
                srocc_cal[lab] = [score]

        # compute loss
        loss = criterion(torch.squeeze(pred), labels)
        loss2 = criterion2(torch.squeeze(pred), labels)
        loss_val = loss.item()
        loss2_val = math.sqrt(loss2.item())
        losses.append(loss_val)

        # save results in one epoch
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    labels_epoch_srocc = []
    pred_epoch_srocc = []
    for k, v in srocc_cal.items():
        labels_epoch_srocc.append(k)
        pred_epoch_srocc.append(np.mean(v))

    labels_epoch_srocc = np.array(labels_epoch_srocc)
    pred_epoch_srocc = np.array(pred_epoch_srocc)

    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch_srocc), np.squeeze(labels_epoch_srocc))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))


    save_info = '[test] epoch:%d / loss:%f / RMSE:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), loss2_val, rho_s, rho_p)
    f.write(save_info + '\n')

    print('test epoch:%d / loss:%f / RMSE:%f / SROCC:%4f / PLCC:%4f' % (epoch + 1, loss.item(), loss2_val, rho_s, rho_p))

    return np.mean(losses), rho_s, rho_p
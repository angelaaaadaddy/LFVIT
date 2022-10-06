import torch.nn.functional as F
import torch
from model.distill_vit import DistillWrapper, DistillableViT
from tqdm import tqdm

def distill_train_epoch(config, epoch, model_teacher, model_student, train_loader, optimizer, criterion, scheduler):
    losses = []

    model_teacher.train()
    model_student.train()

    # mask_inputs = torch.ones(config.batch_size, config.n_enc_seq + 1).to(config.device)

    pred_epoch = []
    labels_epoch = []

    for data in tqdm(train_loader):

        d_img_org = data['d_img_org'].to(config.device)

        labels = data['score']
        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config.device)
        
        distiller = DistillWrapper(config)

        loss = distiller(d_img_org, labels)
        loss_val = loss.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pred_batch_numpy = pred.data.cpu().numpy()
        labels_
        

        
        
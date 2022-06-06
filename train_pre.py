import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
import os

from data_loader import DataLoader
from model_pretrain import vit_IQAModel
from utils import lr_scheduler
from utils import save_checkpoint, load_checkpoint
import config


def train_fn():
    folder_path = {
        'live':     config.DATA_PATH['live'],
        'csiq':     config.DATA_PATH['csiq'],
        'tid2013':  config.DATA_PATH['tid2013'],
        'kadid10k': config.DATA_PATH['kadid10k'],
        'livec':    config.DATA_PATH['livec'],
        'koniq':    config.DATA_PATH['koniq'],
        # 'fblive':    config.data_path['fblive'],
        }

    img_num = {
        'live':     list(range(0, 29)),
        'csiq':     list(range(0, 30)),
        'kadid10k': list(range(0, 80)),
        'tid2013':  list(range(0, 25)),
        'livec':    list(range(0, 1162)),
        'koniq':    list(range(0, 10073)),
        # 'fblive':   list(range(0, 39810)),
        }

    print('Training and Testing on <{}> dataset'.format(config.DATASET.upper()))

    if not os.path.exists(config.MODEL_PATH_P):
        os.mkdir(config.MODEL_PATH_P)

    # fix the seed if needed for reproducibility
    if config.SEED == 0:
        pass
    else:
        print('SEED = {}'.format(config.SEED))
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)

    total_num_images = img_num[config.DATASET]
    random.shuffle(total_num_images)
    train_index = total_num_images[0:int(round(1.0 * len(total_num_images)))]

    # build train and test loader
    dataloader_train = DataLoader(config.DATASET,
                              folder_path[config.DATASET],
                              train_index,
                              config.PATCH_SIZE,
                              config.TRAIN_PATCH_NUM,
                              config.BATCH_SIZE,
                              istrain=True).get_data()

    device = config.DEVICE
    model = vit_IQAModel().to(device)

    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    # loss_mae = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_P)

    if config.LOAD_MODEL:
        dset = '2'
        load_path = config.CKPT_P.format(dset)
        load_path = os.path.join(config.MODEL_PATH_P, load_path)
        load_checkpoint(load_path, model, optimizer, lr=config.LEARNING_RATE_P)

    model.train()
    for epoch in range(config.NUM_EPOCHS):
        losses = []
        print(f'+====================+ Training Epoch: {epoch} +====================+')
        loop = tqdm(dataloader_train)
        optimizer = lr_scheduler(optimizer, epoch)
        for batch_idx, (dist, dist_g, ref, diff) in enumerate(loop):
            dist = dist.to(config.DEVICE).float()
            dist_g = dist_g.to(config.DEVICE).float()
            diff = diff.to(config.DEVICE).float()
            ref = ref.to(config.DEVICE).float()

            optimizer.zero_grad()
            out, attn_list = model(dist)

            loss = loss_fn(out, diff) + 0.2 * loss_fn(dist_g-diff, ref)

            loss.backward()
            optimizer.step()
            losses.append(loss)
            loop.set_postfix(loss=loss.item())

        print(f'Loss: {sum(losses)/len(losses)}')

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_path = config.CKPT_P.format(epoch)
            save_path = os.path.join(config.MODEL_PATH_P, save_path)
            save_checkpoint(model, optimizer, filename=save_path)


if __name__ == '__main__':
    train_fn()

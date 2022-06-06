import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
import os

from data_loader import DataLoader
import model_final
from utils import calc_coefficient, lr_scheduler
from utils import save_checkpoint, load_checkpoint
import model_pretrain
import config
import csv


AVG_COEF = []


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

    if not os.path.exists(config.MODEL_PATH_F):
        os.mkdir(config.MODEL_PATH_F)

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

    # Randomly select 80% images for training and the rest for testing
    random.shuffle(total_num_images)
    train_index = total_num_images[0:int(round(0.8 * len(total_num_images)))]
    test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]

    # build train and test loader
    dataloader_train = DataLoader(config.DATASET,
                              folder_path[config.DATASET],
                              train_index,
                              config.PATCH_SIZE,
                              config.TRAIN_PATCH_NUM,
                              config.BATCH_SIZE,
                              istrain=True).get_data()

    dataloader_test = DataLoader(config.DATASET,
                             folder_path[config.DATASET],
                             test_index,
                             config.PATCH_SIZE,
                             config.TEST_PATCH_NUM,
                             istrain=False).get_data()

    device = config.DEVICE
    model = model_final.vit_IQAModel(pretrained=True).to(device)

    # ========================================================
    # load pretrain model
    load_path = config.CKPT_P.format('30')
    load_path = os.path.join(config.MODEL_PATH_P, load_path)
    ckpt = torch.load(load_path, map_location=config.DEVICE)
    model_err = model_pretrain.vit_IQAModel().to(config.DEVICE)
    model_err.load_state_dict(ckpt['state_dict'])
    model_err.requires_grad_(False)
    # ========================================================


    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_F)

    if config.LOAD_MODEL:
        load_checkpoint(config.CKPT_F, model, optimizer, lr=config.LEARNING_RATE_F)

    model.train()
    coef = {}
    best_srocc = 0
    corre_plcc = 0
    best_model = None
    best_optimzer = None
    for epoch in range(config.NUM_EPOCHS):
        losses = []
        print(f'+====================+ Training Epoch: {epoch} +====================+')
        loop = tqdm(dataloader_train)
        optimizer = lr_scheduler(optimizer, epoch)

        for batch_idx, (dist, rating) in enumerate(loop):
            batch_size = dist.shape[0]
            dist = dist.to(config.DEVICE).float()
            rating = rating.reshape(batch_size, -1).to(config.DEVICE).float()

            # ========================================================
            emap, _ = model_err(dist)
            inp = (dist, emap)

            optimizer.zero_grad()
            out, attn_list = model(inp)
            loss = loss_fn(out, rating)
            # ========================================================
            loss.backward()
            optimizer.step()
            losses.append(loss)
            loop.set_postfix(loss=loss.item())

        print(f'Loss: {sum(losses)/len(losses):.5f}')
        print(f'+====================+ Testing Epoch: {epoch} +====================+')
        sp, pl = calc_coefficient(dataloader_test, model, config.DEVICE)
        print(f'SROCC: {sp:.3f}, PLCC: {pl:.3f}')

        # save models
        if sp > best_srocc:
            best_srocc = sp
            corre_plcc = pl
            best_model = model
            best_optimzer = optimizer
        print(f'BEST SROCC: {best_srocc:.3f} & PLCC: {corre_plcc:.3f}')
    coef['srocc'], coef['plcc'] = best_srocc, corre_plcc
    return coef, best_model, best_optimzer


if __name__ == '__main__':
    srocc_max = 0
    for i in range(config.EXP_CNT):
        coef, model, optimizer = train_fn()
        AVG_COEF.append(coef)

        if coef['srocc'] > srocc_max:
            srocc_max = coef['srocc']
            save_path = config.CKPT_F.format(config.DATASET)
            save_path = os.path.join(config.MODEL_PATH_F, save_path)
            if config.SAVE_MODEL:
                save_checkpoint(model, optimizer, filename=save_path)

    headers = list(AVG_COEF[0].keys())
    with open(f'{config.DATASET}_Results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, headers)
        writer.writeheader()
        writer.writerows(AVG_COEF)

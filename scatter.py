import numpy as np
import matplotlib.pyplot as plt
import os
import config
import torch
from tqdm import tqdm
import model_pretrain, model_final
from data_loader import DataLoader
import random


def scatter_plot(dset):
    folder_path = {
        'live': config.DATA_PATH['live'],
        'csiq': config.DATA_PATH['csiq'],
        'tid2013': config.DATA_PATH['tid2013'],
        'kadid10k': config.DATA_PATH['kadid10k'],
        'livec': config.DATA_PATH['livec'],
        'koniq': config.DATA_PATH['koniq'],
        # 'fblive':    config.data_path['fblive'],
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'kadid10k': list(range(0, 80)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq': list(range(0, 10073)),
        # 'fblive':   list(range(0, 39810)),
    }

    # fix the seed if needed for reproducibility
    if config.SEED == 0:
        pass
    else:
        print('SEED = {}'.format(config.SEED))
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)

    total_num_images = img_num[dset]
    random.shuffle(total_num_images)
    test_index = total_num_images[int(round(0.8 * len(total_num_images))):len(total_num_images)]

    dataloader_test = DataLoader(dset,
                                 folder_path[dset],
                                 test_index,
                                 config.PATCH_SIZE,
                                 config.TEST_PATCH_NUM,
                                 istrain=False).get_data()

    device = config.DEVICE
    model = model_final.vit_IQAModel(pretrained=True).to(device)
    load_path = config.CKPT_F.format(dset)
    load_path = os.path.join(config.MODEL_PATH_F, load_path)
    ckpt = torch.load(load_path, map_location=config.DEVICE)
    model.load_state_dict(ckpt['state_dict'])

    load_path = config.CKPT_P.format('30')
    load_path = os.path.join(config.MODEL_PATH_P, load_path)
    ckpt = torch.load(load_path, map_location=config.DEVICE)
    model_err = model_pretrain.vit_IQAModel().to(config.DEVICE)
    model_err.load_state_dict(ckpt['state_dict'])
    model_err.requires_grad_(False)


    a = []
    b = []
    device = device

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(dataloader_test)):
            batch_size = x.shape[0]
            y = y.reshape(batch_size, -1)
            x, y = x.to(device).float(), y.to(device).float()
            e, _ = model_err(x)
            inp = (x, e)
            p, _ = model(inp)
            a.append(y.cpu().float())
            b.append(p.cpu().float())

        a = np.vstack(a)
        b = np.vstack(b)
        a = a[:, 0]
        b = b[:, 0]

        a = np.reshape(a, (-1, config.TEST_PATCH_NUM))
        b = np.reshape(b, (-1, config.TEST_PATCH_NUM))
        a = np.mean(a, axis=1)
        b = np.mean(b, axis=1)

    a = list(a)
    b = list(b)
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    p = np.polyfit(a, b, 1)
    y_p = np.polyval(p, a)

    plt.scatter(a, b, color='darkcyan', s=15)
    plt.plot(a, y_p, linewidth=2, color='crimson')
    plt.title(f'{dset}'.upper(), fontsize=12)
    plt.grid()
    plt.xlabel('Predicted score', fontsize=12)
    plt.ylabel('GT', fontsize=12)
    # plt.savefig("result.png", dpi=120, bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{dset}.png', dpi=600, bbox_inches='tight')
    # plt.savefig(f'{dset}.eps', dpi=600, bbox_inches='tight', format='eps')
    plt.show()
    model.train()


if __name__ == '__main__':
    scatter_plot(dset='tid2013')
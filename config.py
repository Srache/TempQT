import torch


LOAD_MODEL = False
SAVE_MODEL = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# BATCH_SIZE = 32
BATCH_SIZE = 8
LEARNING_RATE_P = 1e-4
# LEARNING_RATE_F = 3e-5
LEARNING_RATE_F = 2e-5
LEARNING_RATE_L = 1e-4
NUM_EPOCHS = 32
# CKPT_P = 'iqa_pretain_kadid10k_w-o_bn_{}.pt'
CKPT_P = 'iqa_pretain_kadid10k_{}.pt'
# CKPT_P = 'iqa_pretain_pipal_{}.pt'
CKPT_F = 'iqa_final_{}.pth'
# NORM_FACTOR = 100  # live
# NORM_MULTI_FACTOR = 1  # kadid10k
# NORM_ADD_FACTOR = 0  # kadid10k

DATASET = 'kadid10k'
MODEL_PATH_P = 'save_models_pretrain'
MODEL_PATH_F = 'save_models_score'
PATCH_SIZE = 224
TRAIN_PATCH_NUM = 1
TEST_PATCH_NUM = 1
SEED = 0
EXP_CNT = 10

# DATA_PATH = {
#    'live': 'E:\sjs\\NR-IQA-Final\dataset_iqa\live',
#    'kadid10k': 'E:\sjs\\NR-IQA-Final\dataset_iqa\kadid10k',
#    'csiq': 'E:\sjs\\NR-IQA-Final\dataset_iqa\csiq',
#    'tid2013': r'E:\sjs\\NR-IQA-Final\dataset_iqa\tid2013',
#    'koniq': 'E:\sjs\\NR-IQA-Final\dataset_iqa\koniq',
#    'livec': 'E:\sjs\\NR-IQA-Final\dataset_iqa\livec',
# }

# DATA_PATH = {
#      'live': '/home/sjs/srache/IQA_Final/dataset_iqa/live',
#      'kadid10k': '/home/sjs/srache/IQA_Final/dataset_iqa/kadid10k',
#      'csiq': '/home/sjs/srache/IQA_Final/dataset_iqa/csiq',
#      'tid2013': '/home/sjs/srache/IQA_Final/dataset_iqa/tid2013',
#      'koniq': '/home/sjs/srache/IQA_Final/dataset_iqa/koniq',
#      'livec': '/home/sjs/srache/IQA_Final/dataset_iqa/livec',
#  }

DATA_PATH = {
    'live': '/home/cxl/srache/NR-IQA/dataset_iqa/live',
    'kadid10k': '/home/cxl/srache/NR-IQA/dataset_iqa/kadid10k',
    'csiq': '/home/cxl/srache/NR-IQA/dataset_iqa/csiq',
    'tid2013': '/home/cxl/srache/NR-IQA/dataset_iqa/tid2013',
    'koniq': '/home/cxl/srache/NR-IQA/dataset_iqa/koniq',
    'livec': '/home/cxl/srache/NR-IQA/dataset_iqa/livec',
}

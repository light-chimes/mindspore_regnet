origin_config = {
    'MODEL': {'NUM_CLASSES': 1000, 'ACTIVATION_FUN': 'relu', 'SEED': 1, 'DEVICE_ID': 3,
              'TRAIN_BATCH_SIZE': 128,
              'TRAIN_IMAGE_SIZE': 224, 'INITIAL_LR': 0.001, 'SAVE_CHECKPOINTS': False,
              'SAVE_CHECKPOINT_STEPS': 1, 'KEEP_CHECKPOINT_MAX': 5,
              'EVAL_CHECKPOINT_PATH': '/home/regnet/new_regnet/ckpts/ms_regnet200mf.ckpt',
              'CHECKPOINT_SAVE_PATH': '/home/regnet/new_regnet/ckpts/',
              'TRAIN_DATASET_PATH': '/disk0/dataset/imagenet/train',
              'CHECKPOINT_PREFIX': 'RegNetX_datafixed_traineval_dvc7_lr0.00075_e100_',
              'EVAL_DATASET_PATH': '/disk0/dataset/imagenet/val', 'WEIGHT_DECAY': 5e-5,
              'NESTEROV_MOMENTUM': True,
              'WARM_UP_EPOCH': 5,
              'MOMENTUM': 0.9, 'MIN_LR': 0.0, 'WARM_UP_FACTOR': 0.1,
              'MAX_EPOCH': 100, 'IS_DISTRIBUTED': False, 'DEVICE_TARGET': 'Ascend', 'EVAL_BATCH_SIZE': 100,
              'PCA_STD': 0.1,
              'EVAL_IMAGE_SIZE': 256},
    'ANYNET': {'STEM_TYPE': 'simple_stem_in', 'STEM_W': 32, 'BLOCK_TYPE': 'res_bottleneck_block',
               'DEPTHS': [], 'WIDTHS': [], 'STRIDES': [], 'BOT_MULS': [], 'GROUP_WS': [],
               'HEAD_W': 0, 'SE_ON': False, 'SE_R': 0.25},
    'REGNET': {'STEM_TYPE': 'simple_stem_in', 'STEM_W': 32, 'BLOCK_TYPE': 'res_bottleneck_block',
               'STRIDE': 2,
               'SE_ON': False, 'SE_R': 0.25, 'DEPTH': 13, 'W0': 24, 'WA': 36.44, 'WM': 2.49, 'GROUP_W': 8,
               'BOT_MUL': 1.0, 'HEAD_W': 0}, 'BN': {'EPS': 1e-05, 'MOM': 0.9}}


def configdictify(d):
    if isinstance(d, dict):
        for i in d:
            d[i] = configdictify(d[i])
        return ConfigDict(d)
    else:
        return d


class ConfigDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


cfg = configdictify(origin_config)

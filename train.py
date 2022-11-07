import mindspore as ms
from mindspore import nn
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from config import cfg
from new_dataset import create_dataset
from lr_schedule import cos_warmup_lr
from regnet import RegNet


def train(arguments):
    print(arguments)
    seed = arguments['seed']
    device_id = arguments['device_id']
    batch_size = arguments['train_batch_size']
    save_checkpoint_steps = arguments['save_checkpoint_steps']
    keep_checkpoint_max = arguments['keep_checkpoint_max']
    checkpoint_prefix = arguments['checkpoint_prefix']
    checkpoint_save_path = arguments['checkpoint_save_path']
    dataset_path = arguments['train_dataset_path']
    max_epoch = arguments['max_epoch']
    is_distributed = arguments['is_distributed']
    device_target = arguments['device_target']
    image_size = arguments['train_image_size']
    ms.set_seed(seed)
    if device_target == 'Ascend':
        ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', save_graphs=False,
                       device_id=device_id)
        if is_distributed:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
            ms.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                         gradients_mean=True, strategy_ckpt_save_file='/home/regnet/strategy_ckpt')
            checkpoint_save_path = checkpoint_save_path + "ckpt_" + str(get_rank()) + "/"
            dataset = create_dataset(data_path=dataset_path, repeat_num=1, batch_size=batch_size,
                                     rank_id=rank_id,
                                     rank_size=device_num, image_size=image_size)
        else:
            dataset = create_dataset(data_path=dataset_path, repeat_num=1, batch_size=batch_size, rank_id=0,
                                     rank_size=1, image_size=image_size)
    elif device_target == 'GPU':
        if is_distributed:
            ms.set_context(mode=ms.GRAPH_MODE, device_target='GPU', save_graphs=False)
            init('nccl')
            ms.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
            dataset = create_dataset(data_path=dataset_path, repeat_num=1, batch_size=batch_size,
                                     rank_size=get_group_size(), rank_id=get_rank(), image_size=image_size)
        else:
            dataset = create_dataset(data_path=dataset_path, repeat_num=1, batch_size=batch_size, rank_id=0,
                                     rank_size=1, image_size=image_size)
    elif device_target == 'CPU':
        ms.set_context(mode=ms.GRAPH_MODE, device_target='CPU', save_graphs=False)
        dataset = create_dataset(data_path=dataset_path, repeat_num=1, batch_size=batch_size, rank_id=0,
                                 rank_size=1, image_size=image_size)
    ds_train = dataset
    print('dataset size is : \n', ds_train.get_dataset_size())
    regnet = RegNet()
    ce_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    regnet.set_train(True)

    opt = ms.nn.SGD(regnet.trainable_params(),
                    learning_rate=cos_warmup_lr(ds_train.get_dataset_size()), momentum=arguments['momentum'],
                    weight_decay=arguments['weight_decay'], nesterov=arguments['nesterov_momentum'])

    model = Model(regnet, loss_fn=ce_loss, optimizer=opt)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callback_list = [LossMonitor(), time_cb]
    if arguments['save_checkpoints']:
        config_ck = CheckpointConfig(save_checkpoint_steps=save_checkpoint_steps,
                                     keep_checkpoint_max=keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=checkpoint_prefix, directory=checkpoint_save_path, config=config_ck)
        callback_list.append(ckpoint_cb)
    model.train(max_epoch, ds_train, callbacks=callback_list,
                dataset_sink_mode=True)


if __name__ == '__main__':
    args = cfg['MODEL']
    model_args = {}
    for i in args:
        model_args[i.lower()] = args[i]
    train(model_args)

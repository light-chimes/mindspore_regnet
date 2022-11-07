import mindspore as ms
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from new_dataset import create_dataset
from regnet import RegNet
from config import cfg
from lr_schedule import cos_warmup_lr


def train(arguments):
    print(arguments)
    seed = arguments['seed']
    device_id = arguments['device_id']
    train_batch_size = arguments['train_batch_size']
    eval_batch_size = arguments['eval_batch_size']
    save_checkpoint_steps = arguments['save_checkpoint_steps']
    keep_checkpoint_max = arguments['keep_checkpoint_max']
    checkpoint_prefix = arguments['checkpoint_prefix']
    checkpoint_save_path = arguments['checkpoint_save_path']
    train_dataset_path = arguments['train_dataset_path']
    eval_dataset_path = arguments['eval_dataset_path']
    max_epoch = arguments['max_epoch']
    train_image_size = arguments['train_image_size']
    eval_image_size = arguments['eval_image_size']
    ms.set_seed(seed)
    ms.set_context(mode=ms.GRAPH_MODE, device_target=arguments['device_target'], save_graphs=False,
                   device_id=device_id)
    ds_train = create_dataset(data_path=train_dataset_path, do_train=True, repeat_num=1, batch_size=train_batch_size,
                              rank_id=0,
                              rank_size=1, image_size=train_image_size)
    ds_eval = create_dataset(data_path=eval_dataset_path, do_train=False, repeat_num=1, batch_size=eval_batch_size,
                             rank_id=0,
                             rank_size=1, image_size=eval_image_size)
    print('dataset size is : \n', ds_train.get_dataset_size())
    regnet = RegNet()
    ce_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    regnet.set_train(True)

    # opt = ms.nn.SGD(regnet.trainable_params(),
    #                 learning_rate=cos_warmup_lr(ds_train.get_dataset_size()), momentum=arguments['momentum'],
    #                 weight_decay=arguments['weight_decay'], nesterov=arguments['nesterov_momentum'])
    opt = ms.nn.Adam(regnet.trainable_params(),learning_rate=arguments['initial_lr'])
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callback_list = [LossMonitor(), time_cb]
    if arguments['save_checkpoints']:
        config_ck = CheckpointConfig(save_checkpoint_steps=save_checkpoint_steps,
                                     keep_checkpoint_max=keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=checkpoint_prefix, directory=checkpoint_save_path, config=config_ck)
        callback_list.append(ckpoint_cb)
    model = Model(network=regnet, metrics={'accuracy'}, loss_fn=ce_loss, optimizer=opt)
    model.fit(epoch=max_epoch, train_dataset=ds_train, valid_dataset=ds_eval, dataset_sink_mode=True,
              valid_dataset_sink_mode=True, callbacks=callback_list)


if __name__ == '__main__':
    args = cfg['MODEL']
    model_args = {}
    for i in args:
        model_args[i.lower()] = args[i]
    train(model_args)

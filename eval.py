import mindspore as ms
from mindspore import Model
from mindspore import nn
from new_dataset import create_dataset
from regnet import RegNet
from config import cfg


def evaluate(arguments):
    seed = arguments['seed']
    device_id = arguments['device_id']
    eval_batch_size = arguments['eval_batch_size']
    eval_dataset_path = arguments['eval_dataset_path']
    eval_image_size = arguments['eval_image_size']
    train_image_size = arguments['train_image_size']
    ms.set_seed(seed)
    ms.set_context(mode=ms.GRAPH_MODE, device_target=arguments['device_target'], save_graphs=False,
                   device_id=device_id)
    regnet = RegNet()
    param_dict = ms.load_checkpoint(arguments['eval_checkpoint_path'])
    regnet.set_train(False)
    # print(regnet.parameters_dict())
    print('Load trained model done. {}'.format(arguments['eval_checkpoint_path']))
    regnet.init_parameters_data()
    ms.load_param_into_net(regnet, param_dict)
    ce_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    model = Model(regnet, loss_fn=ce_loss, metrics={
        'Top1-Acc': nn.Top1CategoricalAccuracy(),
        'Top5-Acc': nn.Top5CategoricalAccuracy()
    })
    # top1_acc = nn.Top1CategoricalAccuracy()
    ds_eval = create_dataset(data_path=eval_dataset_path, do_train=False, repeat_num=1, batch_size=eval_batch_size,
                             rank_id=0,
                             rank_size=1, image_size=train_image_size, scale_size=eval_image_size)
    print('dataset size is : \n', ds_eval.get_dataset_size())
    # for data in ds_eval.create_dict_iterator():
    #     logits = regnet(data['image'])
    #     print(logits)
    #     print(data['label'])
    #     top1_acc.update(logits, data['label'])
    # acc = top1_acc.eval()
    acc = model.eval(ds_eval, dataset_sink_mode=True)
    print(acc)


if __name__ == '__main__':
    args = cfg['MODEL']
    model_args = {}
    for i in args:
        model_args[i.lower()] = args[i]
    evaluate(model_args)

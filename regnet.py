import numpy as np
import blocks as bk
from config import cfg
from anynet import AnyNet
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore.communication import get_group_size
from mindspore.parallel._auto_parallel_context import auto_parallel_context


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per stage widths and depths from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    # Generate continuous per-block ws
    ws_cont = np.arange(d) * w_a + w_0
    # Generate quantized per-block ws
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws_all = w_0 * np.power(w_m, ks)
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
    # Generate per stage ws and ds (assumes ws_all are sorted)
    ws, ds = np.unique(ws_all, return_counts=True)
    # Compute number of actual stages and total possible stages
    num_stages, total_stages = len(ws), ks.max() + 1
    # Convert numpy arrays to lists and return
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont


def generate_regnet_full():
    """Generates per stage ws, ds, gs, bs, and ss from RegNet cfg."""
    w_a, w_0, w_m, d = cfg.REGNET.WA, cfg.REGNET.W0, cfg.REGNET.WM, cfg.REGNET.DEPTH
    ws, ds = generate_regnet(w_a, w_0, w_m, d)[0:2]
    ss = [cfg.REGNET.STRIDE for _ in ws]
    bs = [cfg.REGNET.BOT_MUL for _ in ws]
    gs = [cfg.REGNET.GROUP_W for _ in ws]
    ws, bs, gs = bk.adjust_block_compatibility(ws, bs, gs)
    return ws, ds, ss, bs, gs


class RegNet(AnyNet):
    """RegNet model."""

    @staticmethod
    def get_params():
        """Get AnyNet parameters that correspond to the RegNet."""
        ws, ds, ss, bs, gs = generate_regnet_full()
        return {
            "stem_type": cfg.REGNET.STEM_TYPE,
            "stem_w": cfg.REGNET.STEM_W,
            "block_type": cfg.REGNET.BLOCK_TYPE,
            "depths": ds,
            "widths": ws,
            "strides": ss,
            "bot_muls": bs,
            "group_ws": gs,
            "head_w": cfg.REGNET.HEAD_W,
            "se_r": cfg.REGNET.SE_R if cfg.REGNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self):
        params = RegNet.get_params()
        super(RegNet, self).__init__(params)

    def construct(self, x):
        x = self.stem(x)
        for module in self.stages:
            x = module(x)
        x = self.head(x)
        return x

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        params = RegNet.get_params() if not params else params
        return AnyNet.complexity(cx, params)


class RegNetWithLossCell(nn.Cell):
    """RegNet with loss cell."""

    def __init__(self, network, loss):
        super(RegNetWithLossCell, self).__init__()
        self.network = network
        self.loss = loss
        self.ce_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    def construct(self, img, label):
        """Forward pass."""
        logits = self.network(img)
        return self.ce_loss(logits, label)


class TrainingWrapper(nn.Cell):
    """Wrap for training."""

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        class_list = [ms.ParallelMode.DATA_PARALLEL, ms.ParallelMode.HYBRID_PARALLEL]
        if self.parallel_mode in class_list:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = ms.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = ms.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *args):
        """Forward pass."""
        weights = self.weights
        loss = self.network(*args)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss

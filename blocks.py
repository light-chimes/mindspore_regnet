import mindspore.nn as nn
import mindspore.common.initializer as init
from mindspore import dtype as mstype
import mindspore.ops as ops
from config import cfg
import numpy as np


def conv2d(w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Helper for building a conv2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    s, p, g, b = stride, (k - 1) // 2, groups, bias
    # return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, group=g, has_bias=b, pad_mode='pad')
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, group=g, has_bias=b, pad_mode='pad',
                     weight_init=init.initializer(init.HeNormal(mode='fan_out', nonlinearity='relu'),
                                                  [w_out, w_in // g, k, k], mstype.float32))
    # return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, group=g, has_bias=b, pad_mode='pad',
    #                  weight_init=init.HeNormal(mode='fan_out', nonlinearity='relu'))


def norm2d(w_in):
    """Helper for building a norm2d layer."""
    return nn.BatchNorm2d(num_features=w_in, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)


def pool2d(_w_in, k, *, stride=1):
    """Helper for building a pool2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    padding = (k - 1) // 2
    pad2d = nn.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="CONSTANT")
    max_pool = nn.MaxPool2d(kernel_size=k, stride=stride, pad_mode="valid")
    return nn.SequentialCell([pad2d, max_pool])


def gap2d(_w_in):
    """Helper for building a gap2d layer."""
    # ops on GPU
    # return ops.AdaptiveAvgPool2D((1, 1))
    # CSDN implement
    # return AdaptiveAvgPool2d((1, 1))
    # other research implement
    return AdaptiveAvgPool2d()


def linear(w_in, w_out, *, bias=False):
    """Helper for building a linear layer."""
    # return nn.Dense(w_in, w_out, has_bias=bias, weight_init='Uniform', bias_init='Uniform')
    return nn.Dense(w_in, w_out, has_bias=bias, weight_init=init.Normal(sigma=0.01, mean=0.0),
                    bias_init='zeros')


def activation(activation_fun=None):
    """Helper for building an activation layer."""
    activation_fun = (activation_fun or cfg.MODEL.ACTIVATION_FUN).lower()
    if activation_fun == "relu":
        return nn.ReLU()
    else:
        raise AssertionError("Unknown MODEL.ACTIVATION_FUN: " + activation_fun)


# --------------------------- Complexity (cx) calculations --------------------------- #


def conv2d_cx(cx, w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Accumulates complexity of conv2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    flops += k * k * w_in * w_out * h * w // groups + (w_out * h * w if bias else 0)
    params += k * k * w_in * w_out // groups + (w_out if bias else 0)
    acts += w_out * h * w
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def norm2d_cx(cx, w_in):
    """Accumulates complexity of norm2d into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    params += 2 * w_in
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def pool2d_cx(cx, w_in, k, *, stride=1):
    """Accumulates complexity of pool2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    acts += w_in * h * w
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def gap2d_cx(cx, _w_in):
    """Accumulates complexity of gap2d into cx = (h, w, flops, params, acts)."""
    flops, params, acts = cx["flops"], cx["params"], cx["acts"]
    return {"h": 1, "w": 1, "flops": flops, "params": params, "acts": acts}


def linear_cx(cx, w_in, w_out, *, bias=False, num_locations=1):
    """Accumulates complexity of linear into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    flops += w_in * w_out * num_locations + (w_out * num_locations if bias else 0)
    params += w_in * w_out + (w_out if bias else 0)
    acts += w_out * num_locations
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


# ---------------------------------- Shared blocks ----------------------------------- #


class SE(nn.Cell):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = gap2d(w_in)
        self.f_ex = nn.SequentialCell(
            conv2d(w_in, w_se, 1, bias=True),
            activation(),
            conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def construct(self, x):
        return x * self.f_ex(self.avg_pool(x))

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx = gap2d_cx(cx, w_in)
        cx = conv2d_cx(cx, w_in, w_se, 1, bias=True)
        cx = conv2d_cx(cx, w_se, w_in, 1, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


# ---------------------------------- Miscellaneous ----------------------------------- #


def adjust_block_compatibility(ws, bs, gs):
    """Adjusts the compatibility of widths, bottlenecks, and groups."""
    assert len(ws) == len(bs) == len(gs)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
    assert all(b < 1 or b % 1 == 0 for b in bs)
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, int(b)) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
    return ws, bs, gs


# class AdaptiveAvgPool2d(nn.Cell):
#     def __init__(self, output_size):
#         """Initialize AdaptiveAvgPool2d."""
#         super(AdaptiveAvgPool2d, self).__init__()
#         self.output_size = output_size
#
#     def adaptive_avgpool2d(self, inputs):
#         """ NCHW """
#         H = self.output_size[0]
#         W = self.output_size[1]
#
#         H_start = ops.Cast()(Tensor(np.arange(start=0, stop=H, dtype=np.float32) * (inputs.shape[-2] / H)),
#                              mstype.int64)
#         H_end = ops.Cast()(
#             Tensor(np.ceil(((np.arange(start=0, stop=H, dtype=np.float32) + 1) * (inputs.shape[-2] / H)))),
#             mstype.int64)
#
#         W_start = ops.Cast()(Tensor(np.arange(start=0, stop=W, dtype=np.float32) * (inputs.shape[-1] / W)),
#                              mstype.int64)
#         W_end = ops.Cast()(
#             Tensor(np.ceil(((np.arange(start=0, stop=W, dtype=np.float32) + 1) * (inputs.shape[-1] / W)))),
#             mstype.int64)
#
#         pooled2 = []
#         for idx_H in range(H):
#             pooled1 = []
#             for idx_W in range(W):
#                 h_s = int(H_start[idx_H].asnumpy())
#                 h_e = int(H_end[idx_H].asnumpy())
#                 w_s = int(W_start[idx_W].asnumpy())
#                 w_e = int(W_end[idx_W].asnumpy())
#                 res = inputs[:, :, h_s:h_e, w_s:w_e]
#                 # res = inputs[:, :, H_start[idx_H]:H_end[idx_H], W_start[idx_W]:W_end[idx_W]]  # 这样写mindspore tensor切片报类型错误，不知道为啥
#                 pooled1.append(ops.ReduceMean(keep_dims=True)(res, (-2, -1)))
#             pooled1 = ops.Concat(-1)(pooled1)
#             pooled2.append(pooled1)
#         pooled2 = ops.Concat(-2)(pooled2)
#
#         return pooled2
#
#     def construct(self, x):
#         x = self.adaptive_avgpool2d(x)
#         return x


class AdaptiveAvgPool2d(nn.Cell):

    def __init__(self):
        super().__init__()
        self.ReduceMean = ops.ReduceMean(keep_dims=True)

    def construct(self, x):
        return self.ReduceMean(x, (-1, -2))

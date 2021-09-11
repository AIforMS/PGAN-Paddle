# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import paddle
import paddle.nn as nn


# set device
paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')


class ConstantNet(nn.Layer):
    r"A network that does nothing"

    def __init__(self,
                 shapeOut=None):

        super(ConstantNet, self).__init__()
        self.shapeOut = shapeOut

    def forward(self, x):

        if self.shapeOut is not None:
            x = x.reshape((x.size[0], self.shapeOut[0],
                       self.shapeOut[1], self.shapeOut[2]))

        return x


class MeanStd(nn.Layer):
    def __init__(self):
        super(MeanStd, self).__init__()

    def forward(self,x):

        # Size : N C W H
        x = x.reshape((x.shape[0], x.shape[1], -1))
        mean_x = paddle.mean(x, axis=2)
        var_x = paddle.mean(x**2, axis=2) - mean_x * mean_x
        return paddle.concat([mean_x, var_x], axis=1)


class FeatureTransform(nn.Layer):
    r"""
    Concatenation of a resize tranform and a normalization
    """

    def __init__(self,
                 mean=None,
                 std=None,
                 size=224):

        super(FeatureTransform, self).__init__()
        self.size = size

        if mean is None:
            mean = [0., 0., 0.]

        if std is None:
            std = [1., 1., 1.]

        self.register_buffer('mean', paddle.to_tensor(
            mean, dtype=paddle.float32).reshape((1, 3, 1, 1)))
        self.register_buffer('std', paddle.to_tensor(
            std, dtype=paddle.float32).reshape((1, 3, 1, 1)))

        if size is None:
            self.upsamplingModule = None
        else:
            self.upsamplingModule = nn.Upsample(
                (size, size), mode='bilinear')

    def forward(self, x):

        if self.upsamplingModule is not None:
            x = self.upsamplingModule(x)

        x = x - self.mean
        x = x / self.std

        return x

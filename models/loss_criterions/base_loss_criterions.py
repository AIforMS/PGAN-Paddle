#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import paddle
import paddle.nn.functional as F


class BaseLossWrapper:
    r"""
    Loss criterion class. Must define 4 members:
    sizeDecisionLayer : size of the decision layer of the discrimator

    getCriterion : how the loss is actually computed

    !! The activation function of the discriminator is computed within the
    loss !!
    """

    def __init__(self, device):
        self.device = device

    def getCriterion(self, input, status):
        r"""
        Given an input tensor and its targeted status (detected as real or
        detected as fake) build the associated loss

        Args:

            - input (Tensor): decision tensor build by the model's discrimator
            - status (bool): if True -> this tensor should have been detected
                             as a real input
                             else -> it shouldn't have
        """
        pass


class MSE(BaseLossWrapper):
    r"""
    Mean Square error loss.
    """

    def __init__(self, device):
        self.generationActivation = F.tanh
        self.sizeDecisionLayer = 1

        BaseLossWrapper.__init__(self, device)

    def getCriterion(self, input, status):
        size = input.shape[0]
        value = float(status)
        reference = paddle.to_tensor([value]).expand((size, 1))
        return F.mse_loss(F.sigmoid(input[:, :self.sizeDecisionLayer]),
                          reference)


class WGANGP(BaseLossWrapper):
    r"""
    Paper WGANGP loss : linear activation for the generator.
    https://arxiv.org/pdf/1704.00028.pdf
    """

    def __init__(self, device):

        self.generationActivation = None
        self.sizeDecisionLayer = 1

        BaseLossWrapper.__init__(self, device)

    def getCriterion(self, input, status):
        if status:
            return -input[:, 0].sum()
        return input[:, 0].sum()


class Logistic(BaseLossWrapper):
    r"""
    "Which training method of GANs actually converge"
    https://arxiv.org/pdf/1801.04406.pdf
    """

    def __init__(self, device):

        self.generationActivation = None
        self.sizeDecisionLayer = 1
        BaseLossWrapper.__init__(self, device)

    def getCriterion(self, input, status):
        if status:
            return F.softplus(-input[:, 0]).mean()
        return F.softplus(input[:, 0]).mean()


class DCGAN(BaseLossWrapper):
    r"""
    Cross entropy loss.
    """

    def __init__(self, device):

        self.generationActivation = F.tanh
        self.sizeDecisionLayer = 1

        BaseLossWrapper.__init__(self, device)

    def getCriterion(self, input, status):
        size = input.shape[0]
        value = int(status)
        reference = paddle.to_tensor(
            [value], dtype=paddle.float32).expand((size))
        return F.binary_cross_entropy(F.sigmoid(input[:, :self.sizeDecisionLayer]), reference)

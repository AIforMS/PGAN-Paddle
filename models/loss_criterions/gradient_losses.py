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


def WGANGPGradientPenalty(input, fake, discriminator, weight, backward=True):
    r"""
    Gradient penalty as described in
    "Improved Training of Wasserstein GANs"
    https://arxiv.org/pdf/1704.00028.pdf

    Args:

        - input (Tensor): batch of real data
        - fake (Tensor): batch of generated data. Must have the same size
          as the input
        - discrimator (nn.Module): discriminator network
        - weight (float): weight to apply to the penalty term
        - backward (bool): loss backpropagation
    """

    batchSize = input.shape[0]
    alpha = paddle.rand((batchSize, 1))
    alpha = alpha.expand((batchSize, int(input.numel() /
                                         batchSize))).reshape(input.shape)
    interpolates = alpha * input + ((1 - alpha) * fake)

    # interpolates = torch.autograd.Variable(
    #     interpolates, requires_grad=True)
    interpolates = paddle.to_tensor(interpolates, stop_gradient=False)

    decisionInterpolate = discriminator(interpolates, False)
    decisionInterpolate = decisionInterpolate[:, 0].sum()

    gradients = paddle.autograd.grad(outputs=decisionInterpolate,
                                     inputs=interpolates,
                                     create_graph=True, retain_graph=True)
    # gradients = torch.autograd.grad(outputs=decisionInterpolate,
    #                                 inputs=interpolates,
    #                                 create_graph=True, retain_graph=True)

    gradients = gradients[0].reshape((batchSize, -1))
    gradients = (gradients * gradients).sum(axis=1).sqrt()
    gradient_penalty = (((gradients - 1.0) ** 2)).sum() * weight

    if backward:
        gradient_penalty.backward(retain_graph=True)

    return gradient_penalty.item()


def logisticGradientPenalty(input, discrimator, weight, backward=True):
    r"""
    Gradient penalty described in "Which training method of GANs actually
    converge
    https://arxiv.org/pdf/1801.04406.pdf

    Args:

        - input (Tensor): batch of real data
        - discrimator (nn.Module): discriminator network
        - weight (float): weight to apply to the penalty term
        - backward (bool): loss backpropagation
    """

    # locInput = torch.autograd.Variable(
    #     input, requires_grad=True)
    locInput = paddle.to_tensor(input, stop_gradient=False)
    gradients = paddle.autograd.grad(outputs=discrimator(locInput)[:, 0].sum(),
                                     inputs=locInput,
                                     create_graph=True, retain_graph=True)[0]

    gradients = gradients.reshape((gradients.shape[0], -1))
    gradients = (gradients * gradients).sum(axis=1).mean()

    gradient_penalty = gradients * weight
    if backward:
        gradient_penalty.backward(retain_graph=True)

    return gradient_penalty.item()

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

import paddle.optimizer as optim

from .base_GAN import BaseGAN
from .utils.config import BaseConfig
from .networks.progressive_conv_net import GNet, DNet


class ProgressiveGAN(BaseGAN):
    r"""
    Implementation of NVIDIA's progressive GAN.
    """

    def __init__(self,
                 dimLatentVector=512,
                 depthScale0=512,
                 initBiasToZero=True,
                 leakyness=0.2,
                 perChannelNormalization=True,
                 miniBatchStdDev=False,
                 equalizedlR=True,
                 **kwargs):
        r"""
        Args:

        Specific Arguments:
            - depthScale0 (int)
            - initBiasToZero (bool): should layer's bias be initialized to
                                     zero ?
            - leakyness (float): negative slope of the leakyRelU activation
                                 function
            - perChannelNormalization (bool): do we normalize the output of
                                              each convolutional layer ?
            - miniBatchStdDev (bool): mini batch regularization for the
                                      discriminator
            - equalizedlR (bool): if True, forces the optimizer to see weights
                                  in range (-1, 1)

        """
        if not 'config' in vars(self):
            self.config = BaseConfig()

        self.config.depthScale0 = depthScale0
        self.config.initBiasToZero = initBiasToZero
        self.config.leakyReluLeak = leakyness
        self.config.depthOtherScales = []
        self.config.perChannelNormalization = perChannelNormalization
        self.config.alpha = 0
        self.config.miniBatchStdDev = miniBatchStdDev
        self.config.equalizedlR = equalizedlR

        BaseGAN.__init__(self, dimLatentVector, **kwargs)

    def getNetG(self):

        gnet = GNet(self.config.latentVectorDim,
                    self.config.depthScale0,
                    initBiasToZero=self.config.initBiasToZero,
                    leakyReluLeak=self.config.leakyReluLeak,
                    normalization=self.config.perChannelNormalization,
                    generationActivation=self.lossCriterion.generationActivation,
                    dimOutput=self.config.dimOutput,
                    equalizedlR=self.config.equalizedlR)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            gnet.addScale(depth)

        # If new scales are added, give the generator a blending layer
        if self.config.depthOtherScales:
            gnet.setNewAlpha(self.config.alpha)

        return gnet

    def getNetD(self):

        dnet = DNet(self.config.depthScale0,
                    initBiasToZero=self.config.initBiasToZero,
                    leakyReluLeak=self.config.leakyReluLeak,
                    sizeDecisionLayer=self.lossCriterion.sizeDecisionLayer +
                    self.config.categoryVectorDim,
                    miniBatchNormalization=self.config.miniBatchStdDev,
                    dimInput=self.config.dimOutput,
                    equalizedlR=self.config.equalizedlR)

        # Add scales if necessary
        for depth in self.config.depthOtherScales:
            dnet.addScale(depth)

        # If new scales are added, give the discriminator a blending layer
        if self.config.depthOtherScales:
            dnet.setNewAlpha(self.config.alpha)

        return dnet

    def getOptimizerD(self):
        # param_trainable = []
        # for param in self.netD.parameters():
        #     if param.trainable:
        #         param_trainable += list(param)
        # return optim.Adam(parameters=param_trainable, beta1=0., beta2=0.99, learning_rate=self.config.learningRate)

        # 在 400 个 step 后 loss 不下降，则学习率降为当前的 0.5倍，最小降到 0.0003
        lr_scheduler = optim.lr.ReduceOnPlateau(learning_rate=self.config.learningRate, mode='min',
                                                factor=0.5, patience=400, min_lr=0.0003, verbose=True)
        return optim.Adam(parameters=filter(lambda p: p.trainable, self.netD.parameters()),
                          beta1=0., beta2=0.99, learning_rate=lr_scheduler)

    def getOptimizerG(self):
        # param_trainable = []
        # for param in self.netG.parameters():
        #     if param.trainable:
        #         param_trainable += list(param)
        # return optim.Adam(parameters=param_trainable, beta1=0., beta2=0.99, learning_rate=self.config.learningRate)

        # 在 400 个 step 后 loss 不下降，则学习率降为当前的 0.5倍，最小降到 0.0003
        lr_scheduler = optim.lr.ReduceOnPlateau(learning_rate=self.config.learningRate, mode='min',
                                                factor=0.5, patience=400, min_lr=0.0003, verbose=True)
        return optim.Adam(parameters=filter(lambda p: p.trainable, self.netG.parameters()),
                          beta1=0., beta2=0.99, learning_rate=lr_scheduler)

    def addScale(self, depthNewScale):
        r"""
        Add a new scale to the model. The output resolution becomes twice
        bigger.
        """
        self.netG = self.getOriginalG()
        self.netD = self.getOriginalD()

        self.netG.addScale(depthNewScale)
        self.netD.addScale(depthNewScale)

        self.config.depthOtherScales.append(depthNewScale)

        self.updateSolversDevice()

    def updateAlpha(self, newAlpha):
        r"""
        Update the blending factor alpha.

        Args:
            - alpha (float): blending factor (in [0,1]). 0 means only the
                             highest resolution in considered (no blend), 1
                             means the highest resolution is fully discarded.
        """
        print("Changing alpha to %.3f" % newAlpha)

        self.getOriginalG().setNewAlpha(newAlpha)
        self.getOriginalD().setNewAlpha(newAlpha)

        if self.avgG:
            self.avgG.setNewAlpha(newAlpha)

        self.config.alpha = newAlpha

    def getSize(self):
        r"""
        Get output image size (W, H)
        """
        return self.getOriginalG().getOutputSize()

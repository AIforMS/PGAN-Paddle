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

from ..utils.utils import loadmodule
from ..networks.constant_net import FeatureTransform


def extractRelUIndexes(sequence, layers):
    layers.sort()

    index = 0
    output = []

    indexRef = 0
    indexScale = 1

    hasCaughtRelUOnLayer = False
    while indexRef < len(layers) and index < len(sequence):

        if isinstance(sequence[index], paddle.nn.ReLU):

            if not hasCaughtRelUOnLayer and indexScale == layers[indexRef]:
                hasCaughtRelUOnLayer = True
                output.append(index)
                indexRef += 1

        if isinstance(sequence[index], paddle.nn.MaxPool2D) \
                or isinstance(sequence[index], paddle.nn.AvgPool2D):
            hasCaughtRelUOnLayer = False
            indexScale += 1

        index += 1

    return output


def extractIndexedLayers(sequence,
                         x,
                         indexes,
                         detach):
    index = 0
    output = []

    indexes.sort()

    for iSeq, layer in enumerate(sequence):

        if index >= len(indexes):
            break

        x = layer(x)

        if iSeq == indexes[index]:
            if detach:
                output.append(x.reshape((x.shape[0], x.shape[1], -1)).detach())
            else:
                output.append(x.reshape((x.shape[0], x.shape[1], -1)))
            index += 1

    return output


class LossTexture(paddle.nn.Layer):
    r"""
    An implenetation of style transfer's (http://arxiv.org/abs/1703.06868) like
    loss.
    """

    def __init__(self,
                 device,
                 modelName,
                 scalesOut):
        r"""
        Args:
            - device (torch.device): torch.device("cpu") or
                                     torch.device("cuda:0")
            - modelName (string): name of the torchvision.models model. For
                                  example vgg19
            - scalesOut (list): index of the scales to extract. In the Style
                                transfer paper it was [1,2,3,4]
        """

        super(LossTexture, self).__init__()
        scalesOut.sort()

        model = loadmodule("paddle.vision.models", modelName, prefix='')
        self.featuresSeq = model(pretrained=True).features
        self.indexLayers = extractRelUIndexes(self.featuresSeq, scalesOut)

        self.reductionFactor = [1 / float(2 ** (i - 1)) for i in scalesOut]

        refMean = [2 * p - 1 for p in [0.485, 0.456, 0.406]]
        refSTD = [2 * p for p in [0.229, 0.224, 0.225]]

        self.imgTransform = FeatureTransform(mean=refMean,
                                             std=refSTD,
                                             size=None)

        # self.imgTransform = self.imgTransform

    def getLoss(self, fake, reals, mask=None):

        featuresReals = self.getFeatures(
            reals, detach=True, prepImg=True, mask=mask).mean(axis=0)
        featuresFakes = self.getFeatures(
            fake, detach=False, prepImg=True, mask=None).mean(axis=0)

        outLoss = ((featuresReals - featuresFakes) ** 2).mean()
        return outLoss

    def getFeatures(self, image, detach=True, prepImg=True, mask=None):

        if prepImg:
            image = self.imgTransform(image)

        fullSequence = extractIndexedLayers(self.featuresSeq,
                                            image,
                                            self.indexLayers,
                                            detach)
        outFeatures = []
        nFeatures = len(fullSequence)

        for i in range(nFeatures):

            if mask is not None:
                locMask = (1. + F.upsample(mask,
                                           size=(image.shape[2] * self.reductionFactor[i],
                                                 image.shape[3] * self.reductionFactor[i]),
                                           mode='bilinear')) * 0.5
                locMask = locMask.reshape((locMask.shape[0], locMask.shape[1], -1))

                totVal = locMask.sum(axis=2)

                meanReals = (fullSequence[i] * locMask).sum(axis=2) / totVal
                varReals = (
                                   (fullSequence[i] * fullSequence[i] * locMask).sum(
                                       aixs=2) / totVal) - meanReals * meanReals

            else:
                meanReals = fullSequence[i].mean(axis=2)
                varReals = (
                               (fullSequence[i] * fullSequence[i]).mean(axis=2)) \
                           - meanReals * meanReals

            outFeatures.append(meanReals)
            outFeatures.append(varReals)

        return paddle.concat(outFeatures, axis=1)

    def forward(self, x, mask=None):

        return self.getFeatures(x, detach=False, prepImg=False, mask=mask)

    def saveModel(self, pathOut):

        paddle.save(dict(model=self, fullDump=True,
                         mean=self.imgTransform.mean.flatten().tolist(),
                         std=self.imgTransform.std.flatten().tolist()),
                    pathOut)

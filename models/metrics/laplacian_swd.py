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
import paddle.nn as nn

import numpy as np
from ..utils.utils import printProgressBar

# set device
paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')


def getDescriptorsForMinibatch(minibatch, patchSize, nPatches):
    r"""
    Extract @param nPatches randomly chosen of size patchSize x patchSize
    from each image of the input @param minibatch

    Returns:

        A tensor of SxCxpatchSizexpatchSize where
        S = minibatch.shape[0] * nPatches is the total number of patches
        extracted from the minibatch.
    """
    S = minibatch.shape

    maxX = S[2] - patchSize
    maxY = S[3] - patchSize

    baseX = paddle.arange(0, patchSize, dtype=paddle.int64).expand((S[0] * nPatches,
                                                                patchSize)) \
        + paddle.randint(0, maxX, (S[0] * nPatches, 1), dtype=paddle.int64)
    baseY = paddle.arange(0, patchSize, dtype=paddle.int64).expand((S[0] * nPatches,
                                                                patchSize)) \
        + paddle.randint(0, maxY, (S[0] * nPatches, 1), dtype=paddle.int64)

    baseX = baseX.reshape((S[0], nPatches, 1, patchSize)).expand((
        S[0], nPatches, patchSize, patchSize))
    baseY = S[2] * baseY.reshape((S[0], nPatches, patchSize, 1))
    baseY = baseY.expand((S[0], nPatches, patchSize, patchSize))

    coords = baseX + baseY
    coords = coords.reshape((S[0], nPatches, 1, patchSize, patchSize)).expand((
        S[0], nPatches, S[1], patchSize, patchSize))
    C = paddle.arange(0, S[1], dtype=paddle.int64).reshape((
        1, S[1])).expand((nPatches * S[0], S[1]))*S[2]*S[3]
    coords = C.reshape((S[0], nPatches, S[1], 1, 1)) + coords
    coords = coords.flatten()

    return (minibatch.flatten()[coords]).reshape((-1, S[1], patchSize, patchSize))


def getMeanStdDesc(desc):
    r"""
    Get the mean and the standard deviation of each channel accross the input
    batch.
    """
    S = desc.shape
    assert len(S) == 4
    mean = paddle.sum(desc.reshape((S[0], S[1], -1)),
                     axis=2).sum(axis=0) / (S[0] * S[3] * S[2])
    var = paddle.sum(
        (desc*desc).reshape((S[0], S[1], -1)), axis=2).sum(axis=0) / \
        (S[0] * S[3] * S[2])
    var -= mean*mean
    var = var.clip(min=0).sqrt().reshape((1, S[1])).expand((S[0], S[1]))
    mean = (mean.reshape((1, S[1]))).expand((S[0], S[1]))

    return mean.reshape((S[0], S[1], 1, 1)), var.reshape((S[0], S[1], 1, 1))


# -------------------------------------------------------------------------------
# Laplacian pyramid generation, with LaplacianSWDMetric.convolution as input,
# matches the corresponding openCV functions
# -------------------------------------------------------------------------------

def pyrDown(minibatch, convolution):
    x = nn.Pad2D(2, mode='reflect')(minibatch)
    return convolution(x)[:, :, ::2, ::2].detach()


def pyrUp(minibatch, convolution):
    S = minibatch.shape
    res = paddle.zeros((S[0], S[1], S[2] * 2, S[3] * 2),
                      dtype=minibatch.dtype)
    res[:, :, ::2, ::2] = minibatch
    res = nn.Pad2D(2, mode='reflect')(res)
    return 4 * convolution(res).detach()

# ----------------------------------------------------------------------------


def sliced_wasserstein(A, B, dir_repeats, dirs_per_repeat):
    r"""
    NVIDIA's approximation of the SWD distance.
    """
    # (neighborhood, descriptor_component)
    assert A.ndim == 2 and A.shape == B.shape
    results = []
    for repeat in range(dir_repeats):
        # (descriptor_component, direction)
        dirs = np.random.randn(A.shape[1], dirs_per_repeat)
        # normalize descriptor components for each direction
        dirs /= np.sqrt(np.sum(np.square(dirs), axis=0, keepdims=True))
        dirs = dirs.astype(np.float32)
        # (neighborhood, direction)
        projA = np.matmul(A, dirs)
        projB = np.matmul(B, dirs)
        # sort neighborhood projections for each direction
        projA = np.sort(projA, axis=0)
        projB = np.sort(projB, axis=0)
        # pointwise wasserstein distances
        dists = np.abs(projA - projB)
        # average over neighborhoods and directions
        results.append(np.mean(dists))
    return np.mean(results).item()


def sliced_wasserstein_torch(A, B, dir_repeats, dirs_per_repeat):
    r"""
    NVIDIA's approximation of the SWD distance.
    """
    results = []
    for repeat in range(dir_repeats):
        # (descriptor_component, direction)
        dirs = paddle.randn((A.shape[1], dirs_per_repeat), dtype=paddle.float32)
        # normalize descriptor components for each direction
        dirs /= paddle.sqrt(paddle.sum(dirs*dirs, 0, keepdim=True))
        # (neighborhood, direction)
        projA = paddle.matmul(A, dirs)
        projB = paddle.matmul(B, dirs)
        # sort neighborhood projections for each direction
        projA = paddle.sort(projA, axis=0)[0]
        projB = paddle.sort(projB, axis=0)[0]
        # pointwise wasserstein distances
        dists = paddle.abs(projA - projB)
        # average over neighborhoods and directions
        results.append(paddle.mean(dists).item())
    return sum(results) / float(len(results))


def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 4  # (neighborhood, channel, height, width)
    desc -= np.mean(desc, axis=(0, 2, 3), keepdims=True)
    desc /= np.std(desc, axis=(0, 2, 3), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc


class LaplacianSWDMetric:
    r"""
    SWD metrics used on patches extracted from laplacian pyramids of the input
    images.
    """

    def __init__(self,
                 patchSize,
                 nDescriptorLevel,
                 depthPyramid):
        r"""
        Args:
            patchSize (int): side length of each patch to extract
            nDescriptorLevel (int): number of patches to extract at each level
                                    of the pyramid
            depthPyramid (int): depth of the laplacian pyramid
        """
        self.patchSize = patchSize
        self.nDescriptorLevel = nDescriptorLevel
        self.depthPyramid = depthPyramid

        self.descriptorsRef = [[] for x in range(depthPyramid)]
        self.descriptorsTarget = [[] for x in range(depthPyramid)]

        self.convolution = None

    def updateWithMiniBatch(self, ref, target):
        r"""
        Extract and store decsriptors from the current minibatch
        Args:
            ref (tensor): reference data.
            target (tensor): target data.

            Both tensor must have the same format: NxCxWxD
            N: minibatch size
            C: number of channels
            W: with
            H: height
        """
        modes = [(ref, self.descriptorsRef), (target, self.descriptorsTarget)]

        assert(ref.shape == target.shape)

        if not self.convolution:
            self.initConvolution(paddle.get_device())  # ref.device

        for item, dest in modes:
            pyramid = self.generateLaplacianPyramid(item, self.depthPyramid)
            for scale in range(self.depthPyramid):
                dest[scale].append(getDescriptorsForMinibatch(pyramid[scale],
                                                              self.patchSize,
                                                              self.nDescriptorLevel).numpy())

    def getScore(self):
        r"""
        Output the SWD distance between both distributions using the stored
        descriptors.
        """
        output = []

        descTarget = [finalize_descriptors(d) for d in self.descriptorsTarget]
        del self.descriptorsTarget

        descRef = [finalize_descriptors(d) for d in self.descriptorsRef]
        del self.descriptorsRef

        for scale in range(self.depthPyramid):
            printProgressBar(scale, self.depthPyramid)
            distance = sliced_wasserstein(
                descTarget[scale], descRef[scale], 4, 128)
            output.append(distance)
        printProgressBar(self.depthPyramid, self.depthPyramid)

        del descRef, descTarget

        return output

    def generateLaplacianPyramid(self, minibatch, num_levels):
        r"""
        Build the laplacian pyramids corresponding to the current minibatch.
        Args:
            minibatch (tensor): NxCxWxD, input batch
            num_levels (int): number of levels of the pyramids
        """
        pyramid = [minibatch]
        for i in range(1, num_levels):
            pyramid.append(pyrDown(pyramid[-1], self.convolution))
            pyramid[-2] -= pyrUp(pyramid[-1], self.convolution)
        return pyramid

    def reconstructLaplacianPyramid(self, pyramid):
        r"""
        Given a laplacian pyramid, reconstruct the corresponding minibatch

        Returns:
            A list L of tensors NxCxWxD, where L[i] represents the pyramids of
            the batch for the ith scale
        """
        minibatch = pyramid[-1]
        for level in pyramid[-2::-1]:
            minibatch = pyrUp(minibatch, self.convolution) + level
        return minibatch

    def initConvolution(self, device):
        r"""
        Initialize the convolution used in openCV.pyrDown() and .pyrUp()
        """
        gaussianFilter = paddle.to_tensor([
            [1, 4,  6,  4,  1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4,  6,  4,  1]], dtype=paddle.float32) / 256.0

        self.convolution = nn.Conv2D(3, 3, (5, 5))

        np_w = np.zeros((3, 3, 5, 5))
        np_w[0][0] = gaussianFilter
        np_w[1][1] = gaussianFilter
        np_w[2][2] = gaussianFilter
        self.convolution.weight.set_value(paddle.to_tensor(np_w, dtype=paddle.float32))
        self.convolution.weight.trainable = False

        # # self.convolution.weight.data.fill_(0)
        # self.convolution.weight.data[0][0] = gaussianFilter
        # self.convolution.weight.data[1][1] = gaussianFilter
        # self.convolution.weight.data[2][2] = gaussianFilter

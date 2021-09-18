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

import numpy as np
import paddle
import paddle.nn.functional as F


def GDPPLoss(phiFake, phiReal, backward=True):
    r"""
    Implementation of the GDPP loss. Can be used with any kind of GAN
    architecture.

    Args:

        phiFake (tensor) : last feature layer of the discriminator on real data
        phiReal (tensor) : last feature layer of the discriminator on fake data
        backward (bool)  : should we perform the backward operation ?

    Returns:

        Loss's value. The backward operation in performed within this operator
    """
    def compute_diversity(phi):
        phi = F.normalize(phi, p=2, axis=1)
        SB = paddle.mm(phi, phi.t())
        # eigVals, eigVecs = torch.symeig(SB, eigenvectors=True)
        eigVals, eigVecs = np.linalg.eigh(SB.numpy())

        return paddle.to_tensor(eigVals), paddle.to_tensor(eigVecs)

    def normalize_min_max(eigVals):
        minV, maxV = paddle.min(eigVals), paddle.max(eigVals)
        if abs(minV.numpy() - maxV.numpy()) < 1e-10:
            return eigVals
        return (eigVals - minV) / (maxV - minV)

    fakeEigVals, fakeEigVecs = compute_diversity(phiFake)
    realEigVals, realEigVecs = compute_diversity(phiReal)

    # Scaling factor to make the two losses operating in comparable ranges.
    magnitudeLoss = 0.0001 * F.mse_loss(label=realEigVals, input=fakeEigVals)
    structureLoss = -paddle.sum(paddle.multiply(fakeEigVecs, realEigVecs), 0)
    normalizedRealEigVals = normalize_min_max(realEigVals)
    weightedStructureLoss = paddle.sum(
        paddle.multiply(normalizedRealEigVals, structureLoss))
    gdppLoss = magnitudeLoss + weightedStructureLoss

    if backward:
        gdppLoss.backward(retain_graph=True)

    return gdppLoss.item()

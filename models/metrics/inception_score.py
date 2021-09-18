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

import math
import paddle
import paddle.nn.functional as F


class InceptionScore():
    def __init__(self, classifier):

        self.sumEntropy = 0
        self.sumSoftMax = None
        self.nItems = 0
        self.classifier = classifier.eval()

    def updateWithMiniBatch(self, ref):
        y = self.classifier(ref).detach()

        if self.sumSoftMax is None:
            self.sumSoftMax = paddle.zeros(paddle.to_tensor(y.shape[1]))

        # Entropy
        x = F.softmax(y, axis=1) * F.log_softmax(y, axis=1)
        self.sumEntropy += x.sum().item()

        # Sum soft max
        self.sumSoftMax += F.softmax(y, axis=1).sum(axis=0)

        # N items
        self.nItems += y.shape[0]

    def getScore(self):

        x = self.sumSoftMax
        x = x * paddle.log(x / self.nItems)
        output = self.sumEntropy - (x.sum().item())
        output /= self.nItems
        return math.exp(output)

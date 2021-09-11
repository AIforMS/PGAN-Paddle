# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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

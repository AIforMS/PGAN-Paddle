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
import argparse
from models.loss_criterions.loss_texture import LossTexture

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('model_name', type=str,
                        choices=["vgg19", "vgg16"],
                        help="""Name of the desured featire extractor:
                        - vgg19, vgg16 : a variation of the style transfer \
                        feature developped in \
                        http://arxiv.org/abs/1703.06868""")
    parser.add_argument('--layers', type=int, nargs='*',
                        help="For vgg models only. Layers to select. \
                        Default ones are 3, 4, 5.", default=None)
    parser.add_argument('output_path', type=str,
                        help="""Path of the output feature extractor""")

    args = parser.parse_args()

    if args.model_name in ["vgg19", "vgg16"]:
        if args.layers is None:
            args.layers = [3, 4, 5]
        featureExtractor = LossTexture(paddle.get_device(),
                                       args.model_name,
                                       args.layers)
        featureExtractor.saveModel(args.output_path)
    else:
        raise AttributeError(args.model_name + " not implemented yet")

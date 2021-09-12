# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import visdom
import paddle
import paddle.vision.transforms as Transforms

import numpy as np
import imageio
import random
from .np_visualizer import make_numpy_grid

vis = visdom.Visdom()


def resizeTensor(data, out_size_image):

    out_data_size = (data.shape[0], data.shape[
                     1], out_size_image[0], out_size_image[1])

    outdata = paddle.empty(out_data_size)
    data = paddle.clip(data, min=-1, max=1)

    interpolationMode = 0
    if out_size_image[0] < data.shape[0] and out_size_image[1] < data.shape[1]:
        interpolationMode = 2

    transform = Transforms.Compose([Transforms.Normalize((-1., -1., -1.), (2, 2, 2)),
                                    Transforms.ToPILImage(),
                                    Transforms.Resize(
                                        out_size_image, interpolation=interpolationMode),
                                    Transforms.ToTensor()])

    for img in range(out_data_size[0]):
        outdata[img] = transform(data[img])

    return outdata


def publishTensors(data, out_size_image, caption="", window_token=None, env="main", nrow=16):
    global vis
    outdata = resizeTensor(data, out_size_image)
    return vis.images(outdata, opts=dict(caption=caption), win=window_token, env=env, nrow=nrow)


def saveTensor(data, out_size_image, path):

    interpolation = 'nearest'
    if isinstance(out_size_image, tuple):
        out_size_image = out_size_image[0]
    data = paddle.clip(data, min=-1, max=1)
    outdata = make_numpy_grid(
        data.numpy(), imgMinSize=out_size_image, interpolation=interpolation)
    imageio.imwrite(path, outdata)


def publishLoss(data, name="", window_tokens=None, env="main"):

    if window_tokens is None:
        window_tokens = {key: None for key in data}

    for key, plot in data.items():

        if key in ("scale", "iter"):
            continue

        nItems = len(plot)
        inputY = np.array([plot[x] for x in range(nItems) if plot[x] is not None])
        inputX = np.array([data["iter"][x] for x in range(nItems) if plot[x] is not None])

        opts = {'title': key + (' scale %d loss over time' % data["scale"]),
                'legend': [key], 'xlabel': 'iteration', 'ylabel': 'loss'}

        window_tokens[key] = vis.line(X=inputX, Y=inputY, opts=opts,
                                      win=window_tokens[key], env=env)

    return window_tokens


def delete_env(name):

    vis.delete_env(name)


def publishScatterPlot(data, name="", window_token=None):
    r"""
    Draws 2D or 3d scatter plots

    Args:

        data (list of tensors): list of Ni x 2 or Ni x 3 tensors. Each tensor
                        representing a cloud of data
        name (string): plot name
        window_token (token): ID of the window the plot should be done

    Returns:

        ID of the window where the data are plotted
    """

    if not isinstance(data, list):
        raise ValueError("Input data should be a list of tensors")

    nPlots = len(data)
    colors = []

    random.seed(None)

    for item in range(nPlots):
        N = data[item].shape[0]
        colors.append(paddle.randint(0, 256, (1, 3)).expand((N, 3)))

    colors = paddle.concat(colors, axis=0).numpy()
    opts = {'markercolor': colors,
            'caption': name}
    activeData = paddle.concat(data, axis=0)

    return vis.scatter(activeData, opts=opts, win=window_token, name=name)

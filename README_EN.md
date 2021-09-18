# Paper reproduction：Progressive Growing of GANs for Improved Quality, Stability, and Variation

[English](./README_EN.md) | [简体中文](./README.md)

- **PGAN-Paddle**
  - [1. Introduction](#1-introduction)
  - [2. Accuracy](#2-accuracy)
  - [3. Dataset](#3-dataset)
  - [4. Environment](#4-environment)
  - [5. Quick start](#5-quick-start)
    - [5.1 train](#51-train)
      - [step1: data preprocess](#step1-data-preprocess)
      - [step2: training](#step2-training)
    - [5.2 evaluation](#52-evaluation)
      - [step1: image generation](#step1-image-generation)
      - [step2: eval metrics](#step2-eval-metrics)
  - [6. Code structure](#6-code-structure)
    - [6.1 structure](#61-structrue)
    - [6.2 Parameter description](#62-parameter-description)
    - [6.3 Training process](#63-training-process)
    - [6.4 Evaluation process](#64-evaluation-process)
  - [7. Reproduction experience](#7-reproduction-experience)
  - [8. Model information](#8-model-information)


## 1. Introduction
This article proposes a new method of training GAN: Gradually increase the convolutional layers of the generator and discriminator during the training process: start with low resolution, and add higher resolution convolutional layers as the training progresses, to model finer details to generate higher resolution and quality images.
![0](https://img-blog.csdnimg.cn/13d251cb1f6441e5b8efb3f963af29d7.jpg)

This method not only speeds up the training speed of GAN, but also increases the stability of training, because the pre-trained low-resolution layer can bring more difficult-to-converge high-resolution layers with hidden codes that are more conducive to training.

This paper also proposes a new metric for evaluating GAN generated images: Sliced ​​Wasserstein Distance (SWD), to evaluate the quality and changes among source images and generated images. 

Paper link：[Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://paperswithcode.com/paper/progressive-growing-of-gans-for-improved)


## 2. Accuracy
Refer to the official open source pytorch version: [https://github.com/facebookresearch/pytorch_GAN_zoo](https://github.com/facebookresearch/pytorch_GAN_zoo)，Based on the paddlepaddle deep learning framework, after reproducing the literature algorithm, the test accuracy achieved by this project is shown in the following table. The highest accuracy of the reference is **CelebA MS-SSIM=0.2838, SWD=2.64(64)**
| Metrics | SWD × $10^3$ | MS-SSIM |
| --- | --- | -- |
| Resolution | 128、64、32、16 | 128 |
| Paddle version accuracy | 4.46、**2.61**、4.98、11.41 | **0.2719** |
| Paper's accuracy | 4.62、**2.64**、3.78、6.06 | **0.2838** |

The hyperparameter configuration is as follows：
> More detail: `PGAN-Paddle/models/trainer/standard_configurations/pgan_config.py`

|hyperparameter|settings| explain |
| --- | --- | --- |
| miniBatchSize | "miniBatchScheduler": {"0": 64, "1": 64, "2": 64, "3": 64, "4": 32, "5": 22}| Mini batch size |
| initBiasToZero | True | Should bias be initialized to zero ? |
|perChannelNormalization | True| Per channel normalization |
| lossMode | WGANGP | loss mode |
| lambdaGP | 10.0 |  Gradient penalty coefficient (WGANGP) |
|leakyness|0.2| Leakyness of the leakyRelU activation function |
| epsilonD| 0.001 | Weight penalty on $D(x)^2$ |
| miniBatchStdDev | True | Mini batch regularization |
| baseLearningRate | 0.001 | Base learning rate| 
| GDPP | False | Activate GDPP loss ?|

## 3. Dataset
This project uses the celeba data set. (CelebA) is a large-scale face attribute data set with more than 200,000 celebrity avatars. The images in this dataset contain a lot of pose changes and background noise and blur.

- Dataset overview：
  - image number：202599 face images
  - image size：178 × 218
  - dataset name：`img_align_celeba`

- Dataset link：[CELEBA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


## 4. Environment
- Hardware：
  - x86 cpu（RAM >= 16 GB）
  - NVIDIA GPU（VRAM >= 32 GB）
  - CUDA + cuDNN
- Framework：
  - paddlepaddle-gpu==0.0.0（nightly build）
- Other dependencies：
  - numpy >= 1.19.2
  - scipy = 1.6.2
  - h5py = 3.2.1
  - imageio = 2.9.0

## 5. Quick start

### 5.1 Train

#### step1: data preprocess

Decompress the downloaded `img_align_celeba.zip` data set before starting training, and then use the `datasets.py` script to preprocess the decompressed data set:

Each image will be cropped to 128×128 resolution.
```
python datasets.py celeba_cropped $PATH_TO_CELEBA/img_align_celeba/ -o $OUTPUT_DATASET
```
After the processing is completed, the configuration file `config_celeba_cropped.json` will be generated in the project root directory and the following content will be automatically written, specifying the preprocessing data set path and the corresponding number of iterations for layer-by-layer training:
```json
{
  "pathDB": "img_dataset/celeba_cropped",
  "config": {
    "maxIterAtScale": [
      48000,
      96000,
      96000,
      96000,
      96000,
      96000
    ]
  }
}
```
You can modify the training configuration in config, such as adjusting the batch_size, it will overwrite the default configuration in the `standard configuration`, the **following is my training configuration**:
```json
{
  "pathDB": "img_dataset/celeba_cropped",
  "config": {
    "miniBatchScheduler": {"0": 64, "1": 64, "2": 64, "3": 64, "4": 32, "5": 22},
    "configScheduler": {
      "0": {"baseLearningRate": 0.003},
      "1": {"baseLearningRate": 0.003},
      "2": {"baseLearningRate": 0.003},
      "3": {"baseLearningRate": 0.003},
      "4": {"baseLearningRate": 0.001},
      "5": {"baseLearningRate": 0.001}
    },
    "maxIterAtScale": [
      48000,
      96000,
      96000,
      96000,
      96000,
      160000
    ]
  }
}
```
> Different batch_size can be set for different scales in `miniBatchScheduler`, because as the scale increases, the batch_size needs to be reduced to prevent the video memory from exploding. Different learning_rate can be set in `configScheduler` for different scales. In the code `PGAN-Paddle/models/progressive_gan.py` I also added an adaptive learning rate decay strategy (lr.ReduceOnPlateau).
    
#### step2: training

Then run the following command to train PGAN from scratch:
```
python train.py PGAN -c config_celeba_cropped.json --restart -n celeba_cropped --np_vis
```
Then wait a few days (I used T4 and Baidu AI studio's V100, and ran for 6 days.).  The trained model at each stage will be dumped into `output_networks/celeba_cropped`. After training, you should get a 128 x 128 resolution generated image.

If the training is interrupted, you can remove `--restart` when restarting the training, and the training will start from the latest model saved in `output_networks/celeba_cropped`. If you want to use GDPP loss, you can add `--GDPP True` to the command line.

The completed training of each stage will be saved in `output_networks/celeba_cropped`:
- model：`celeba_cropped_s$scale_i$iters.pdparams`
- config file：`celeba_cropped_s$scale_i$iters_tmp_config.json`
- refVectors：`celeba_cropped_refVectors.pdparams`
- losses：`celeba_cropped_losses.pkl`
- generated images：`celeba_cropped_s$scale_i$iters_avg.jpg`、`celeba_cropped_s$scale_i$iters.jpg`，`_avg.jpg` image effect is better, and it is used to calculate the metrics by default in evaluation.
![2](https://img-blog.csdnimg.cn/7fe8ba1e0259449ebd00d035819fec49.jpg)

### 5.2 evaluation

**The final trained model can be picked up from Baidu SkyDrive: [celeba_cropped_s5_i96000](https://pan.baidu.com/s/1-wvYpLYiEUGpBi3xT31roA )**, Code: **6nv9**. 

Put the files in `output_networks/celeba_cropped`, specify the path of `refVectors.pdparams` in the `.json` file, `losses.pkl` can be omitted.
> If you need to run the i80000.pdparams model, you can change the file name of the `.json` file to the corresponding i80000, because the code need to find the path of `refVectors.pdparams` through this file.

#### step1: image generation

Use the latest model saved in `output_networks/celeba_cropped` to generate an image with the following command:
```
python eval.py visualization -n celeba_cropped -m PGAN --np_vis
```
If you want to specify the model of a certain stage, add `-s $scale` and `-i $iter`:
```
python eval.py visualization -n celeba_cropped -m PGAN -s $SCALE -i $ITER --np_vis
```
The image generated by the above two commands is saved in `output_networks/celeba_cropped`, named: `celeba_cropped_s$scale_i$iter_fullavg.jpg`

Randomly generate some images:
```
python eval.py visualization -n celeba_cropped -m PGAN --save_dataset $PATH_TO_THE_OUTPUT_DATASET --size_dataset $SIZE_OF_THE_OUTPUT --np_vis
```
Where `$SIZE_OF_THE_OUTPUT` indicates how many images to generate.

#### step2: eval metrics

**SWD & MS-SSIM metric**

run：
```
python eval.py laplacian_SWD -c config_celeba_cropped.json -n celeba_cropped -m PGAN -s 5 -i 64000 --np_vis
```
It will randomly traverse 16000 source images and their generated images in the data path specified in `config_celeba_cropped.json` to calculate the SWD metric. The process of `Merging the results` will take up a lot of CPU memory (about 18 GB) and time. After running, it will output:
```
Running laplacian_SWD
Checkpoint found at scale 5, iter 64000
Average network found !
202599 images found
Generating the fake dataset...
 |####################################################################################################| 100.0% 
 |####################################################################################################| 100.0% 
Merging the results, please wait it can take some time...
 |####################################################################################################| 100.0% 

     resolution               128               64               32  16 (background)
	   score         0.006042         0.002615         0.004997         0.011406 
     ms-ssim score    0.2719      
...OK
```
The corresponding indicator values ​​will be saved in `output_networks/celeba_cropped/celeba_cropped_swd.json`.

## 6. Code structure
### 6.1 structure
```
├── logs                    # Training log file
├── models                    # Including model definition, loss function, data set reading, training and testing methods
│   ├── datasets              # Read dataset
│   ├── eval                  # Use pre-trained models for prediction and index evaluation
│   ├── loss_criterions       # Loss function definition
│   ├── metrics               # Evaluation metrics
│   ├── networks              # Network model definition
│   ├── trainer               # Training strategy package
│   ├── utils                 # Toolkit
│   ├── UTs                   # Unused
│   ├── base_GAN.py           # GAN parent class
│   ├── gan_visualizer.py     # GAN Training intermediate image saving
│   ├── progressive_gan.py    # PGAN
│   ├── README.md             # models' readme
├── output_networks           # Save training and prediction results
├── visualization             # Visualization and image saving
├── CODE_OF_CONDUCT.md              
├── config_celeba_cropped.json   # Configuration file generated after data preprocessing
├── CONTRIBUTING.md            
├── datasets.py                # Data preprocessing script
├── eval.py                    # Predict and generate image script
├── hubconf.py                 # Used to load the reference code for pre-training, not used
├── LICENSE                    # Open source protocol
├── README.md                  # Home readme
├── requirements.txt           # Other dependencies of the project
├── save_feature_extractor.py    # unused
├── train.py                     # Training script
```

### 6.2 Parameter description
more detail:  [2. Accuracy](#2-accuracy)

### 6.3 Training process
more detail: [5. Quick start](#5-quick-start)

After the execution of the training starts, you will get an output similar to the following. Every 100 iterations will print the current [scale: iters] and generator loss, discriminator loss.

A scale represents the addition of a layer, `scale = len(maxIterAtScale)`, `maxIterAtScale` specifies the corresponding number of iterations for each layer of layer-by-layer training.
 `config_celeba_cropped.json`：
```json
{
  "pathDB": "img_dataset/celeba_cropped",
  "config": {
    "maxIterAtScale": [
      48000,
      96000,
      96000,
      96000,
      96000,
      96000
    ]
  }
}
```

The loss at the beginning will be relatively large, and the size will be proportional to the set batch_size. After 3000 iterations, the loss will stabilize. The time to stabilize may also be related to the set batch_size.

```
Running PGAN
size 10
202599 images found
202599 images detected
size (4, 4)
202599 images found
Changing alpha to 0.000
[0 :    100] loss G : 614.970 loss D : 750532.237
[0 :    200] loss G : 1535.155 loss D : 322471.667
[0 :    300] loss G : 1557.878 loss D : 211534.072
[0 :    400] loss G : 1459.596 loss D : 155299.552
[0 :    500] loss G : 1289.707 loss D : 108436.870
[0 :    600] loss G : 926.709 loss D : 85481.609
[0 :    700] loss G : 616.158 loss D : 55485.711
[0 :    800] loss G : 521.535 loss D : 37811.031
[0 :    900] loss G : 426.269 loss D : 31965.410
[0 :   1000] loss G : 330.425 loss D : 24301.256
[0 :   1100] loss G : 183.268 loss D : 19704.261
[0 :   1200] loss G : 53.901 loss D : 16482.146
[0 :   1300] loss G : -63.348 loss D : 11397.357
[0 :   1400] loss G : 22.371 loss D : 8459.339
[0 :   1500] loss G : -13.653 loss D : 6577.623
[0 :   1600] loss G : 8.768 loss D : 6329.811
[0 :   1700] loss G : -2.990 loss D : 4607.002
[0 :   1800] loss G : 29.571 loss D : 3684.394
[0 :   1900] loss G : 27.713 loss D : 3607.460
[0 :   2000] loss G : -43.031 loss D : 2106.303
[0 :   2100] loss G : -95.974 loss D : 1928.345
[0 :   2200] loss G : -74.860 loss D : 2405.030
[0 :   2300] loss G : -65.015 loss D : 1664.527
[0 :   2400] loss G : -62.593 loss D : 1063.161
[0 :   2500] loss G : -12.376 loss D : 1379.406
[0 :   2600] loss G : 36.092 loss D : 549.926
[0 :   2700] loss G : 49.579 loss D : 691.503
[0 :   2800] loss G : 49.356 loss D : 52.687
[0 :   2900] loss G : 31.852 loss D : 570.363
[0 :   3000] loss G : 54.769 loss D : 382.479
[0 :   3100] loss G : 62.957 loss D : 491.729
[0 :   3200] loss G : 39.215 loss D : 37.412
[0 :   3300] loss G : 29.801 loss D : 215.652
[0 :   3400] loss G : 20.525 loss D : 27.800
[0 :   3500] loss G : 18.882 loss D : 338.726
[0 :   3600] loss G : 39.331 loss D : 128.357
[0 :   3700] loss G : -11.004 loss D : 93.745
[0 :   3800] loss G : 4.962 loss D : 205.661
[0 :   3900] loss G : 10.032 loss D : 187.112
[0 :   4000] loss G : 15.935 loss D : 11.016
[0 :   4100] loss G : 43.358 loss D : 183.713
[0 :   4200] loss G : 5.674 loss D : 5.614
[0 :   4300] loss G : -21.695 loss D : 285.515
[0 :   4400] loss G : 5.493 loss D : 9.029
```

### 6.4 Evaluation process
more detail: [5. Quick start](#5-quick-start)

The image generated using the final pre-trained model `celeba_cropped_s5_i96000.pdparams` is as follows:
![3](https://img-blog.csdnimg.cn/26afed935c61443da4d0e5bb7f9bee97.png)


## 7. Reproduction experience
![5](https://img-blog.csdnimg.cn/670632d67ade4085985397c04bb1717f.png)

**miniBatchSize**

In the experiment of the paper, the batch_size configuration of PGAN is 64, not the default setting of 16 in the source code. The configuration of batch_size = 16 in the paper is adjusted down after adding a high-resolution layer (also has the effect of reducing video memory), if Using batch_size=16 from beginning to end will result in poor image generation. But in order to prevent the video memory from overflowing, I use the configuration file to set the miniBatchSize that dynamically adapts to each scale.

**SWD metric**

The prediction process will randomly sample 16000 images in the entire celeba_cropped data set to predict and calculate the SWD index of each pair of images (input image and corresponding generated image) under different scales of a model, and use the same model to calculate the index result each time It's different. If the number of samples is changed to several thousand or less, the value of SWD will be very large, but if the number of samples is around 16000, SWD will basically remain unchanged. Since they are all sampled in the training set, the model should fit all the information of more than 200,000 avatars. Why does the SWD indicator become larger when the number of samples is small? I don't understand for the time being.

**MS-SSIM metric**

Since the source code does not provide the implementation of MS-SSIM, I refer to the open source pytorch version of GitHub [https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py](https://github. com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py) to calculate the MS-SSIM index, the result obtained is similar to the test result on the celeba data set in the paper. The paper said that the SWD index can better reflect the difference and change of image quality and structure, while MS-SSIM only measures the change between the output and does not reflect the difference between the generated image and the training set, so after the generated image has been significantly improved , The MS-SSIM index has hardly changed, and the result of the SWD index has become a little better.

**Generate effect**

The paper stated that the network did not fully converge within the specified number of iterations, but stopped training after reaching the specified number of iterations, so the generated image is not perfect enough. If you want to generate a more perfect image, you have to wait a few more days.

**API transform**

Convert the pytorch version code to paddle. Some APIs are not available in paddle, but numpy must have them :smile:. Use numpy to build a bridge for APIs that cannot be found. This is a very good way to reproduce.

## 8. Model information
| information | description |
| --- | --- |
| Author | 绝绝子 |
| Date | 2021.09 |
| Framework version | paddlepaddle 0.0.0 （develop version） |
| Application scenarios | GAN 图像生成 |
| Support hardware | GPU、CPU（RAM >= 16 GB） |
| CELEBA download link | [CELEBA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| AI Studio link | [https://aistudio.baidu.com/aistudio/projectdetail/2351963](https://aistudio.baidu.com/aistudio/projectdetail/2351963) |

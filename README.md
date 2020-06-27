# ultimate-image-classification
A black box module to use on any image classification problem using Tensorflow 2 and Keras. Plug in your data, choose which pre-trained model to finetune, and start training!


![ultimate image](https://user-images.githubusercontent.com/6074821/85757853-1a7d4b00-b710-11ea-8da6-256a97d6edd3.PNG)

# Features

### Model selection (pre-trained or not). You can choose from the following models: 
   - VGG16, VGG19
   - Densenet121, Densenet169, Densenet201
   - Xception
   - ResNet50, ResNet101, ResNet152
   - InceptionV3
   - InceptionResNetV2
   - NASNetMobile, NASNetLarge
   - MobileNet, MobileNetV2
   - EfficientNetB0 to EfficientNetB7

### Data Augmentations
Automatic data augmentations to the images like:
- Rotations
- Scaling
- Flipping
- Color augmentations 
- Contrast augmentations
- Shear
- Translation

![image](https://user-images.githubusercontent.com/6074821/85763025-9a0d1900-b714-11ea-8d32-553eff595106.png)

We use cpu workers and queues to prepare the data fetching, normalization, and augmentaions to not bottleneck the gpu during training

### Downscaling CNN
An optional CNN to downscale the image taken from this [paper](https://arxiv.org/pdf/1707.09482.pdf). <br/>
Use it in case you feel that the normal image resizing takes too much information from the image and want the downscaling to be learnable.
<br/>
<img src="https://user-images.githubusercontent.com/6074821/85765512-79de5980-b716-11ea-94f1-f3f1bff61e99.png" width="500" align="left|top">
<br/>

### Learning <br/>
There are many parameters you can control during the training very easily from the configs file:
- Control train/test split 
- Control input image size
- Single label or multi-label classification. The loss and final activation will change automatically.
- Specify the architecture of the fully connected layers at the end 
- Specify which layers to train and which to freeze
- Number of layers to be popped from the pre-trained model
- Optimizer type (Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam)
- Batch size
- learning rate and learning rate decay policy
- Automatic class weighting in case of imbalanced classes
- Automatic best checkpoint saving
- Specify model path and continue training from a checkpoint

### Evaluation <br/>
We want to make your life easier when it comes to evaluation so we added the following options:

- Choose output path that will contain the model, configs used, and training logs
- Automatic tensorboard support and logging
- Draw activations on the input images using [GradCam](https://arxiv.org/abs/1610.02391)
- Get accuracy, precision, recall, and confusion matric for single label classification
- Get AUC-ROC, precision, recall, F1, and automatic best threshold for exact match for multilabel classification

# Usage

- Download this repo
- Have python 3.6+ 
- pip install -r requirements.txt
- Have a csv describing the data using path and class columns like:
   <br/>
   <img src="https://user-images.githubusercontent.com/6074821/85778023-705aee80-b722-11ea-936a-38b6d20329f8.png" width="500" align="left|top">
   <br/>
   - Note that if it's a multilabel problem the classes should be in the following format: class1$class2$class3...
   - If your data is in a structure where each folder represent a class and the images are inside that folder like:<br/>
    &nbsp; images\\<br/>
    &nbsp;&nbsp;&nbsp;        class1\\<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          img1<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          img2<br/>
    &nbsp;&nbsp;&nbsp;    class2\\<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;          img1<br/>
    You can run make_csv_from_folders.py to make the csv automatically
- After having a csv for the data you can split it into training and testing by running split_train_test.py which will generate a csv for training and testing. Of course you can also specify them manually.
- Open configs.py and link to the training and testing csvs, and change any training parameters
- Run train.py
- Link to the saved model in configs.py in 'load_model_path' and run test.py
- Run draw_activations.py to get activation highlighting on the original images. The images will be saved next to the model (optional)
- Run compress weights to get smaller saved model (optional)
- Draw training log by running draw_training_log.py (optional)


# Example
**Note: The following is a brief overview of an example usecase, for a complete walkthrough please visit this [blog](https://omar-mohamed.github.io/technical/2020/06/27/Ultimate-Image-Classification(walkthrough)/).** <br/> <br/>
This is an example usecase on a small dataset of about 450 images. We have 3 classes: gym, pool, and operating room. <br/>
- Download images from [here](https://drive.google.com/drive/folders/1JaXOQW6MUSE5aSFNv_nQyz45h_WtkoJy?usp=sharing) 
- Extract inside 'data' folder to have 'data/images'
- Run train.py
### Results
- These are the results when running on the default configs.py using a pre-trained Densenet121.
 <br/>
 <img src="https://user-images.githubusercontent.com/6074821/85786603-b5831e80-b72a-11ea-9769-117b2ccd44a2.png" width="700" align="left|top">
 <br/>

- Test set results:
 <br/>
 <img src="https://user-images.githubusercontent.com/6074821/85793670-a48bda80-b735-11ea-9fb5-10a4cde986b0.png" width="500" align="left|top">
  
<p float="left|top">

 <img src="https://user-images.githubusercontent.com/6074821/85793918-0c422580-b736-11ea-85ac-3b7bfc883593.png" width="300" height="250" >
 <img src="https://user-images.githubusercontent.com/6074821/85794105-5aefbf80-b736-11ea-8257-0abc79bc2e02.png" width="300" height = "250" >
 <img src="https://user-images.githubusercontent.com/6074821/85794298-a6a26900-b736-11ea-9a5d-752d81a479fa.png" width="300" height = "250">
</p>

# Acknowledgements
- Used Grad-Cam implementation [here](https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/)
- Used & extended on many ideas from this [repo](https://github.com/brucechou1983/CheXNet-Keras).



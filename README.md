# Soccer_Robot_perception
Code for the [RoboCup 2019 AdultSize Winner NimbRo](https://arxiv.org/abs/1912.07405)

## Objective of the project
It is to make a robot perception that outperforms humans in playing soccer in the game field. For this purpose, the Nimbronet2 is trying to mimic human perception by doing two tasks, one task to detect the ball, goalposts and other robots. The other task is to segment the field boundaries, to understand the scene, in which direction should Nimbro Robot shoot, and to ignore any objects outside the arena's boundaries.

## Architecture
![Screenshot from 2022-01-27 09-28-48](https://user-images.githubusercontent.com/49837627/152350927-869ca367-04be-45a3-962c-0fd541c06c2a.png)

The network follows the pixel-wise segmentation encoder-decoder architecture used in [U-Net](https://arxiv.org/abs/1505.04597v1). The enconder layers are a pre-trained ResNet-18, where the Global Average Pooling (GAP) and the fully connected layers are removed (to change the task of the encoder). The output of the encoder pass through a ReLU function, and then through the decoder. The decoder consists of three transpose-convolution layers, each transpose-convolution is followed by a ReLU activation function and batch normalization. The output of each encoder's layer pass through 1x1 convolution and then is concatenated (pixel-wisely) with the output of the corresponding decoder's layer (to provide high-resolution details of the output). The decoder has two output heads, one for detection, and one for segmentation, where both of them uses a [location-dependent convolution](https://arxiv.org/abs/1810.04937) layer with a shared learnable bias (shared between the two heads). The detection head produces a heatmap that represents the probability for the locations of the balls, robots, and goalposts (each in a separate channel). The segmentation head outputs the probability of each class (field, line, or background) for each pixel. For the seek of the number of parameters reduction, the decoder is shorter than the encoder, resulting in output images that are 1/4 of the size of the input images. 

## Dataset
- The dataset was collected to consider different **lighting conditions**, and **camera views**, to make the model robust to those cases.
- The data contains 8858 samples of the Detection dataset and 1192 samples of the Segmentation dataset.  
### Detection Dataset
It consists of:
- Input images, in the form of frames from old RoboCup games videos, in addition to newly captured images; in the lab arena.
- Xml files that contain the coordinates of the bounding box - of the location - of the objects, along with their labels. Only objects inside the field were annotated, to force the model, consider only the objects inside the field and ignore all the objects outside (besides the help of location-dependent convolution layer).
### Segmentation Dataset
It consists of:
- Input images, in the form of frames from old RoboCup games videos, in addition to newly captured images; in the lab arena.
- Target images - grayscale images -, in the form of field segmented images.
### Data Augmentation
- Data are being augmented using the Color Jitter transformer, to randomly change the Brightness, Contrast, and Saturation of the input images.
### Data Split
Each dataset is being split into an 80:10:10 ratio.
### Data Imbalance
- To mitigate the problem of the data imbalance, the Segmentation loss is having a higher weight - than the Detection loss - in the total loss.
### Problems in the Dataset
- The input images are coming from different sources, so they had different sizes, and the model is expecting a fixed size of the input.
- Segmentation target images have different distributions of the color values for the line, field, and background.
- Some segmentation target images have the class values {0,1,2,3} instead of the RGB values.
- The input images are of different extensions.
- The available input images for the detection are mostly different from the ones available for the segmentation. 
- There are noisy images, that needed to be removed.
### Data processing
#### Preprocessing
To reduce the burden of computation during the training phase, steps 1., 2., and 3. were done in the preprocessing stage; to avoid computing them each time fetching the data using the dataloader. 
1. Creating the heatmap images (the images with the Gaussian blobs that represents the belief of the location of each object) (that will be used as target images in the Detection dataset) from the Xml files. Where each channel of this heatmap image, represent the probability of the existence of one of the 3 possible objects (i.e. ball, goalpost, robot). The blob in the channel that represents the robot, is being double the size of the other two channels blobs, to prevent the model from extra penalizing for the wrong robots.
2. Resize all the images to a fixed size.
3. Fixing the values' distribution of the segmentation target images.
4. Saving the paths of the images, heatmaps, and segmentation label images.
#### Real-time Processing
- Normalizing the images according to the ImageNet distribution, before feeding them to the model.
- In the tensor space: converting segmentation target images from their continuous values, into the class discrete values (i.e. 0 for background, 1 for line, 2 for the game field).   
#### Post-processing
- The output of the segmentation head (pixel-wise class index), is mapped back to each class color value!
- The output of the detection head, is clipped for any value is out of the min and maximum ranges. 
- Using the openCV function **findContours**, to detect the center of the detected objects in the output heatmaps (that are coming from the detection head). Where the label of the object can be inferred from the color of the detected object.
## Training
- To mitigate the problem that the input images of the Segmentation and Detection Dataset are different, the model is being trained by fetching a batch of Detection dataset, computing its loss (**MSE** + **Variational loss**), then fetching a batch of Segmentation Dataset, computing its loss (**Cross-Entropy** + **Variational loss**), and then the gradient of the total of both losses is being computed (to update the model's parameters). 
- To mitigate the problem of the data imbalance, the Segmentation loss is having a higher weight - than the Detection loss - in the total loss.
- Adam optimizer is being used. 
- To mitigate the problem of finding the best learning rate, **Cyclic Learning Rate** is being used.
- A batch size of 32, 200 epochs, 200 iteration/epoch are being used.
- Garbage collector, Releasing all unoccupied cached data, Resetting the starting point in tracking maximum GPU memory, and Automatic Mixed Precision Package are being used to avoid any CUDA out of memory errors.
### Transfer Learning
- In the first 50 epochs, the encoder part's layers were frozen (the pre-trained Resnet18), and the learning was happening only for the decoder's layers. After those 50 epochs, the encoder was unfrozen back and the encoder and decoder were trained jointly.
## Results
### output
#### Detection
![Screenshot from 2022-02-04 02-25-36](https://user-images.githubusercontent.com/49837627/152457211-48ef6be1-d4c0-4b0a-879f-581da1c6c2b6.png)
![Screenshot from 2022-02-04 02-24-39](https://user-images.githubusercontent.com/49837627/152457264-938bb267-3b10-4ff1-95bb-04d0fd4bbffa.png)
#### Segmentation
![Screenshot from 2022-02-04 02-24-08](https://user-images.githubusercontent.com/49837627/152457308-8fff04a8-7dcc-4dd7-823d-f6a9b7c5b897.png)
![Screenshot from 2022-02-04 02-26-04](https://user-images.githubusercontent.com/49837627/152457320-e691933b-1c7d-4c2b-bc1a-5f34227556c8.png)
### Performance
#### Detection
![Screenshot from 2022-02-04 02-02-46](https://user-images.githubusercontent.com/49837627/152457390-6fabb73d-a394-45f3-8389-02ad676d1912.png)
#### Segmentation
![Screenshot from 2022-02-04 02-12-13](https://user-images.githubusercontent.com/49837627/152457422-8c48f2a1-101c-4a66-b3e1-577157013187.png)
## Debug
In the debugging phase, I was trying to overfit one batch, to troubleshoot the problems, also I was removing the regaulizer, and making the model as simple as possible, to find the issues that I was facing.

## Code Architecture
- **Pre-processing_functions.ipynb**: is containing the functions being used for the dataset preprocessing.
- **Extra_Utilities.ipynb**: is containing some auxulary functions being used in the other files.
- **Custom_Dataset.ipynb**: is containg the customized dataset classes for detection and segmentation.
- **Nibronet2_Model.ipynb**: is containg the Nimbronet2 model archicture implementation.
- **Training.ipynb**: is containg - the core of the project -, the implementation needed for the training iterations.
- **Evaluation.ipynb**: is containing the testing phase implementations, and the computation of the performance metrics.
- **Soccer Robot Perception Report.pdf**: is the detailed report about this project.

For the details the report is in the file Soccer Robot Perception Report.pdf

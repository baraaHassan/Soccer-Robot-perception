# Soccer_Robot_perception
Code for the [RoboCup 2019 AdultSize Winner NimbRo](https://arxiv.org/abs/1912.07405)

## Objective of the project

## Architecture

## Dataset
The dataset was collected to consider all lighting conditions, and all camera views, to make the model robust to those cases.
### Detection Dataset
It consists of:
- Input images, in the form of frames from old robocup games videos, in addition to new captured images.
- Xml files that contain the coordinates of the bounding box - of the location - of the objects, along with their labels. Only objects inside the field was annotated, to force the model, cosider only the objects inside the field and ignore all the objects outside (besides the help of location-dependent convolution layer).
### Segmentation Dataset
- Input images, as frames from old robocup games videos, in addition to new captured images.
- Label images - grayscale images -, in the form of field segmented images.
### Problems in the Dataset
- The input images are coming from different sources, so they had different sizes, and the model is expecting a fixed size of input.
- Segmentation target images have different distribution of the color values for the line,field, and background.
- Some segmentation target images have the class values {0,1,2,3} instead the RGB values.
- The input images are of different extensions.
- The available input images for the detection are mostly different from the one available for the segmentation. 
- There are noisy images, that needed to be removed.
### Data processing
#### Preprocessing
To reduce the burden of computation during training phase, step 1., 2., and 3. were done in the preprocessing stage; to avoid computing them each time fetching the data using the dataloader. 
1. Creating the heatmap images (the images with the blobs that represents the believe of the location of each object) (that will be used as target for the Detection dataset) from the Xml files.
2. Resizing all the images to fixed size.
3. Fixing the distribution of the segmentation target images values.
4. Normalizing the images
5. Saving the pathes of the images, heatmaps, and segmentation label images.
#### Real-time Processing
- In the tensor space : converting segmentation target images from thier continuous values, into the class discrete values (i.e. 0 for background, 1 for line, 2 for the gamefield).   
#### Post-processing
- Using the opencV
### Data Imbalance
- To mitigate the problem of the data imbalance, the Segmentation loss is having higher weight - than the Detection loss - in the total loss.

## Data Augmentation
- Data are being augemented using the Color Jitter transformer, to randomly change the Brightness, Contrast, and Saturation of the input images.
## Training
- To mitigate the problem that the input images of the Segmentation Dataset and Detection Dataset are different, the model is being trained by fetching a batch of Detection dataset, computing its loss (**MSE** + **Varitional loss**), then fetching a batch of Segmentation Dataset, computing its loss (**Cross Entropy** + **Varitional loss**), and then the gradient of the total losses is being computed (to update the model parameters). 
- To mitigate the problem of the data imbalance, the Segmentation loss is having higher weight - than the Detection loss - in the total loss.
- Adam optimizer is being used. 
- To mitigate the problem of finding best learning rate, **Cyclic Learning Rate** is being used.
- Batch size 32 is being used.
- Garbage collector, Releasing all unoccupied cached data, Reseting the starting point in tracking maximum GPU memory, and Automatic Mixed Precision Package are being used to avoid any CUDA out of memory errors.

## Results
### output
#### Detection
#### Segmentation
### Performance
#### Detection
#### Segmentation

## Code Architecture


For the details the report is in the file Soccer Robot Perception Report.pdf

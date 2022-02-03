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
- Some segmentation images have the class values {0,1,2,3} instead the RGB values.
- Some segmentation images have different distribution of the color values for the line,field, and background.
- The input images are of different extensions.
- The available input images for the detection are mostly different from the one available for the segmentation. 
- There are noisy images, that needed to be removed.
### Data processing
#### Preprocessing
#### Real-time Processing
#### Post-processing


## Data Augmentation

## Training

## Results
### output
#### Detection
#### Segmentation
### Performance
#### Detection
#### Segmentation

## Code Architecture


For the details the report is in the file Soccer Robot Perception Report.pdf

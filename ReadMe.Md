# Project III : Real Time 2-D Object Detection 

## Created by Rucha Pendharkar on 2/25/24

The goal of this project is to have the system identify a specified set of objects placed on a white surface in a translation, scale, and rotation invariant manner from a camera looking straight down. To achieve this a sequence of tasks such as thresholding, cleaning, segmentation, feature calculation, is performed. Different methods of feature matching and classification are carried out using the database of objects.


## Features

- **Real-time Video Capture**: The system captures video in real-time using the camera from a mobile phone. Make sure to update the IP address in the code to run

- **Image Processing**: The video is converted thresholded to create a binary image. The binary image is then cleaned for processing

- **Region Identification**: Regions are identified using OpenCV's connected components method.

- **Feature Extraction**: For each identified region, several features are computed - centroid, Hu moments, the percentage of the region filled, and the height-to-width ratio of the oriented bounding box.

- **Classification**: Each region is classified using two algorithms: nearest neighbor and K nearest neighbor 

- **Training Mode**: The system has a training mode that allows users to add new features to the training set. The training set is saved to a CSV file after every update.
 
## Usage

- Press 'q' to exit the application.
- Press 'n' to enter training mode.

## Environment 
The scripts were authored using VS Code, and code compilation took place in the Ubuntu 20.04.06 LTS environment, utilizing CMake through the terminal.

## Notes
I am using 1 timetravel day

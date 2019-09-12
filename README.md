
Details |         |
--------|---------|
Target OS | Ubuntu* 16.04 LTS |
Programming Language | C++ |
Time to complete | 45min |

# What it Does
This application demonstrates how to create a working smart video solution to detect human poses while generating a motion heatmap. The implementation can be used as a foundation to analyze traffic patterns on factory floors and if workers are performing correct procedures. It can also be used to analyze retail/store shelves to see if people are grabbing certain products more than others. Users can also perform sports analysis on players poses and frequency of movement on areas in a court or field. 

# How it Works
The application detects a human skeleton and pixel changes in a video input or camera stream to generate the heatmap for each frame. The heatmap is generated by OpenCV 4.0+ functions: background subtraction, application of a threshold, accumlation of changed pixels over time and then adding a color/heat map. 
The pose detection uses a two-branch multistage CNN using OpenPose architecture to detect 18 keypoints: ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees and ankles. The first branch predicts confidence maps S n and The second branch predicts Part Affinity Fields L n. The two map predictions are parsed with the input image features and will product the keypoints. For more information please refer to [human-pose-estimation-001](https://docs.openvinotoolkit.org/latest/_intel_models_human_pose_estimation_0001_description_human_pose_estimation_0001.html) in the OpenVINO Pre-Trained Open Model Zoo. [And the network architecture information] (https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/) 
The pose detector and heatmap generator are merged together 

## Note
If the trained model you use is in RGB order please manually rearrange or use the Model Optimizer tool with --reverse_input_channels argument to convert to BGR order which is the expected input for Inference Engine samples and demos. Refer to When to Specify Input Shapes section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/2019_R1/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html)

# Required
Install and setup the [Intel Distribution of OpenVINO toolkit 2019 R2 Release] (https://software.intel.com/en-us/openvino-toolkit/choose-download)

# Running

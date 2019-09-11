
Details |         |
--------|---------|
Target OS | Ubuntu* 16.04 LTS |
Programming Language | C++ |
Time to complete | 45min |

# What it Does
This application demonstrates how to create a working smart video solution to detect human poses while generating a motion heatmap. The implementation can be used as a foundation to analyze traffic patterns on factory floors and if workers are performing correct procedures. It can also be used to analyze retail/store shelves to see if people are grabbing certain products more than others. Users can also perform sports analysis on players poses and frequency of movement on areas in a court or field. 

# How it Works
First The application uses either video input or a camera stream and detects for a human skeleton to detect a person and pixel changes to generate the heatmap for each frame. The pose uses a multi-person 2D pose estimation network to detect 18 keypoints: ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees and ankles. Please refer to the pretrained model [human-pose-estimation-001](https://docs.openvinotoolkit.org/latest/_intel_models_human_pose_estimation_0001_description_human_pose_estimation_0001.html) in the OpenVINO Pre-Trained Open Model Zoo. 

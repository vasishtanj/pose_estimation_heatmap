
Details |         |
--------|---------|
Target OS | Ubuntu* 16.04 LTS |
Programming Language | C++ |
Time to complete | 1hr |


# What it Does
This application demonstrates how to create a working smart video solution to detect human poses while generating a motion heatmap. The implementation can be used as a foundation to analyze traffic patterns on factory floors and if workers are performing correct procedures. It can also be used to analyze retail/store shelves to see if people are grabbing certain products more than others. Users can also perform sports analysis on players poses and frequency of movement on areas in a court or field. 
This application is built upon the [Human Pose Estimation C++ Demo](https://docs.openvinotoolkit.org/2019_R1/_inference_engine_samples_human_pose_estimation_demo_README.html) sample c++ which is provided in the OpenVINO Toolkit.

# How it Works
The application detects a human skeleton and pixel changes in a video input or camera stream to generate the heatmap for each frame. The heatmap implementation was added to main.cpp of the Human Pose Estimation C++ Demo. The heatmap is generated by OpenCV 4.0+ functions: background subtraction, application of a threshold, accumlation of changed pixels over time and then adding a color/heat map. 
The pose detection uses a two-branch multistage CNN using OpenPose architecture to detect 18 keypoints: ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees and ankles. The first branch predicts confidence maps S n and The second branch predicts Part Affinity Fields L n. The two map predictions are parsed with the input image features and will product the keypoints. For more information please refer to [human-pose-estimation-001](https://docs.openvinotoolkit.org/latest/_intel_models_human_pose_estimation_0001_description_human_pose_estimation_0001.html) in the OpenVINO Pre-Trained Open Model Zoo. [And the network architecture information](https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/) 
The pose detector and heatmap generator are merged together by estimating and rendering the poses in each frame with the heatmap and a new image is created.

## Note
If the trained model you use is in RGB order please manually rearrange or use the Model Optimizer tool with --reverse_input_channels argument to convert to BGR order which is the expected input for Inference Engine samples and demos. Refer to When to Specify Input Shapes section of [Converting a Model Using General Conversion Parameters](https://docs.openvinotoolkit.org/2019_R1/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html)

# Required
Because this application is built upon the human pose estimation c++ sample provided in the toolkit, you will need to install and setup the [Intel Distribution of OpenVINO toolkit 2019 R2 Release](https://software.intel.com/en-us/openvino-toolkit/choose-download)

# Setup
This application is set up to run the [human-pose-estimation-001](https://docs.openvinotoolkit.org/latest/_intel_models_human_pose_estimation_0001_description_human_pose_estimation_0001.html) from the OpenVINO Pre-Trained Open Model Zoo, however you can use public or pre-trained models. To download use the [OpenVINO Model Downloader](https://software.intel.com/en-us/articles/model-downloader-essentials). 
Also it is required to use the [OpenVINO Model Optimizer](https://docs.openvinotoolkit.org/2019_R1/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) to convert files to Inference Engine format (.xml and .bin) 
Go to the human_pose_estimation_demo directory

    cd /opt/intel/openvino/inference_engine/samples/human_pose_estimation_demo/ 

Then update enviornment variables required to compile and run OpenVINO toolkit using the following script.
    
    source /opt/intel/openvino/bin/setupvars.sh

# Running
Then build the samples by going back to the samples directory and run 

    cd /opt/intel/openvino/inference_engine/samples
    ./build_samples.sh

***NOTE:*** Because OpenCV libraries are being used to implement the heatmap, the ***video*** library was manually linked for CMake to read and use. The library was linked on line 8 of the CMakefiles.txt highgui. 

    find_package(OpenCV COMPONENTS highui video QUIET)

Finally, go to the directory from where you will run the application

    cd inference_engine_samples_build/intel64/Release

and run using the following command

    ./human_pose_estimation_demo -i [path to your video file] -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/human_pose_estimation/mobilenet-v1/dldt/human-pose-estimation-0001.xml -d CPU 


./human_pose_estimation_demo  -h will bring up a menu 

    ./human_pose_estimation_demo -h
 
    InferenceEngine:
    API version ............ <version>
    Build .................. <number>
   
    human_pose_estimation_demo [OPTION]
    Options:

        -h                         Print a usage message.
        -i "<path>"                Required. Path to a video. Default value is "cam" to work with camera.
        -m "<path>"                Required. Path to the Human Pose Estimation model (.xml) file.
        -d "<device>"              Optional. Specify the target device for Human Pose Estimation (the list of available devices is shown      below). Default value is CPU. Use "-d HETERO:<comma-separated_devices_list>" format to specify HETERO plugin. The application looks for a suitable plugin for the specified device.
        -pc                        Optional. Enable per-layer performance report.
        -no_show                   Optional. Do not show processed video.
        -r                         Optional. Output inference results as raw values.

# Demo Output

The demo should be running and saved as result_overlay_final.jpg in your Release folder.
The demo uses OpenCV to display the resulting frame with estimated poses while generating a motion heatmap. 
***Follow Up***
The application may glitch and the assumption is that it needs to run on a more powerful device such as an IEI tank and/or needs to be optimized for the two features to run in parallel 

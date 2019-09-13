// Copyright C 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


/**
* \brief The entry point for the Inference Engine Human Pose Estimation demo application
* \file human_pose_estimation_demo/main.cpp
* \example human_pose_estimation_demo/main.cpp
*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <ctime>
#include "opencv2/video/background_segm.hpp"


#include <vector>


#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>

#include "human_pose_estimation_demo.hpp"
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"

using namespace InferenceEngine;
using namespace human_pose_estimation;
using namespace std;
using namespace cv;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    std::cout << "[ INFO ] Parsing input parameters" << std::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

int main(int argc, char* argv[]) {
	 			
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return EXIT_SUCCESS;
        }

        HumanPoseEstimator estimator(FLAGS_m, FLAGS_d, FLAGS_pc);
        cv::VideoCapture cap;   
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
	
	 }

        int delay = 33;
        double inferenceTime = 0.0;
        cv::Mat image;
        if (!cap.read(image)) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }
        estimator.estimate(image);  // Do not measure network reshape, if it happened
        if (!FLAGS_no_show) {
            std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
        } 

	
//heatmap integration starts here
	
	
	int maxValue = 2; 
	// Default resolution of the frame is obtained.The default resolution is system dependent. Parsing frame properties
	int frame_width = cap.get(CAP_PROP_FRAME_WIDTH) , frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
	int fps = cap.get(CAP_PROP_FPS);
	

	//Video Saving APIs
	VideoWriter heatmap_vid_obj_file("heatmap_video.avi",cv::VideoWriter::fourcc('M','J','P','G'),fps, Size(frame_width,frame_height));
	VideoWriter integrated_video_object_file("integrated_video.avi", VideoWriter::fourcc('M','J','P','G'), fps, Size(frame_width, frame_height));

	//begsegmentation pointer initialised
	Ptr<BackgroundSubtractorMOG2> background_segmentor_object_file = createBackgroundSubtractorMOG2();

	
	Mat gray,accum_image,first_frame,frame,frame_dup,fgbgmask,threshold_image,color_image,result_overlay,accum_image_duplicate,color_image1,duplicate_final,result_overlay_video,final_img;
	cap.read(first_frame);


	while(cap.isOpened())
	{
		cap.read(frame);
		if (frame.empty())
		{
			std::cout<<"Video stream ended";
			return -1;	
		}
	
		
			
	//Converting to Grayscale
	cvtColor(frame, gray, COLOR_BGR2GRAY);

	// Remove the background
	background_segmentor_object_file ->apply(gray, gray);

	int thres =2;
	// Thresholding the image
	threshold(gray, gray, thres, maxValue, THRESH_BINARY);

	// Adding to the accumulated image
	accum_image = gray + accum_image;

	// Saving the accumulated image
	applyColorMap(accum_image, result_overlay_video, COLORMAP_HOT);
	addWeighted(frame, 0.5, result_overlay_video, 0.5, 0.0, result_overlay_video);

	cv::Mat image = cv::Mat::zeros(frame_height, frame_width, CV_8UC3); 
	auto poses = estimator.estimate(frame);
	renderHumanPose(poses, image);
	image.copyTo(result_overlay_video, image);

	// Heatmap video generation
	heatmap_vid_obj_file.write(result_overlay_video);
	//imwrite("final", result_overlay_video);
	//imshow("final", result_overlay_video);
	//count = setup(frame);
	

	// Integrated video generation
	addWeighted(frame, 0.45, result_overlay_video, 0.55, 0.0, final_img);   
	integrated_video_object_file.write(final_img);				


// Adding all accumulated frames to the first frame
	
	imwrite( "result_overlay_final.jpg", final_img);
	imshow ("result_overlay_final.jpg", final_img);
	

	}
//heatmap integration ends here
	    
        do {
            double t1 = static_cast<double>(cv::getTickCount());
            std::vector<HumanPose> poses = estimator.estimate(image);
            double t2 = static_cast<double>(cv::getTickCount());
            if (inferenceTime == 0) {
                inferenceTime = (t2 - t1) / cv::getTickFrequency() * 1000;
            } else {
                inferenceTime = inferenceTime * 0.95 + 0.05 * (t2 - t1) / cv::getTickFrequency() * 1000;
            }
            if (FLAGS_r) {
                for (HumanPose const& pose : poses) {
                    std::stringstream rawPose;
                    rawPose << std::fixed << std::setprecision(0);
                    for (auto const& keypoint : pose.keypoints) {
                        rawPose << keypoint.x << "," << keypoint.y << " ";
                    }
                    rawPose << pose.score;
                    std::cout << rawPose.str() << std::endl;
                }
            }

            if (FLAGS_no_show) {
                continue;
            }

            renderHumanPose(poses, image);

            cv::Mat fpsPane(35, 155, CV_8UC3);
            fpsPane.setTo(cv::Scalar(153, 119, 76));
            cv::Mat srcRegion = image(cv::Rect(8, 8, fpsPane.cols, fpsPane.rows));
            cv::addWeighted(srcRegion, 0.4, fpsPane, 0.6, 0, srcRegion);
	    
	    std::stringstream fpsSs;
            fpsSs << "FPS: " << int(1000.0f / inferenceTime * 100) / 100.0f;
            cv::putText(image, fpsSs.str(), cv::Point(16, 32),
                        cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255));
            cv::imshow("ICV Human Pose Estimation", image);

            int key = cv::waitKey(delay) & 255;
            if (key == 'p') {
                delay = (delay == 0) ? 33 : 0;
            } else if (key == 27) {
                break;
            }
	
        } while (cap.read(image));
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[ INFO ] Execution successful" << std::endl;
    return EXIT_SUCCESS;
}

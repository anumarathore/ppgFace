///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED “AS IS” FOR ACADEMIC USE ONLY AND ANY EXPRESS
// OR IMPLIED WARRANTIES WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY.
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called “open source” software licenses (“Open Source
// Components”), which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensee’s request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////
// Author : Ramya kolli
// Date Modified : 21th December, 2016
// FaceRecognition_train.cpp : Defines the entry point for the console application for training images for face recognition.
////////////////////////////////////////////////////////////////////////////////

#include "LandmarkCoreIncludes.h"

// System includes
#include <fstream>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

#include <dlib/image_processing/frontal_face_detector.h>

#include <tbb/tbb.h>
#include "2DfaceRecognition.hpp"


using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

void convert_to_grayscale(const cv::Mat& in, cv::Mat& out)
{
	if(in.channels() == 3)
	{
		// Make sure it's in a correct format
		if(in.depth() != CV_8U)
		{
			if(in.depth() == CV_16U)
			{
				cv::Mat tmp = in / 256;
				tmp.convertTo(tmp, CV_8U);
				cv::cvtColor(tmp, out, CV_BGR2GRAY);
			}
		}
		else
		{
			cv::cvtColor(in, out, CV_BGR2GRAY);
		}
	}
	else if(in.channels() == 4)
	{
		cv::cvtColor(in, out, CV_BGRA2GRAY);
	}
	else
	{
		if(in.depth() == CV_16U)
		{
			cv::Mat tmp = in / 256;
			out = tmp.clone();
		}
		else if(in.depth() != CV_8U)
		{
			in.convertTo(out, CV_8U);
		}
		else
		{
			out = in.clone();
		}
	}
}


int main (int argc, char **argv)
{
	cv::RNG rng(12345);
	bool train_status=false;
	FaceRecognition *faceRecognizer = new FaceRecognition;

	//Convert arguments to more convenient vector form
	vector<string> arguments = get_arguments(argc, argv);

	// Some initial parameters that can be overriden from command line
	vector<string> files, depth_files, output_images, output_landmark_locations, output_pose_locations;

	// Bounding boxes for a face in each image (optional)
	vector<cv::Rect_<double> > bounding_boxes;
	
	LandmarkDetector::get_image_input_output_params(files, depth_files, output_landmark_locations, output_pose_locations, output_images, bounding_boxes, arguments);
	LandmarkDetector::FaceModelParameters det_parameters(arguments);	
	// No need to validate detections, as we're not doing tracking
	det_parameters.validate_detections = false;

	// Grab camera parameters if provided (only used for pose and eye gaze and are quite important for accurate estimates)
	float fx = 0, fy = 0, cx = 0, cy = 0;
	int device = -1;
	LandmarkDetector::get_camera_params(device, fx, fy, cx, cy, arguments);

	// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
	bool cx_undefined = false;
	bool fx_undefined = false;
	if (cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}
	if (fx == 0 || fy == 0)
	{
		fx_undefined = true;
	}

	// The modules that are being used for tracking
	cout << "Loading the model" << endl;
	LandmarkDetector::CLNF clnf_model(det_parameters.model_location);
	cout << "Model loaded" << endl;
	
	cv::CascadeClassifier classifier(det_parameters.face_detector_location);
	dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();

	bool visualise = !det_parameters.quiet_mode;
	
	// Do some image loading
	for(size_t i = 0; i < files.size(); i++)
	{
		string file = files.at(i);
		boost::filesystem::path input_file_path(file);
		boost::filesystem::path fname = input_file_path.parent_path();


		// Loading image
		cv::Mat read_image = cv::imread(file, -1);
		Mat display_image=read_image.clone();
		if (read_image.empty())
		{
			cout << "Could not read the input image" << endl;
			return -1;
		}

		// Loading depth file if exists (optional)
		cv::Mat_<float> depth_image;

		if(depth_files.size() > 0)
		{
			string dFile = depth_files.at(i);
			cv::Mat dTemp = cv::imread(dFile, -1);
			dTemp.convertTo(depth_image, CV_32F);
		}

		// Making sure the image is in uchar grayscale
		cv::Mat_<uchar> grayscale_image;
		convert_to_grayscale(read_image, grayscale_image);
		

		// If optical centers are not defined just use center of image
		if (cx_undefined)
		{
			cx = grayscale_image.cols / 2.0f;
			cy = grayscale_image.rows / 2.0f;
		}
		// Use a rough guess-timate of focal length
		if (fx_undefined)
		{
			fx = 500 * (grayscale_image.cols / 640.0);
			fy = 500 * (grayscale_image.rows / 480.0);

			fx = (fx + fy) / 2.0;
			fy = fx;
		}

			// Detect faces in an image
			vector<cv::Rect_<double> > face_detections;

			if(det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
			{
				vector<double> confidences;
				LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, face_detector_hog, confidences);
			}
			else
			{
				LandmarkDetector::DetectFaces(face_detections, grayscale_image, classifier);
			}
		if(face_detections.size() > 1)
			{
				cerr<<"skipping image,multiple faces";
				continue;
			}
			// perform landmark detection for every face detected
			for(size_t face=0; face < face_detections.size(); ++face)
				{
				// if there are multiple detections go through them
				bool success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, depth_image, face_detections[face], clnf_model, det_parameters);
				double detection_certainty = clnf_model.detection_certainty;

					double visualisation_boundary = -0.1;

					// Only draw if the reliability is reasonable, the value is slightly ad-hoc
					if (detection_certainty < visualisation_boundary)
					{
						//LandmarkDetector::Draw(display_image, clnf_models[model]);

						if (detection_certainty > 1)
							detection_certainty = 1;
						if (detection_certainty < -1)
							detection_certainty = -1;

						detection_certainty = (detection_certainty + 1) / (visualisation_boundary + 1);

				// Estimate head pose and eye gaze				
				cv::Vec6d headPose = LandmarkDetector::GetCorrectedPoseWorld(clnf_model, fx, fy, cx, cy);

				
				//FACE RECOGNITION MODULE 
						Mat tmp_img = Mat::zeros(display_image.size(), CV_32FC2);
						tmp_img = display_image.clone();
						Rect roi = clnf_model.GetBoundingBox();
						double confidence = 0.5 * (1.0 - clnf_model.detection_certainty);
						//cout << "Confidence : " << confidence << "\t" << "Success : " << clnf_models[model].detection_success << endl;
						if (confidence > 0.9 && clnf_model.detection_success)
						{
							try {
								// Boundary cases are being ignored for now
							
								float Rx = headPose[3];//pitch
								float Ry = headPose[4]; //yaw
								float Rz = headPose[5]; //roll
								

									//constrain pitch roll and yaw.
									//NOTES: how to put a limit on pitch and yaw, should we do it? 
									//if ((Rx < pitch_max) && (Rx > pitch_min) && (Ry < yaw_max) && (Ry > yaw_min) && (Rz < roll_max) && (Rz > roll_min))
									if(true)
									{
										//NOTE : background model was trained with (abs(Rx) < 0.3) && (abs(Ry) < 0.2) && (abs(Rz) < 0.35)
										//rotating the image.. 

										Mat rotated_img, alligned_img;
										Point2f eye1 = (Point(clnf_model.detected_landmarks.at<double>(37), clnf_model.detected_landmarks.at<double>(37 + 68)) + Point(clnf_model.detected_landmarks.at<double>(40), clnf_model.detected_landmarks.at<double>(40 + 68))) / 2;
										Point2f eye2 = (Point(clnf_model.detected_landmarks.at<double>(43), clnf_model.detected_landmarks.at<double>(43 + 68)) + Point(clnf_model.detected_landmarks.at<double>(46), clnf_model.detected_landmarks.at<double>(46 + 68))) / 2;
										Point2f nose = Point(clnf_model.detected_landmarks.at<double>(34), clnf_model.detected_landmarks.at<double>(34 + 68));
										/*circle(display_image, eye1, 2, Scalar(255, 255, 255));
										circle(display_image, eye2, 2, Scalar(255, 255, 255));

										cout << eye1 << " " << eye2 << endl;*/
										Point2f eye_mp = (eye1 + eye2) / 2;
										float  angle = atan(eye1.y - eye2.y) / (eye1.x - eye2.x);
										//cout << "angle = " << angle << endl;
										Mat roatxn_mat = getRotationMatrix2D(nose, Rz * 180 / M_PI, 1); //angle * 180 / (M_PI)
										warpAffine(tmp_img, rotated_img, roatxn_mat, alligned_img.size(), 1, 0, Scalar(0));
										vector<cv::Point> rotated_landmarks;
										for (int iter = 0; iter < clnf_model.pdm.NumberOfPoints(); iter++)
										{
											cv::Point2f p = Point(clnf_model.detected_landmarks.at<double>(iter), clnf_model.detected_landmarks.at<double>(iter + 68));
											cv::Point2f op;
											op.x = std::cos(-Rz)*(p - nose).x - std::sin(-Rz)*(p - nose).y;
											op.y = std::sin(-Rz)*(p - nose).x + std::cos(-Rz)*(p - nose).y;
											op = op + nose;
											rotated_landmarks.push_back(op);
										}
										Rect boundRect = cv::boundingRect(rotated_landmarks);
										Point2f eye1_rotated = (rotated_landmarks[37 - 1] + rotated_landmarks[40 - 1]) / 2;
										Point2f eye2_rotated = (rotated_landmarks[43 - 1] + rotated_landmarks[46 - 1]) / 2;
									
										Rect cropROI;
										int eyesDist = sqrt((eye1_rotated.x - eye2_rotated.x)*(eye1_rotated.x - eye2_rotated.x) + (eye1_rotated.y - eye2_rotated.y)*(eye1_rotated.y - eye2_rotated.y));
										cropROI.x = max(eye1_rotated.x - 0.5*eyesDist, 0.0);
										cropROI.y = max(eye1_rotated.y - 0.5*eyesDist, 0.0);
										cropROI.width = 2 * eyesDist;
										cropROI.height = 2 * eyesDist;
										//imshow("Alligned image", rotated_img(cropROI)); waitKey(5);
										rotated_landmarks.clear();
										train_status = faceRecognizer->Train(rotated_img(cropROI),fname.string(),files.size(),i );
										
									}
									
									if (train_status)
									{
										cout<<"Training successful"<<endl;
										return 1;
									}

							}
							catch (cv::Exception &e)
							{
							
								cout << " Exception Handled in FACE RECOGNITION MODULE :" << e.what() << endl;
								
							}
						}
			}
}
	
}
	return 0;
}


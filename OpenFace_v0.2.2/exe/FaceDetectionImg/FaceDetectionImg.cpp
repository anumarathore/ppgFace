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
// FaceDetectionImg.cpp : Defines the entry point for the console application for detecting landmarks in images.

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

#include <FaceAnalyser.h>
#include <GazeEstimation.h>

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

// Useful utility for creating directories for storing the output files
void create_directory_from_file(string output_path)
{

	// Creating the right directory structure

	// First get rid of the file
	auto p = boost::filesystem::path(boost::filesystem::path(output_path).parent_path());

	if (!p.empty() && !boost::filesystem::exists(p))
	{
		bool success = boost::filesystem::create_directories(p);
		if (!success)
		{
			cout << "Failed to create a directory... " << p.string() << endl;
		}
	}
}

void write_out_landmarks(const string& outfeatures, const LandmarkDetector::CLNF& clnf_model, const cv::Vec6d& pose,int faceID)
{
	create_directory_from_file(outfeatures);
	std::ofstream featuresFile;
	if (faceID == 1) 
		featuresFile.open(outfeatures,ios::out);
		else
		featuresFile.open(outfeatures,ios::app);


	if (featuresFile.is_open())
	{
		// open the { 
		//				"results":[
		if (faceID == 1) 
		{
		featuresFile<<"{"<<endl<<"\"results\":["<<endl;
		featuresFile<<"{\"id\":"<<faceID<<","<<endl;
}
			if (faceID > 1)
				featuresFile<<",{\"id\":\""<<faceID<<"\","<<endl;

			featuresFile<<"\"location\":{"<<endl;
			cv::Rect roi=clnf_model.GetBoundingBox();
			featuresFile<<"\"left\":"<<roi.tl().x<<","<<endl;
			featuresFile<<"\"top\":"<<roi.tl().y<<","<<endl;
			featuresFile<<"\"height\":"<<roi.height<<","<<endl;
			featuresFile<<"\"width\":"<<roi.width<<","<<endl;
			featuresFile<<"},"<<endl;
			int n = clnf_model.patch_experts.visibilities[0][0].rows;
			featuresFile << "\"npoints\":" << n << ","<<endl;
			featuresFile<<"\"LandmarkPoints\":["<<endl;
		for (int i = 0; i < n; ++i)
		{
			// Use matlab format, so + 1
			featuresFile <<"{\"x\":"<<clnf_model.detected_landmarks.at<double>(i) + 1 << ",\"y\": " << clnf_model.detected_landmarks.at<double>(i + n) + 1 << "}";
			if(i <n-1) featuresFile<<","<<endl;
			else 
				featuresFile<<endl;
		}
		featuresFile << "]," << endl;

		// Do the pose and eye gaze if present as well
		featuresFile << "\"pose\":{"<<endl;
		featuresFile << "\"eul_x\":"<<pose[3] << "," << endl;
		featuresFile<<"\"eul_y\":" << pose[4] << "," <<endl;
		featuresFile<<"\"eul_z\":"<< pose[5] << endl;
		featuresFile << "}" << endl;
		featuresFile<<"}"<<endl; //closing { "id"


		featuresFile.close();
	}
}
void create_display_image(const cv::Mat& orig, cv::Mat& display_image, LandmarkDetector::CLNF& clnf_model)
{
	// preparing the visualisation image
	display_image = orig.clone();		
						
}

int main (int argc, char **argv)
{
	cv::RNG rng(12345);	
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
	//cout << "Loading the model" << endl;
	LandmarkDetector::CLNF clnf_model(det_parameters.model_location);
	//cout << "Model loaded" << endl;
	
	cv::CascadeClassifier classifier(det_parameters.face_detector_location);
	dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();
	bool visualise = !det_parameters.quiet_mode;

	// Do some image loading
	for(size_t i = 0; i < files.size(); i++)
	{
		string file = files.at(i);

		// Loading image
		cv::Mat read_image = cv::imread(file, -1);

		if (read_image.empty())
		{
			cout << "Could not read the input image" << endl;
			return 1;
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


		// if no pose defined we just use a face detector
		if(bounding_boxes.empty())
		{
			
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

			// Detect landmarks around detected faces
			int face_det = 0;
			// displaying detected landmarks
			cv::Mat display_image;
			create_display_image(read_image, display_image, clnf_model);
			string outfeatures="";
			// perform landmark detection for every face detected
			for(size_t face=0; face < face_detections.size(); ++face)
			{
				// if there are multiple detections go through them
				bool success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, depth_image, face_detections[face], clnf_model, det_parameters);
			if(success)
				{
					face_det++;
				// Estimate head pose and eye gaze				
				cv::Vec6d headPose = LandmarkDetector::GetCorrectedPoseWorld(clnf_model, fx, fy, cx, cy);

				// Writing out the detected landmarks (in an OS independent manner)
			// Writing out the detected landmarks (in an OS independent manner)
				if(!output_landmark_locations.empty())
				{
					char name[100];
					// append detection number (in case multiple faces are detected)
					sprintf(name, "_meta");

					// Construct the output filename
					boost::filesystem::path slash("/");
					std::string preferredSlash = slash.make_preferred().string();

					boost::filesystem::path out_feat_path(output_landmark_locations.at(i));
					boost::filesystem::path dir = out_feat_path.parent_path();
					boost::filesystem::path fname = out_feat_path.filename().replace_extension("");
					boost::filesystem::path ext = out_feat_path.extension();
					outfeatures = dir.string() + preferredSlash + fname.string() + string(name) + ext.string();
					write_out_landmarks(outfeatures, clnf_model, headPose, face_det);
				}

				cv::Rect roi=clnf_model.GetBoundingBox();
				cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
				rectangle(display_image,roi,color,0.5,1);
				LandmarkDetector::Draw(display_image, clnf_model);


				
					
				}

			}
			if(face_det > 0){
			std::ofstream featuresFile;
		featuresFile.open(outfeatures,ios::app);
		featuresFile<<"]"<<endl<<"}"<<endl;
		featuresFile.close();
}
				// Saving the display images (in an OS independent manner)
				if(!output_images.empty() && (face_det > 0))
				{
					string outimage = output_images.at(i);
					if(!outimage.empty())
					{
						char name[100];
						sprintf(name, "_annotated");

						boost::filesystem::path slash("/");
						std::string preferredSlash = slash.make_preferred().string();

						// append detection number
						boost::filesystem::path out_feat_path(outimage);
						boost::filesystem::path dir = out_feat_path.parent_path();
						boost::filesystem::path fname = out_feat_path.filename().replace_extension("");
						boost::filesystem::path ext = out_feat_path.extension();
						outimage = dir.string() + preferredSlash + fname.string() + string(name) + ext.string();
						create_directory_from_file(outimage);
						//cout<<"output img dir :"<<outimage<<endl;
						bool write_success = cv::imwrite(outimage, display_image);
						//imshow("",display_image); cv::waitKey(5);
						
						if (!write_success)
						{
							cout << "Could not output a processed image" << endl;
							return 1;
						}

					}

				}
		}
	}
	
	return 0;
}


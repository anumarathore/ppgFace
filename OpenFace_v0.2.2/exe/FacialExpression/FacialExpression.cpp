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
// FaceLandmarkImg.cpp : Defines the entry point for the console application for detecting landmarks in images.

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
#include "PredictExpression.h"
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
void write_meta_file(const string& outfeatures, const LandmarkDetector::CLNF& clnf_model, string PredictedExpressionName,int faceID)
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
			featuresFile<<"\"Predicted Expression\":\""<<PredictedExpressionName<<"\""<<endl;
		featuresFile<<"}"<<endl; //closing { "id"


		featuresFile.close();
	}
}
void create_display_image(const cv::Mat& orig, cv::Mat& display_image, LandmarkDetector::CLNF& clnf_model)
{
	
	// Draw head pose if present and draw eye gaze as well

	// preparing the visualisation image
	display_image = orig.clone();		

	// Creating a display image			
	cv::Mat xs = clnf_model.detected_landmarks(cv::Rect(0, 0, 1, clnf_model.detected_landmarks.rows/2));
	cv::Mat ys = clnf_model.detected_landmarks(cv::Rect(0, clnf_model.detected_landmarks.rows/2, 1, clnf_model.detected_landmarks.rows/2));
	double min_x, max_x, min_y, max_y;

	cv::minMaxLoc(xs, &min_x, &max_x);
	cv::minMaxLoc(ys, &min_y, &max_y);

	double width = max_x - min_x;
	double height = max_y - min_y;

	int minCropX = max((int)(min_x-width/3.0),0);
	int minCropY = max((int)(min_y-height/3.0),0);

	int widthCrop = min((int)(width*5.0/3.0), display_image.cols - minCropX - 1);
	int heightCrop = min((int)(height*5.0/3.0), display_image.rows - minCropY - 1);

	double scaling = 350.0/widthCrop;
	
	// first crop the image
	display_image = display_image(cv::Rect((int)(minCropX), (int)(minCropY), (int)(widthCrop), (int)(heightCrop)));
		
	// now scale it
	cv::resize(display_image.clone(), display_image, cv::Size(), scaling, scaling);

	// Make the adjustments to points
	xs = (xs - minCropX)*scaling;
	ys = (ys - minCropY)*scaling;

	cv::Mat shape = clnf_model.detected_landmarks.clone();

	xs.copyTo(shape(cv::Rect(0, 0, 1, clnf_model.detected_landmarks.rows/2)));
	ys.copyTo(shape(cv::Rect(0, clnf_model.detected_landmarks.rows/2, 1, clnf_model.detected_landmarks.rows/2)));

	// Do the shifting for the hierarchical models as well
	for (size_t part = 0; part < clnf_model.hierarchical_models.size(); ++part)
	{
		cv::Mat xs = clnf_model.hierarchical_models[part].detected_landmarks(cv::Rect(0, 0, 1, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2));
		cv::Mat ys = clnf_model.hierarchical_models[part].detected_landmarks(cv::Rect(0, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2, 1, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2));

		xs = (xs - minCropX)*scaling;
		ys = (ys - minCropY)*scaling;

		cv::Mat shape = clnf_model.hierarchical_models[part].detected_landmarks.clone();

		xs.copyTo(shape(cv::Rect(0, 0, 1, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2)));
		ys.copyTo(shape(cv::Rect(0, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2, 1, clnf_model.hierarchical_models[part].detected_landmarks.rows / 2)));

	}

	LandmarkDetector::Draw(display_image, clnf_model);
						
}

int main (int argc, char **argv)
{
	cv::RNG rng(12345);
	PredictExpression *expressionPredicter = new PredictExpression;
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
	int frame_count = 1;

	double sim_scale = 0.7;
	int sim_size = 112;
	bool grayscale = false;
	bool video_output = false;
	bool rigid = false;
	int num_hog_rows;
	int num_hog_cols;

	// The modules that are being used for tracking
	//cout << "Loading the model" << endl;
	LandmarkDetector::CLNF clnf_model(det_parameters.model_location);
	//cout << "Model loaded" << endl;
	
	cv::CascadeClassifier classifier(det_parameters.face_detector_location);
	dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();

	// Loading the AU prediction models
	string au_loc = "AU_predictors/AU_all_static.txt";

	if (!boost::filesystem::exists(boost::filesystem::path(au_loc)))
	{
		boost::filesystem::path loc = boost::filesystem::path(arguments[0]).parent_path() / au_loc;

		if (boost::filesystem::exists(loc))
		{
			au_loc = loc.string();
		}
		else
		{
			cout << "Can't find AU prediction files, exiting" << endl;
			return 1;
		}
	}

	// Used for image masking for AUs
	string tri_loc;
	if (boost::filesystem::exists(boost::filesystem::path("model/tris_68_full.txt")))
	{
		std::ifstream triangulation_file("model/tris_68_full.txt");
		tri_loc = "model/tris_68_full.txt";
	}
	else
	{
		boost::filesystem::path loc = boost::filesystem::path(arguments[0]).parent_path() / "model/tris_68_full.txt";
		tri_loc = loc.string();

		if (!exists(loc))
		{
			cout << "Can't find triangulation files, exiting" << endl;
			return 1;
		}
	}

	FaceAnalysis::FaceAnalyser face_analyser(vector<cv::Vec3d>(), 0.7, 112, 112, au_loc, tri_loc);
double time_stamp = 0;
	time_stamp = (double)frame_count * (1.0 / 30.0);
	bool visualise = !det_parameters.quiet_mode;

	// Do some image loading
	for(size_t i = 0; i < files.size(); i++)
	{
		
		string file = files.at(i);
			std::ofstream feature_vector;
			char name[100];
			sprintf(name,"_features.txt");
			boost::filesystem::path slash("/");
			std::string preferredSlash = slash.make_preferred().string();
			boost::filesystem::path out_feat_path(output_images.at(i));
			boost::filesystem::path in_file_path(file);
			boost::filesystem::path dir = out_feat_path.parent_path();
			boost::filesystem::path fname = in_file_path.filename().replace_extension("");
			boost::filesystem::path ext = out_feat_path.extension();
			string featuresfilename = dir.string() + preferredSlash + fname.string() + string(name);
			cout<<"featuresfilename : "<<featuresfilename<<endl;
			string probability_output=dir.string() + preferredSlash + fname.string() + "_probability_output.txt";
			feature_vector.open(featuresfilename.c_str(),ios::out);

		// Loading image
		cv::Mat read_image = cv::imread(file, -1);

		if (read_image.empty())
		{
			cout << "Could not read the input image" << endl;
			return 1;
		}


		// Loading depth file if exists (optional)
		cv::Mat_<float> depth_image;

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
		cv::Mat display_image;
		display_image=read_image.clone();

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
			// perform landmark detection for every face detected
			bool success=false;
			string outfeatures="";


			for(size_t face=0; face < face_detections.size(); ++face)
			{
				// if there are multiple detections go through them
				 success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, depth_image, face_detections[face], clnf_model, det_parameters);

				if(success)
				{
				face_det++;
				}
				// Estimate head pose and eye gaze				
				cv::Vec6d headPose = LandmarkDetector::GetCorrectedPoseWorld(clnf_model, fx, fy, cx, cy);

				// displaying detected landmarks
				double confidence = 0.5 * (1.0 - clnf_model.detection_certainty);
				if (confidence >0.9 && clnf_model.detection_success) {
					// Do face alignment
					try {
						cv::Mat sim_warped_img;
						cv::Mat_<double> hog_descriptor;
						face_analyser.AddNextFrame(read_image, clnf_model, time_stamp, false, false);

						// But only if needed in output
						//if (!output_similarity_align.empty() || hog_output_file.is_open() || output_AUs)
						//face_analyser.AddNextFrame(read_image, clnf_model, time_stamp, false, false);
							//face_analyser.GetLatestAlignedFace(sim_warped_img);

							//	if (true)
							//{
							//	cv::imshow("sim_warp", sim_warped_img);
							//}
							auto aus_reg = face_analyser.GetCurrentAUsReg();

							vector<string> au_reg_names = face_analyser.GetAURegNames();
							std::sort(au_reg_names.begin(), au_reg_names.end());

							string AU = "";
							// write out ar the correct index
							feature_vector << 99 << " ";
							int iter = 1;
							for (string au_name : au_reg_names)
							{
								for (auto au_reg : aus_reg)
								{
									if (au_name.compare(au_reg.first) == 0)
									{
										feature_vector << iter++ << ":" << au_reg.second << " ";
										break;
									}
								}
							}
							feature_vector << endl;
							if (aus_reg.size() == 0)
							{
								for (size_t p = 0; p < face_analyser.GetAURegNames().size(); ++p)
								{
									feature_vector << ", 0";
									AU = "no AUs found";
								}
							}


						 // run expressionPrediction
							feature_vector.close();
							//call Expression Prediction SVM here! 
							int PredictedExpressionLabel = expressionPredicter->RunPrediction(featuresfilename,probability_output);
							string PredictedExpressionName = expressionPredicter->expressionNames[PredictedExpressionLabel - 1];
							cout << "Predicted Expression " << PredictedExpressionName << endl;
							cv::Scalar colour = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
							cv::putText(display_image, PredictedExpressionName.c_str(), cv::Point(face_detections[face].tl().x+5,face_detections[face].tl().y-2), CV_FONT_HERSHEY_SIMPLEX, 0.5, colour);
							rectangle(display_image, face_detections[face],colour, 1.5, 8);
					
							// witing meta data into the file
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
					write_meta_file(outfeatures, clnf_model,PredictedExpressionName,face_det);
					}
					
					}
					catch (std::exception &e)
					{
						cout << "Exception in Expression module : " << e.what() << endl;
						continue;
					}
				}
				//cout << face_detections[face] << endl;
				/*	cv::imshow("", display_image); cv::waitKey(1);*/
					//LandmarkDetector::Draw(display_image, clnf_model);
					
				//}

				// Saving the display images (in an OS independent manner)
			if(face_det > 1){
			std::ofstream featuresFile;
		featuresFile.open(outfeatures,ios::app);
		featuresFile<<"]"<<endl<<"}"<<endl;
		featuresFile.close();	
				}

			}

			// Saving the display images (in an OS independent manner)
				if(!output_images.empty() && success)
				{
					string outimage = output_images.at(i);
					if(!outimage.empty())
					{
						char name[100];
						sprintf(name, "_exp");

						boost::filesystem::path slash("/");
						std::string preferredSlash = slash.make_preferred().string();

						// append detection number
						boost::filesystem::path out_feat_path(outimage);
						boost::filesystem::path dir = out_feat_path.parent_path();
						boost::filesystem::path fname = out_feat_path.filename().replace_extension("");
						boost::filesystem::path ext = out_feat_path.extension();
						outimage = dir.string() + preferredSlash + fname.string() + string(name) + ext.string();
						create_directory_from_file(outimage);
						bool write_success = cv::imwrite(outimage, display_image);	
						
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


#include "LandmarkCoreIncludes.h"
// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

#include <dlib/image_processing/frontal_face_detector.h>

#include <tbb/tbb.h>

#include <FaceAnalyser.h>
#include <GazeEstimation.h>
//opencv
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <unistd.h>
#include <fstream>

using namespace cv;
using namespace std;

#define min_face_size_factor 0.01  // minimum size of face wrt image dimensions
#define max_face_size_factor 0.30  // maximum size of face wrt image dimensions

static void write_purpose()
{
    cout
        << "------------------------------------------------------------------------------" << endl
        << "This program gives the predominant color of the upper part of the dress" << endl
        << "./runfilename start_frame_number end_frame_number "      << endl
        << "Example:" << endl
        << "./runfilename 1 10" << endl
	<< "Enter correct option....Enter R for right W for wrong" << endl
        << "------------------------------------------------------------------------------" << endl
        << endl;
}
 
bool face_size_comparator(const Rect l, const Rect r)
{
    return l.height*l.width > r.width*r.height;
}

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
 
int detect_face(vector<string> arguments, vector<cv::Rect_<double> >& faces){

	cv::RNG rng(12345);	
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
	//cout<<"here"<<"-----------"<<files.size()<<"-----------"<<arguments.size()<<"-----------"<<depth_files.size()<<endl;
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
			//vector<cv::Rect_<double> > face_detections;

			if(det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
			{
				vector<double> confidences;
				LandmarkDetector::DetectFacesHOG(faces, grayscale_image, face_detector_hog, confidences);
			}
			else
			{
				LandmarkDetector::DetectFaces(faces, grayscale_image, classifier);
			}
			// Detect landmarks around detected faces
			int face_det = 0;
			// perform landmark detection for every face detected
			for(size_t face=0; face < faces.size(); ++face)
			{
				// if there are multiple detections go through them
				bool success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, depth_image, faces[face], clnf_model, det_parameters);
			if(success)
				{
					face_det++;		
				}

			}
		}
	}
	if (faces.size() > 0)
	{
		//cout << "Number of faces detected: " << faces.size() << endl;
		return 1;
	}

	else
	{
		//cout << "No face detected!" << endl;
		return 0;
	}
}

int main(int argc,char **argv)
{	
	//Convert arguments to more convenient vector form
	vector<string> arguments = get_arguments(argc, argv);

	string imgfile="default.jpg";
	imgfile=argv[2];



	Mat dst,b,hsv,img,gray_img;
	int a,h,w,n,x1,x2,y1,y2,count,z,c,max,pi,yy;
	char color[30],tshirt[30];
	string filename1,filename2,filename3;
	float hsum,ssum,vsum,pixels,accuracy,mean[3],value[3],value1[3],sigma[3],variance[3],hue,sat,val,fs,th;
	
	//vector<Rect> faces; 
	vector<cv::Rect_<double> > faces;
	string answer;
	Rect torso_roi;
	
	Mat input_image_gray,input_torso;
	int torso_roi_x,torso_roi_y,torso_roi_width,torso_roi_height;

	z = c = 0;
	value[0] = value[1] = value[2] = 0;
	value1[0] = value1[1] = value1[2] = 1;
	sigma[0] = sigma[1] = sigma[2] = 0;
	hsum = ssum = vsum = pixels = count = max = pi = 0;
/*********************************checking for all data one by one*********************************/
	
		img = cv::imread(imgfile,1);
		if (img.empty()) 
       	 	{
        		cout << "Error : Image cannot be loaded..!!" << endl;
        		exit(-1);
        	}

		//cout << tshirt << endl;
		cvtColor(img,hsv,CV_BGR2HSV);
		Size size(480,640);
		resize(hsv,dst,size);
		
		 filename1 = "huedata.txt";
		 filename2 = "satdata.txt";
		 filename3 = "valdata.txt";

		ofstream fout1(filename1.c_str());
		ofstream fout2(filename2.c_str());
		ofstream fout3(filename3.c_str());

	/********************************check for presence of face ****************************************/
	cvtColor(img, input_image_gray, CV_BGR2GRAY);
	faces.clear();
	if (detect_face(arguments,faces))
		{ 
		c = 1;
	//	cout << "Approximating Torso Region w.r.t Face." << endl;
		sort(faces.begin(), faces.end(), face_size_comparator);
try{
cv::waitKey(5);
		Point pt1(faces[0].x+faces[0].width,faces[0].y+faces[0].height); 
		Point pt2(faces[0].x,faces[0].y); 
		//Mat FaceROI = gray_img(faces[0]); 
		rectangle(img,pt1,pt2,cvScalar(0,255,0),2,8,0);
		torso_roi_x = faces[0].x - faces[0].width/3;
		torso_roi_y = faces[0].y + 8*faces[0].height/5;
		torso_roi_width =1.5*faces[0].width;
		torso_roi_height =  0.5*faces[0].height;
		torso_roi = Rect(torso_roi_x, torso_roi_y, torso_roi_width, torso_roi_height);
		rectangle(img, torso_roi, Scalar(200, 200, 0), 2, 8, 0);
		//imshow("read_image",img);
		//waitKey(0);
		input_torso = img(torso_roi);
		putText(img,"ROI" , Point(torso_roi_x - 10, torso_roi_y - 10), 1, fs, Scalar(0, 255, 255), th, 8, false);
		x1 = faces[0].x - faces[0].width/3;
		x2 = faces[0].x - faces[0].width/3 + 1.5*faces[0].width;
		y1 = faces[0].y + 8*faces[0].height/5;
		y2 = faces[0].y + 8*faces[0].height/5 + 0.5*faces[0].height;
		//cout << x1 << " "<< x2 << " "<< y1 << " "<< y2 << endl;
		//cout << "Input torso roi size: " << input_torso.size() << endl;
		}
		catch(cv::Exception &e)
		{
			//cout<<"exception caught at "<<__LINE__<<" in "<<__FILE__<<" in function  "<<__FUNCTION__<<" : "<<e.what()<<endl;
			cout<<"error"<<endl;
			exit(-1);
		}
	}
	else
	{
		cout<<"unable to detect the face"<<endl;
		exit(-2);
	}
	
/*****************finding frequency of the pixels and storing them in a file************************/
		
	if (c)
	{
		max = 0;
		for(int j=0;j<=180;j++)
		{
			int varh = 0;
			for(int x=x1;x<x2;x++)
			{
				for(int y=y1;y<=y2;y++)
				{
					Vec3b hsv1=hsv.at<Vec3b>(y,x);
					int h = hsv1.val[0];
					if(h == j)
					varh++;
				}
			}
			if(varh > max)
			{
				max = varh;
				pi = j;
			}
				
			fout1 << j << "," << varh << endl;
			varh = 0;
		}
	//	cout << "max occuring pixel value is " << pi << endl; 
		for(int k=0;k<=256;k++)
		{
			int vars = 0;
			int varv = 0;
			for(int x=x1;x<x2;x++)
			{
				for(int y=y1;y<=y2;y++)
				{
					Vec3b hsv1=hsv.at<Vec3b>(y,x);	
					int s = hsv1.val[1];
					int v = hsv1.val[2];
					if(s == k)
					vars++;
					if(v == k)
					varv++;
				}
			}
			fout2 << k << "," << vars << endl;
			fout3 << k << "," << varv << endl;
		}

/***********obtaining HSV values of the pixels in patch and finding mean of it ********************/

		for(int x=x1;x<x2;x++)
		{
			for(int y=y1;y<=y2;y++)
			{
				Vec3b hsv1=hsv.at<Vec3b>(y,x);
				int H=hsv1.val[0];
				int S=hsv1.val[1];
				int V=hsv1.val[2];
				
				hsum = hsum + H;
				ssum = ssum + S;
				vsum = vsum + V;
				pixels = pixels + 1;
				z++;

			}
		}
		z = 0;
		mean[0] = hsum / pixels;
		mean[1] = ssum / pixels;
		mean[2] = vsum / pixels;

		//cout << "mean of hue = " << mean[0] << endl;
		//cout << "mean of saturation = " << mean[1] << endl;
		//cout << "mean of value = " << mean[2] << endl; 
		
		hue = mean[0];
		sat = mean[1];
		val = mean[2];

/***********obtaining HSV values of the pixels in a patch and finding variance of it***************/
	
	for(int x=x1;x<x2;x++)
	{
		for(int y=y1;y<=y2;y++)
		{
			Vec3b hsv1=hsv.at<Vec3b>(y,x);
			int H=hsv1.val[0];
			int S=hsv1.val[1];
			int V=hsv1.val[2];
			
			value[0] = H - mean[0];
			value[1] = S - mean[1];
			value[2] = V - mean[2];
			
			value1[0] = value[0] * value[0];
			value1[1] = value[1] * value[1];
			value1[2] = value[2] * value[2];
			
			value1[0] = value1[0] / pixels;
			value1[1] = value1[1] / pixels;
			value1[2] = value1[2] / pixels;
			
			sigma[0] = sigma[0] + value1[0];
			sigma[1] = sigma[1] + value1[1];
			sigma[2] = sigma[2] + value1[2];
		}
	}

	variance[0] = sqrt(sigma[0] / pixels);
	variance[1] = sqrt(sigma[1] / pixels);
	variance[2] = sqrt(sigma[2] / pixels);
	
/**************************setting rule for detecting the color************************************/
		
		if(val > 240)
		{
			sprintf(color,"T-shirt is white");
			//putText(img,color, Point(25,yy) , FONT_HERSHEY_SIMPLEX, fs, Scalar(0,255,0), th,8,false);
			//cout << "------T-shirt is white------" << endl;
			cout<<"white"<<endl;
		}
	
		else if(sat < 70 && val < 70)
		{
			sprintf(color,"T-shirt is black");
			//cout << fs << th << endl;
			//putText(img,color, Point(25,yy) , FONT_HERSHEY_SIMPLEX, fs , Scalar(0,255,0), th,8,false);
			//cout << "------T-shirt is black------" << endl;
			cout<<"black"<<endl;
		}

		else if((pi >=0 && pi <= 10) || (pi >= 170 && pi <=180))
		{	
			sprintf(color,"T-shirt is red");
			//putText(img,color, Point(25,yy) , FONT_HERSHEY_SIMPLEX, fs , Scalar(0,255,0), th,8,false );
			//cout << "------T-shirt is red------" << endl;
			cout<<"red"<<endl;
		}
	
		else if(pi > 10 && pi <= 18) 
		{
			sprintf(color,"T-shirt is orange");
			//putText(img,color, Point(25,yy) , FONT_HERSHEY_SIMPLEX, fs, Scalar(0,255,0), th,8,false );
			//cout << "------T-shirt is orange------" << endl;
			cout<<"orange"<<endl;
		}
	
		else if(pi > 18 && pi <= 35) 
		{
			sprintf(color,"T-shirt is yellow");
			//putText(img,color, Point(25,yy) , FONT_HERSHEY_SIMPLEX, fs, Scalar(0,255,0), th,8,false );
			//cout << "------T-shirt is yellow------" << endl;
			cout<<"yellow"<<endl;
		}
		
		else if(pi > 35 && pi <= 92) 
		{
			sprintf(color,"T-shirt is green");
			//putText(img,color, Point(25,yy) , FONT_HERSHEY_SIMPLEX, fs, Scalar(0,255,0), th,8,false );
			//cout << "------T-shirt is green------" << endl;
			cout<<"green"<<endl;
		}
	
		else if(pi > 92 && pi <= 130) 
		{
			sprintf(color,"T-shirt is blue");
			//putText(img,color, Point(25,yy) , FONT_HERSHEY_SIMPLEX, fs, Scalar(0,255,0), th,8,false );
			//cout << "------T-shirt is blue------" << endl;
			cout<<"blue"<<endl;
		}
		
		else if(pi > 130 && pi <= 145) 
		{
			sprintf(color,"T-shirt is purple");
			//putText(img,color, Point(25,yy) , FONT_HERSHEY_SIMPLEX, fs, Scalar(0,255,0), th,8,false );
			//cout << "------T-shirt is purple------" << endl;
			cout<<"purple"<<endl;
		}

		else if(pi > 145 && pi < 172) 
		{
			sprintf(color,"T-shirt is pink");
			//putText(img,color, Point(25,yy) , FONT_HERSHEY_SIMPLEX, fs, Scalar(0,255,0), th,8,false );
			//cout << "------T-shirt is pink------" << endl;
			cout<<"pink"<<endl;
		}

		else 
		{
			sprintf(color,"Cannot say about the color");
			//putText(img,color, Point(25,yy) , FONT_HERSHEY_SIMPLEX, fs, Scalar(0,255,0), th,8,false);
			//cout << "------Cannot say about the color------" << endl;
			cout<<"undefined"<<endl;
		}
		
		
/*******************************showing the image and clearing mat and variables*******************/
		
		// //string file = "/home/arti/data/try/working_data1/test" + to_string(a) + ".jpg";
		// //imwrite(file,img);
		// namedWindow("Torso ROI",CV_WINDOW_NORMAL);
		// imshow("Torso ROI", img);
		// waitKey(10);

		// cout << "Detected color right or wrong?....Enter R for right W for wrong" << endl;
		// cin >> answer;
		// while(answer.compare("R") && answer.compare("W"))
		// {
		// 	cin >> answer;
		// }
		// cout << "color detection : " << answer << endl;
		// if(!(answer.compare("R")))
		// {
		// 	count = count + 1;
		// }
		
		// sleep(2);
		
		
	}
		// cout << endl;
		// cout << endl;
		// hsum = ssum = vsum = pixels = 0;
		// value[0] = value[1] = value[2] = 0;
		// value1[0] = value1[1] = value1[2] = 1;
		// sigma[0] = sigma[1] = sigma[2] = 0;
		// c = 0;
		
		// img.setTo(Scalar::all(0));
		// dst.setTo(Scalar::all(0));
		// hsv.setTo(Scalar::all(0));

	
		// cout << count << "/" << n << " " << "correct" << endl; 
		// accuracy = ((float(count)/ float(n))*100);
		// cout << "accuracy = " << accuracy << "%" << endl;
		// cout << endl;
	return 0;
}
	
	

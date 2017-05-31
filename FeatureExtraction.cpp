// FeatureExtraction.cpp : Defines the entry point for the feature extraction console application.
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include "opencv2/videoio.hpp"
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <sstream>
#include <math.h>


// System includes
#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>

// Local includes
#include "LandmarkCoreIncludes.h"

#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>

// //SQL
#include <cppconn/driver.h>
#include <cppconn/connection.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>
#include <cppconn/resultset.h>
#include <cppconn/metadata.h>
#include <cppconn/resultset_metadata.h>
#include <cppconn/exception.h>
#include <cppconn/warning.h>

#include "mysql_connection.h"


using namespace cv;
using namespace std;
using namespace std;
using namespace sql;
using namespace boost::filesystem;

//global variables
Driver *driver;
auto_ptr<Connection> con;
auto_ptr<Connection> con2;
Statement *stmt;
ResultSet *res;
int datacount=0;
auto_ptr<PreparedStatement> prep_stmt;
Savepoint *savept;
string url="tcp://127.0.0.1:3306";
string user="root";
string password="wipro123";
string database="kioskdb";
string statement,action_name="",timestamp="";


#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

#define PI 3.14159265


//SQL Functions
bool connectDatabase(){

    try {
               driver = get_driver_instance();

                /* create a database connection using the Driver */
               con.reset(driver->connect(url, user, password));

                /* alternate syntax using auto_ptr to create the db connection */
                //auto_ptr <Connection> con (driver -> connect(url, user, password));

                /* turn off the autocommit */
                con -> setAutoCommit(0);

                cout << "\nDatabase connection\'s autocommit mode = " << con -> getAutoCommit() << endl;

                /* select appropriate database schema */
                con -> setSchema(database);
                con->commit();
                                return true;

                            } catch (SQLException &e) {
                                cout << "ERROR: SQLException in " << __FILE__;
                                cout << " (" << __func__ << ") on line " << __LINE__ << endl;
                                cout << "ERROR: " << e.what();
                                cout << " (MySQL error code: " << e.getErrorCode();
                                cout << ", SQLState: " << e.getSQLState() << ")" << endl;

                                if (e.getErrorCode() == 1047) {
                                    /*
                                     Error: 1047 SQLSTATE: 08S01 (ER_UNKNOWN_COM_ERROR)
                                     Message: Unknown command
                                     */
                                    cout	<< "\nYour server does not seem to support Prepared Statements at all. ";
                                    cout << "Perhaps MYSQL < 4.1?" << endl;
                                }

                                return false;

                        } catch (std::runtime_error &e) {

                            cout << "ERROR: runtime_error in " << __FILE__;
                            cout << " (" << __func__ << ") on line " << __LINE__ << endl;
                            cout << "ERROR: " << e.what() << endl;

                                return false;
                        }

}


bool insertAction(string action){


    try {
        statement.clear();
        statement = "insert INTO action_status(action) values('";
        statement.append(action); statement.append("');");
        prep_stmt.reset(con->prepareStatement(statement));

        prep_stmt->execute();
        con->commit();
        return true;
    }
    catch (...){
        cout << "ERROR: runtime_error in " << __FILE__;
        cout << " (" << __func__ << ") on line "
        << __LINE__ << endl;
        cout << "ERROR: " << endl;
        return false;
    }
}


//Open Face pre-written func
vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	// First argument is reserved for the name of the executable
	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

// Useful utility for creating directories for storing the output files
void create_directory_from_file(string output_path)
{

	// Creating the right directory structure
	
	// First get rid of the file
	auto p = path(path(output_path).parent_path());

	if(!p.empty() && !boost::filesystem::exists(p))		
	{
		bool success = boost::filesystem::create_directories(p);
		if(!success)
		{
			cout << "Failed to create a directory... " << p.string() << endl;
		}
	}
}

void create_directory(string output_path)
{

	// Creating the right directory structure
	auto p = path(output_path);

	if(!boost::filesystem::exists(p))		
	{
		bool success = boost::filesystem::create_directories(p);
		
		if(!success)
		{
			cout << "Failed to create a directory..." << p.string() << endl;
		}
	}
}

void get_output_feature_params(vector<string> &output_similarity_aligned, vector<string> &output_hog_aligned_files, double &similarity_scale, 
	int &similarity_size, bool &grayscale, bool &rigid, bool& verbose, 
	bool &output_2D_landmarks, bool &output_3D_landmarks, bool &output_model_params, bool &output_pose, bool &output_AUs, bool &output_gaze,
	vector<string> &arguments)
{
	output_similarity_aligned.clear();
	output_hog_aligned_files.clear();

	bool* valid = new bool[arguments.size()];

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
	}

	string input_root = "";
	string output_root = "";

	// First check if there is a root argument (so that videos and outputs could be defined more easilly)
	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-root") == 0)
		{
			input_root = arguments[i + 1];
			output_root = arguments[i + 1];
			i++;
		}
		if (arguments[i].compare("-inroot") == 0)
		{
			input_root = arguments[i + 1];
			i++;
		}
		if (arguments[i].compare("-outroot") == 0)
		{
			output_root = arguments[i + 1];
			i++;
		}
	}

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-simalign") == 0)
		{
			output_similarity_aligned.push_back(output_root + arguments[i + 1]);
			create_directory(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-hogalign") == 0)
		{
			output_hog_aligned_files.push_back(output_root + arguments[i + 1]);
			create_directory_from_file(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-verbose") == 0)
		{
			verbose = true;
		}
		else if (arguments[i].compare("-rigid") == 0)
		{
			rigid = true;
		}
		else if (arguments[i].compare("-g") == 0)
		{
			grayscale = true;
			valid[i] = false;
		}
		else if (arguments[i].compare("-simscale") == 0)
		{
			similarity_scale = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-simsize") == 0)
		{
			similarity_size = stoi(arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		} 
		else if (arguments[i].compare("-no2Dfp") == 0)
		{
			output_2D_landmarks = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-no3Dfp") == 0)
		{
			output_3D_landmarks = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noMparams") == 0)
		{
			output_model_params = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noPose") == 0)
		{
			output_pose = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noAUs") == 0)
		{
			output_AUs = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noGaze") == 0)
		{
			output_gaze = false;
			valid[i] = false;
		}
	}

	for (int i = arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}

}
// Can process images via directories creating a separate output file per directory
void get_image_input_output_params_feats(vector<vector<string> > &input_image_files, bool& as_video, vector<string> &arguments)
{
	bool* valid = new bool[arguments.size()];
		
	for(size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-fdir") == 0) 
		{                    

			// parse the -fdir directory by reading in all of the .png and .jpg files in it
			path image_directory (arguments[i+1]); 

			try
			{
				 // does the file exist and is it a directory
				if (exists(image_directory) && is_directory(image_directory))   
				{
					
					vector<path> file_in_directory;                                
					copy(directory_iterator(image_directory), directory_iterator(), back_inserter(file_in_directory));

					// Sort the images in the directory first
					sort(file_in_directory.begin(), file_in_directory.end()); 

					vector<string> curr_dir_files;

					for (vector<path>::const_iterator file_iterator (file_in_directory.begin()); file_iterator != file_in_directory.end(); ++file_iterator)
					{
						// Possible image extension .jpg and .png
						if(file_iterator->extension().string().compare(".jpg") == 0 || file_iterator->extension().string().compare(".png") == 0)
						{																
							curr_dir_files.push_back(file_iterator->string());															
						}
					}

					input_image_files.push_back(curr_dir_files);
				}
			}
			catch (const filesystem_error& ex)
			{
				cout << ex.what() << '\n';
			}

			valid[i] = false;
			valid[i+1] = false;		
			i++;
		}
		else if (arguments[i].compare("-asvid") == 0) 
		{
			as_video = true;
		}
	}
	
	// Clear up the argument list
	for(int i=arguments.size()-1; i >= 0; --i)
	{
		if(!valid[i])
		{
			arguments.erase(arguments.begin()+i);
		}
	}

}

void output_HOG_frame(std::ofstream* hog_file, bool good_frame, const cv::Mat_<double>& hog_descriptor, int num_rows, int num_cols)
{

	// Using FHOGs, hence 31 channels
	int num_channels = 31;

	hog_file->write((char*)(&num_cols), 4);
	hog_file->write((char*)(&num_rows), 4);
	hog_file->write((char*)(&num_channels), 4);

	// Not the best way to store a bool, but will be much easier to read it
	float good_frame_float;
	if(good_frame)
		good_frame_float = 1;
	else
		good_frame_float = -1;

	hog_file->write((char*)(&good_frame_float), 4);

	cv::MatConstIterator_<double> descriptor_it = hog_descriptor.begin();

	for(int y = 0; y < num_cols; ++y)
	{
		for(int x = 0; x < num_rows; ++x)
		{
			for(unsigned int o = 0; o < 31; ++o)
			{

				float hog_data = (float)(*descriptor_it++);
				hog_file->write ((char*)&hog_data, 4);
			}
		}
	}
}

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;

void get_point(double x,double y,double slope,double d, double *x1,double *y1 ,int flag){

		if(flag==1){
			*x1=x+sqrt(d*d/(1+slope*slope));
			*y1=y-slope*(*x1-x);
		}
		else if(flag==2){
			*x1=x-sqrt(d*d/(1+slope*slope));
			*y1=y-slope*(*x1-x);
		}
		else if(slope>1000 || slope<-1000){
			*x1=x;
			*y1=y+d;
			//cout<<"infinite";
		}
		else if(slope>0){
			//cout<<"abhi";
			*x1=x-sqrt(d*d/(1+slope*slope));
			*y1=y-slope*(*x1-x);
		}
		else{
			//cout<<"SAdasdsad";
			*x1=x+sqrt(d*d/(1+slope*slope));
			*y1=y-slope*(*x1-x);	
		}

}

// Visualising the results
int result1=0,result2=0;
void visualise_tracking(std::ofstream* label_file,char a,int f_n,vector<string> &output_similarity_align,cv::Mat& captured_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, int frame_count, double fx, double fy, double cx, double cy)
{

	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = face_model.detection_certainty;
	bool detection_success = face_model.detection_success;

	double visualisation_boundary = 0.2;

	std::ofstream train_data;
    train_data.open("testing.txt", ios::out | ios::trunc );
    train_data<<"0 ";

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		//LandmarkDetector::Draw(captured_image, face_model);
		int idx = face_model.patch_experts.GetViewIdx(face_model.params_global, 0);
		cv::Mat_<double> shape2D = face_model.detected_landmarks;
		cv::Mat_<int> visibilities = face_model.patch_experts.visibilities[0][idx];

		int n = shape2D.rows/2;

		// Drawing feature points
		/*if(n >= 66)
		{
			for( int i = 1; i < n; ++i)
			{		
				if(visibilities.at<int>(i))
				{
					cv::Point featurePoint((int)shape2D.at<double>(i), (int)shape2D.at<double>(i +n));

					// A rough heuristic for drawn point size
					int thickness = (int)std::ceil(3.0* ((double)captured_image.cols) / 640.0);
					int thickness_2 = (int)std::ceil(1.0* ((double)captured_image.cols) / 640.0);

					
					cv::circle(captured_image, featurePoint, 1, cv::Scalar(0,0,255), thickness);
					cv::circle(captured_image, featurePoint, 1, cv::Scalar(255,0,0), thickness_2);
				}
			}
			//i=1 left most point
			//i=8 for lower point
			//i=15 for right most point
			// i=24 for right eyebrow top most point
			//i=27,28,29,20 nose
			// i=36  left eye left point
			//i=39 left eye right point
			//i=42 right eye left point
			//i=45 right eye right point
		}*/
		cv::Point ll((int)shape2D.at<double>(36), (int)shape2D.at<double>(36 +n));
		cv::Point lr((int)shape2D.at<double>(39), (int)shape2D.at<double>(39 +n));
		cv::Point rl((int)shape2D.at<double>(42), (int)shape2D.at<double>(42 +n));
		cv::Point rr((int)shape2D.at<double>(45), (int)shape2D.at<double>(45 +n));

		int facewidthapprox = abs((int)shape2D.at<double>(15)-(int)shape2D.at<double>(1));
		int faceheightapprox = abs((int)shape2D.at<double>(24+n)-(int)shape2D.at<double>(8+n));
		int lcnx = MAX(0,(int)shape2D.at<double>(1)-0.5*facewidthapprox);
		int lcny = MAX(0,(int)shape2D.at<double>(24)-1.2*faceheightapprox);
		int rcnx = MIN(captured_image.cols,(int)shape2D.at<double>(15)+0.5*facewidthapprox);
		int rcny = MIN(captured_image.rows,(int)shape2D.at<double>(8)+faceheightapprox);
		int lcx = MAX(0,(int)shape2D.at<double>(1)-1.5*facewidthapprox);
		int lcy = MAX(0,(int)shape2D.at<double>(24)-1.2*faceheightapprox);
		int rcx = MIN(captured_image.cols,(int)shape2D.at<double>(15)+1.5*facewidthapprox);
		int rcy = captured_image.rows;
		Rect roi_1(lcnx,lcny,rcnx-lcnx,rcny-lcny);
		Rect roi_2(lcx,lcy,rcx-lcx,rcy-lcy);
		//cout<<lcnx<<":"<<lcny<<":"<<rcnx<<":"<<rcny<<endl;
		//cout.flush();
		imwrite("/home/rahul/Pictures/ag.jpg",captured_image(roi_1));
		imwrite("/home/rahul/Pictures/cl.jpg",captured_image(roi_2));


		//cv::circle(captured_image,ll,1,cv::Scalar(255,0,255),2);
		//cv::circle(captured_image,rr,1,cv::Scalar(355,0,255),2);
		/*cv::circle(captured_image,rr,1,cv::Scalar(0,0,255),2);
		cv::circle(captured_image,rl,1,cv::Scalar(0,0,255),2);

		cout<<rr.y-rl.y<<endl;
		cout<<lr.y-ll.y<<endl;*/

		//cv::line(captured_image, ll, lr, cv::Scalar(0, 0, 255), 2);
		//cv::line(captured_image, rr, rl, cv::Scalar(0, 0, 255), 2);


		double eye_width1=abs(lr.x-ll.x);
		double eye_width2=abs(rr.x-rl.x);
		double final_width=0;
		if(eye_width1>eye_width2){
			final_width=eye_width1;
		}
		else{
			final_width=eye_width2;
		}

		if(final_width==0){
			cout<<"Could Not Detect Face"<<endl;
			return;
		}
		//cout<<"visualize 1"<<endl;


		double vis_certainty = detection_certainty;
		if (vis_certainty > 1)
			vis_certainty = 1;
		if (vis_certainty < -1)
			vis_certainty = -1;

		vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);

		// Draw it in reddish if uncertain, blueish if certain
		cx=cx+100;
		fy=fy+100;

		vector<std::pair<cv::Point, cv::Point>> edge_left;
		vector<std::pair<cv::Point, cv::Point>> edge_right;
		cv::Vec3d angle;

		cv::Vec6d pose_estimate;
		cv::Vec6d pose_estimate1;
			//if(use_world_coordinates)
		//cout<<"between vis 1 and 2"<<endl;
		//cout.flush();	
		pose_estimate = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);
			
		//cout<<"between vis 1 and 2,2"<<endl;
		//cout.flush();	
			
				//pose_estimate1 = LandmarkDetector::GetCorrectedPoseCamera(face_model, fx, fy, cx, cy);
		

		//angle= LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);
		//cout<<angle<<endl;
		//cout<<pose_estimate<<endl;
		//cout<<pose_estimate[3]<<endl;
		//cout<<pose_estimate1<<endl;
		//waitKey(0); 
		//cx=cx-200;
		//edge_left= LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);

		//cv::Point3f pupil_left;
		//cv::Point3f pupil_right;
		cv:: Point left;
		cv:: Point right;

		//if (det_parameters.track_gaze && detection_success && face_model.eye_model)
		//{
	//		FaceAnalysis::DrawGaze(&left,&right,captured_image, face_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
	//	}

		//cout<<left.x<<"  "<<left.y <<"  "<<endl;
		//left.x=pupil_left.x;
		//left.y = pupil_left.y;
		//if((left.x==0 && left.y==0) || abs(right.x-left.x)<15){
		///	cout<<right.x-left.x<<endl;
			//cout<<"check"<<endl;
			//return;
		//}
		//cv:: circle(captured_image,ll,10,Scalar( 0, 0, 255 ),5,8);
		//cv:: circle(captured_image,rr,10,Scalar( 0, 0, 255 ),5,8);
		
		if(abs(pose_estimate[4]>0.3)){
			return;
		}

		double width,width1;
		if(pose_estimate[4]>0){
			width=3*final_width*cos(pose_estimate[4]);
			width1=3*final_width/cos(pose_estimate[4]);
		}
		else{
			width=3*final_width/cos(pose_estimate[4]);
			width1=3*final_width*cos(pose_estimate[4]);
		}
		//cout<<"visualize 2"<<endl;
		//cout.flush();

		double height= 5*final_width*cos(pose_estimate[3]);//abs(edge_left[1].first.y - edge_left[1].second.y);

		//abs(edge_right[0].first.x - edge_right[0].second.x) ;
		double height1=5*final_width*cos(pose_estimate[3]);

		double m1r=(float)(rr.y-rl.y)/(float)(rr.x-rl.x);
		//cout<<atan(m1r)*180/PI<<endl;
		double m2r = 1/m1r;
		//cout<<atan(m2r)*180/PI<<endl;
		double m1l=(float)(lr.y-ll.y)/(float)(lr.x-ll.x);
		//cout<<m1l<<endl;
		double m2l = 1/m1l;
		//cout<<m2l<<endl;

		double pointrx,pointry,pointlx,pointly;
		get_point(rr.x,rr.y,m2r,1.5*final_width,&pointrx,&pointry,0);
		get_point(ll.x,ll.y,m2l,1.5*final_width,&pointlx,&pointly,0);

		//cv:: circle(captured_image,ll,10,Scalar( 0, 0, 255 ),2,2);
		//cv:: circle(captured_image,rr,10,Scalar( 0, 0, 255 ),2,2);

		//cv:: circle(captured_image,Point2f(pointrx,pointry),10,Scalar( 0, 0, 255 ),2,2);
		//cv:: circle(captured_image,Point2f(pointlx,pointly),10,Scalar( 0, 0, 255 ),2,2);

		double crx,cry,clx,cly;
		get_point(pointrx,pointry,-m1r,1.5*final_width,&crx,&cry,1);
		get_point(pointlx,pointly,-m1l,1.5*final_width,&clx,&cly,2);

		//cv:: circle(captured_image,Point2f(crx,cry),10,Scalar( 0, 0, 255 ),2,2);
		//cv:: circle(captured_image,Point2f(clx,cly),10,Scalar( 0, 0, 255 ),2,2);

		
		double slope1 =tan(pose_estimate[5]); 
		double slope2 = -1/slope1;
		double x1=0,y1=0,x6=0,y6=0,lx=0,ly=0,ry=0,rx=0,x5=0,y5=0;
		if(1){//slope2>1000 || slope2<-1000){
			x1=rr.x;
			y1=rr.y-final_width;
			//cout<<"infinite";
		}
		else if(slope2>0){
			//cout<<"abhi";
			x1=rr.x+sqrt(final_width*final_width/(1+slope2*slope2));
			y1=rr.y-slope2*(x1-rr.x);
		}
		else{
			//cout<<"SAdasdsad";
			x1=rr.x-sqrt(final_width*final_width/(1+slope2*slope2));
			y1=rr.y-slope2*(x1-rr.x);	
		}



		if(1){//slope2>1000 || slope2<-1000){
			x6=ll.x;
			y6=ll.y-final_width;
			//cout<<"infinite";
		}
		else if(slope2>0){
			//cout<<"abhi";
			x6=ll.x+sqrt(final_width*final_width/(1+slope2*slope2));
			y6=ll.y-slope2*(x6-ll.x);
		}
		else{
			//cout<<"SAdasdsad";
			x6=ll.x-sqrt(final_width*final_width/(1+slope2*slope2));
			y6=ll.y-slope2*(x6-ll.x);	
		}


		if(1){//slope1>1000 || slope1<-1000){
			x5=x6-width;
			y5=y6;
			//cout<<"infinite";
		}
		else if(slope1>0){
			//cout<<"abhi";
			x5=x6-sqrt(width*width/(1+slope1*slope1));
			y5=y6-slope1*(x5-x6);
		}
		else{
			//cout<<"SAdasdsad";
			x5=x6-sqrt(width*width/(1+slope1*slope1));
			y5=y6-slope1*(x5-x6);	
		}

		/*if(slope1>1000 || slope1<-1000){
			rx=x1+1.5*final_width;
			ry=y1;
			cout<<"infinite";
		}
		else if(slope1>0){
			//cout<<"abhi";
			rx=x1+sqrt(1.5*1.5*final_width*final_width/(1+slope1*slope1));
			ry=y1-slope1*(rx-x1);
		}
		else{
			//cout<<"SAdasdsad";
			rx=x1+sqrt(1.5*1.5*final_width*final_width/(1+slope1*slope1));
			ry=y1-slope1*(rx-x1);	
		}

		float degree=pose_estimate[5]*90*7/22;

		/*if(pose_estimate[4]>0){
			lx=lx*cos(pose_estimate[4]);
			rx=rx/cos(pose_estimate[4]);
		}
		else{
			lx=lx;//cos(pose_estimate[4]);
			rx=rx;//cos(pose_estimate[4]);
		}*/
		

		float degree1 = atan(m1r) *180/PI;
		float degree2 = atan(m1l) *180/PI;

		RotatedRect rRect = RotatedRect(Point2f(crx,cry),Size2f(width1,height1),degree2);
		  //cout<<rRect<<endl;;
		   

		  Point2f vertices[4];
		  rRect.points(vertices);
		  
		  //for (int i = 0; i < 4; i++)
		  //  line(captured_image, vertices[i], vertices[(i+1)%4], Scalar(0,255,0));

		  Rect brect = rRect.boundingRect();
		  //cout<<rect<<endl;
		  //cout<<brect<<endl;
		  //rectangle(captured_image, brect, Scalar(255,0,0));

		  

		  RotatedRect rRect1 = RotatedRect(Point2f(clx,cly),Size2f(width,height),degree2);
		  //cout<<rRect<<endl;;
		   

		  Point2f vertices1[4];
		  rRect1.points(vertices1);
		  
		  //for (int i = 0; i < 4; i++)
		  //  line(captured_image, vertices1[i], vertices1[(i+1)%4], Scalar(0,255,0));

		  Rect brect1 = rRect1.boundingRect();
		  //cout<<rect<<endl;
		  //cout<<brect<<endl;
		  //rectangle(captured_image, brect1, Scalar(255,0,0));


		Mat M, rotated, im2;
        // get angle and size from the bounding box
        float angle2 = rRect.angle;
        Size rRect_size = rRect.size;
        
        if (rRect.angle < -45.) {
            angle2 += 90.0;
            swap(rRect_size.width, rRect_size.height);
        }
        // get the rotation matrix
        M = getRotationMatrix2D(rRect.center, angle2, 1.0);
        // perform the affine transformation
        warpAffine(captured_image, rotated, M, captured_image.size(), INTER_CUBIC);
        // crop the resulting image
        getRectSubPix(rotated, rRect_size, rRect.center, im2);

        //imshow("rotated",rotated);
        //imshow("cropped",im2);



        Mat M1, rotated1, im1,im3;
        // get angle and size from the bounding box
        float angle1 = rRect1.angle;
        Size rRect1_size = rRect1.size;
        
        if (rRect1.angle < -45.) {
            angle1 += 90.0;
            swap(rRect1_size.width, rRect1_size.height);
        }
        // get the rotation matrix
        M1 = getRotationMatrix2D(rRect1.center, angle1, 1.0);
        // perform the affine transformation
        warpAffine(captured_image, rotated1, M1, captured_image.size(), INTER_CUBIC);
        // crop the resulting image
        getRectSubPix(rotated1, rRect1_size, rRect1.center, im1);

        //imshow("rotated1",rotated1);
        //imshow("cropped1",im1);
		  

		/*if(0 >rr.x && 0 > width1 && rr.x + width1 > captured_image.cols && 0 > rr.y && 0 > height1 && rr.y + height1 > captured_image.rows){
			return;
		}

		if(0 >ll.x && 0 > width && ll.x + width > captured_image.cols && 0 > ll.y && 0 > height && ll.y + height > captured_image.rows){
			return;
		}

		if( x5+width >=captured_image.cols){
			width=captured_image.cols-x5;
		}
		if(y5+height>=captured_image.rows){
			height = captured_image.rows - y5;
		}
		if( x1+width1 >=captured_image.cols ){
			width1=captured_image.cols-x1;
		}
		if(y1+height1>=captured_image.rows){
			height1 = captured_image.rows - y1;
		}
		if(x5<0){
			x5=0;
		}
		if(y5<0){
			y5=0;
		}
		if(x1>captured_image.cols){
			x1=captured_image.cols-width1;
		}
		if(y1<0){
			y1=0;
		}
		cv:: Mat im1 , im2,im3;
		cv:: Rect rect = Rect( max(x5,0.0), max(y5,0.0),width,height); 
		captured_image(rect).copyTo(im1); 
		cv:: Rect rect1 = Rect( max(x1,0.0), max(y1,0.0),width1,height1); 
		captured_image(rect1).copyTo(im2); */

		char name[100];			
		// output the frame number
		std::sprintf(name, "l2_%06d.png", frame_count);

		// Construct the output filename
		boost::filesystem::path slash("/");
			
		std::string preferredSlash = slash.make_preferred().string();
		
		string out_file1 = output_similarity_align[f_n] + preferredSlash + string(name);


		char name1[100];			
		// output the frame number
		std::sprintf(name1, "r2_%06d.png", frame_count);
		
		string out_file2 = output_similarity_align[f_n] + preferredSlash + string(name1);

		char name2[100];			
		// output the frame number
		std::sprintf(name2, "nl2_%06d.png", frame_count);
		
		string out_file3 = output_similarity_align[f_n] + preferredSlash + string(name2);

		char name3[100];			
		// output the frame number
		std::sprintf(name3, "nr2_%06d.png", frame_count);
		
		string out_file4 = output_similarity_align[f_n] + preferredSlash + string(name3);

		//Mat im1 = imread("sample.jpg");
	    //Mat im2 = imread("sample1.jpg");
	    //Size sz1 = im1.size();
	    //Size sz2 = im2.size();
	    //cout<<sz1.width<<" "<<sz2.width<<endl;
	    //Mat im3(sz1.height, sz1.width+sz2.width, CV_8UC3);
	    //Mat left(im3, Rect(0, 0, sz1.width, sz1.height));
	    //im1.copyTo(left);
	    //Mat right(im3, Rect(sz1.width, 0, sz2.width, sz2.height));
	    //im2.copyTo(right);
	 cv::flip(im2, im3, 1);
	    //imshow("im3", im3);

		//imwrite(out_file1,im1);
		//imwrite(out_file2,im3);

	    //char a=waitKey(2);
		
		//if(a=='n'){
			imwrite(out_file3,im1);
			imwrite(out_file4,im3);
			//cout<<out_file4<<endl;
			//imshow("im1", im1);
			//imshow("im3", im3);
			*label_file << frame_count <<"  "<<"0"<<endl;

			Mat output_image;
    		resize(im1,output_image,Size(64,64));
    		 Mat gray_image;
    	 	cvtColor( output_image, gray_image, CV_BGR2GRAY );

    	 	Mat output_image1;
    		resize(im3,output_image1,Size(64,64));
    		 Mat gray_image1;
    	 	cvtColor( output_image1, gray_image1, CV_BGR2GRAY );

    		HOGDescriptor hog(Size(64,64),Size(16,16),Size(16,16),Size(8,8),9);//need to modify params based on the images we are using. 
    		vector<float> descriptorsValues;
    		vector<Point> locations;

    		vector<float> descriptorsValues1;
    		vector<Point> locations1;
    		//cout<<"visualize 3"<<endl;

    		if(frame_count%20==0){
    		hog.compute(gray_image, descriptorsValues, Size(0,0), Size(0,0), locations);
    		hog.compute(gray_image1, descriptorsValues1, Size(0,0), Size(0,0), locations);
    		system("./age-gen.sh &");
    		//cout<<"HOG descriptor size is "<<hog.getDescriptorSize()<<endl;
    		//cout << "img dimensions: " << output_image.cols << " width x " << output_image.rows << "height" << endl;
    		//cout<<"Found "<<descriptorsValues.size()<<" descriptor values"<<endl;
    		//cout<<"Nr of locations specified : "<<locations.size()<<endl;
    		//cout<<"descriptorsValues:  "<< descriptorsValues.size()<<endl;
    		//cout<<descriptorsValues[0]<<endl;

    		for(int k=0;k<descriptorsValues.size();k++){
    			train_data<<k+1<<":"<<descriptorsValues[k]<<" ";
    			//train_data<<k+1<<":"<<descriptorsValues1[k]<<" ";
    		}
    		train_data<<endl;
    		train_data<<"0 ";
    		for(int k=0;k<descriptorsValues1.size();k++){
    			//train_data<<k+1<<":"<<descriptorsValues[k]<<" ";
    			train_data<<k+1<<":"<<descriptorsValues1[k]<<" ";
    		}
    		train_data<<endl;
    		


	    		system("cp ~/Downloads/OpenFace/build/testing.txt ~/Downloads/libsvm-3.21");
	    		//system("cd ~/Downloads/libsvm-3.21");
	    		system("~/Downloads/libsvm-3.21/svm-scale -r ~/Downloads/libsvm-3.21/scalingparamsmpd ~/Downloads/libsvm-3.21/testing.txt > ~/Downloads/libsvm-3.21/scaledtesting1.txt");
	    		system("~/Downloads/libsvm-3.21/svm-predict ~/Downloads/libsvm-3.21/scaledtesting1.txt ~/Downloads/libsvm-3.21/mpd_full.model finaloutput.txt");

	    		std:: ifstream output("finaloutput.txt");
	    		
	    		output>>result1;
	    		output >> result2;
	    	}

	    	//cout<<"visualize 4"<<endl;

    		if(result2==1 || result1==1){
    			//cout<<"Mobile Phone Detected. Dont Use Mobile Phone\n";
    			if(result2==1){
    				for (int k = 0; k < 4; k++)
		  				line(captured_image, vertices[k], vertices[(k+1)%4], Scalar(0,255,0));
    				//rectangle(captured_image, rect, Scalar(0,0,255));
    			}
    			else{
    				for (int k = 0; k < 4; k++)
		  				line(captured_image, vertices1[k], vertices1[(k+1)%4], Scalar(0,255,0));
    			}
    			//	rectangle(captured_image, rect1, Scalar(0,0,255));
    			cv::putText(captured_image,"Mobile Phone Detected", cv::Point(100, 20), CV_FONT_HERSHEY_SIMPLEX,1, CV_RGB(255, 0, 0));
    			if(frame_count%20==0){

                action_name="mobile_usage";
                insertAction(action_name);

                }

    		}
    		else{
    			//cout<<"Mobile Phone NOT detected"<<endl;
    			if(frame_count%20==0){
                //system("./update0.sh");
                //TODO - ADD MYSQL CONNECTOR - not needed now
                }
    		}

    	//}
		/*else if(a == 'r'){
			imwrite(out_file1,im1);
			imwrite(out_file3,im3);
			imshow("im3", im1);
			*label_file << frame_count <<"  "<<"1"<<endl;
		}
		else if (a == 'l'){
			imwrite(out_file3,im1);
			imwrite(out_file2,im3);
			imshow("im3", im3);
			*label_file << frame_count <<"  "<<"2"<<endl;
		}*/

	}

	// Work out the framerate
	if (frame_count % 20 == 0)
	{
		double t1 = cv::getTickCount();
		fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
		t0 = t1;
	}

	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps_tracker);
	string fpsSt("FPS:");
	fpsSt += fpsC;
	cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0));

	if (!det_parameters.quiet_mode)
	{
        cv::namedWindow("tracking_result", 1);
        cv::imshow("tracking_result", captured_image);

	}
}

void prepareOutputFile(std::ofstream* output_file, bool output_2D_landmarks, bool output_3D_landmarks,
	bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
	int num_landmarks, int num_model_modes, vector<string> au_names_class, vector<string> au_names_reg)
{

	*output_file << "frame, timestamp, confidence, success";

	if (output_gaze)
	{
		*output_file << ", gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_2_z";
	}

	if (output_pose)
	{
		*output_file << ", pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz";
	}

	if (output_2D_landmarks)
	{
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", x_" << i;
		}
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", y_" << i;
		}
	}

	if (output_3D_landmarks)
	{
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", X_" << i;
		}
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", Y_" << i;
		}
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", Z_" << i;
		}
	}

	// Outputting model parameters (rigid and non-rigid), the first parameters are the 6 rigid shape parameters, they are followed by the non rigid shape parameters
	if (output_model_params)
	{
		*output_file << ", p_scale, p_rx, p_ry, p_rz, p_tx, p_ty";
		for (int i = 0; i < num_model_modes; ++i)
		{
			*output_file << ", p_" << i;
		}
	}

	if (output_AUs)
	{
		std::sort(au_names_reg.begin(), au_names_reg.end());
		for (string reg_name : au_names_reg)
		{
			*output_file << ", " << reg_name << "_r";
		}

		std::sort(au_names_class.begin(), au_names_class.end());
		for (string class_name : au_names_class)
		{
			*output_file << ", " << class_name << "_c";
		}
	}

	*output_file << endl;

}

// Output all of the information into one file in one go (quite a few parameters, but simplifies the flow)
void outputAllFeatures(std::ofstream* output_file, bool output_2D_landmarks, bool output_3D_landmarks,
	bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
	const LandmarkDetector::CLNF& face_model, int frame_count, double time_stamp, bool detection_success,
	cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, const cv::Vec6d& pose_estimate, double fx, double fy, double cx, double cy,
	const FaceAnalysis::FaceAnalyser& face_analyser)
{

	double confidence = 0.5 * (1 - face_model.detection_certainty);

	*output_file << frame_count + 1 << ", " << time_stamp << ", " << confidence << ", " << detection_success;

	// Output the estimated gaze
	if (output_gaze)
	{
		*output_file << ", " << gazeDirection0.x << ", " << gazeDirection0.y << ", " << gazeDirection0.z
			<< ", " << gazeDirection1.x << ", " << gazeDirection1.y << ", " << gazeDirection1.z;
	}

	// Output the estimated head pose
	if (output_pose)
	{
		*output_file << ", " << pose_estimate[0] << ", " << pose_estimate[1] << ", " << pose_estimate[2]
			<< ", " << pose_estimate[3] << ", " << pose_estimate[4] << ", " << pose_estimate[5];
	}

	// Output the detected 2D facial landmarks
	if (output_2D_landmarks)
	{
		for (int i = 0; i < face_model.pdm.NumberOfPoints() * 2; ++i)
		{
			*output_file << ", " << face_model.detected_landmarks.at<double>(i);
		}
	}

	// Output the detected 3D facial landmarks
	if (output_3D_landmarks)
	{
		cv::Mat_<double> shape_3D = face_model.GetShape(fx, fy, cx, cy);
		for (int i = 0; i < face_model.pdm.NumberOfPoints() * 3; ++i)
		{
			*output_file << ", " << shape_3D.at<double>(i);
		}
	}

	if (output_model_params)
	{
		for (int i = 0; i < 6; ++i)
		{
			*output_file << ", " << face_model.params_global[i];
		}
		for (int i = 0; i < face_model.pdm.NumberOfModes(); ++i)
		{
			*output_file << ", " << face_model.params_local.at<double>(i, 0);
		}
	}



	if (output_AUs)
	{
		auto aus_reg = face_analyser.GetCurrentAUsReg();

		vector<string> au_reg_names = face_analyser.GetAURegNames();
		std::sort(au_reg_names.begin(), au_reg_names.end());

		// write out ar the correct index
		for (string au_name : au_reg_names)
		{
			for (auto au_reg : aus_reg)
			{
				if (au_name.compare(au_reg.first) == 0)
				{
					*output_file << ", " << au_reg.second;
					break;
				}
			}
		}

		if (aus_reg.size() == 0)
		{
			for (size_t p = 0; p < face_analyser.GetAURegNames().size(); ++p)
			{
				*output_file << ", 0";
			}
		}

		auto aus_class = face_analyser.GetCurrentAUsClass();

		vector<string> au_class_names = face_analyser.GetAUClassNames();
		std::sort(au_class_names.begin(), au_class_names.end());

		// write out ar the correct index
		for (string au_name : au_class_names)
		{
			for (auto au_class: aus_class)
			{
				if (au_name.compare(au_class.first) == 0)
				{
					*output_file << ", " << au_class.second;
					break;
				}
			}
		}

		if (aus_class.size() == 0)
		{
			for (size_t p = 0; p < face_analyser.GetAUClassNames().size(); ++p)
			{
				*output_file << ", 0";
			}
		}
	}
	*output_file << endl;
}

int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

void post_process_output_file(FaceAnalysis::FaceAnalyser& face_analyser, string output_file);


int main (int argc, char **argv)
{
    connectDatabase();
	char a = 'n';
	vector<string> arguments = get_arguments(argc, argv);

	// Some initial parameters that can be overriden from command line	
	vector<string> input_files, depth_directories, output_files, tracked_videos_output;
	
	LandmarkDetector::FaceModelParameters det_parameters(arguments);
	// Always track gaze in feature extraction
	det_parameters.track_gaze = true;

	// Get the input output file parameters
	
	// Indicates that rotation should be with respect to camera or world coordinates
	bool use_world_coordinates;
	LandmarkDetector::get_video_input_output_params(input_files, depth_directories, output_files, tracked_videos_output, use_world_coordinates, arguments);
	/*cout<<input_files.size()<<"  old"<<endl;

	string dir1 = string("data");
    //string dir3 = string("cropedtext");
    //vector<string> input_files = vector<string>();
    //vector<string> files3 = vector<string>();

    getdir(dir1,input_files);
    //getdir(dir3,files3);
    sort(input_files.begin(),input_files.end());
    //sort(files3.begin(),files3.end());

    cout<<input_files.size()<<"  new"<<endl;*/

    


	bool video_input = true;
	bool verbose = true;
	bool images_as_video = false;

	vector<vector<string> > input_image_files;

	// Adding image support for reading in the files
	if(input_files.empty())
	{
		vector<string> d_files;
		vector<string> o_img;
		vector<cv::Rect_<double>> bboxes;
		get_image_input_output_params_feats(input_image_files, images_as_video, arguments);	

		if(!input_image_files.empty())
		{
			video_input = false;
		}

	}

	// Grab camera parameters, if they are not defined (approximate values will be used)
	float fx = 0, fy = 0, cx = 0, cy = 0;
	int d = 0;
	// Get camera parameters
	LandmarkDetector::get_camera_params(d, fx, fy, cx, cy, arguments);    
	
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
	LandmarkDetector::CLNF face_model(det_parameters.model_location);	

	vector<string> output_similarity_align;
	vector<string> output_hog_align_files;

	double sim_scale = 0.7;
	int sim_size = 112;
	bool grayscale = false;	
	bool video_output = false;
	bool rigid = false;	
	int num_hog_rows;
	int num_hog_cols;

	// By default output all parameters, but these can be turned off to get smaller files or slightly faster processing times
	// use -no2Dfp, -no3Dfp, -noMparams, -noPose, -noAUs, -noGaze to turn them off
	bool output_2D_landmarks = true;
	bool output_3D_landmarks = true;
	bool output_model_params = true;
	bool output_pose = true; 
	bool output_AUs = true;
	bool output_gaze = true;

	get_output_feature_params(output_similarity_align, output_hog_align_files, sim_scale, sim_size, grayscale, rigid, verbose, 
		output_2D_landmarks, output_3D_landmarks, output_model_params, output_pose, output_AUs, output_gaze, arguments);
	

	//
	//cout<<output_similarity_align[0]<<endl;

	/*for(int j=0;j<input_files.size();j++){
		stringstream convert;
	    convert<<j;
	    string index;
	    index =convert.str();

		cout<<input_files[j]<<endl;
		output_similarity_align.push_back("output_features/file"+index);
		create_directory("output_features/file"+index);
	}

	cout<<output_similarity_align.size()<<endl;

	for(int j=0; j<output_similarity_align.size();j++){
		cout<<output_similarity_align[j]<<endl;
	}*/
	// Used for image masking

	cv::Mat_<int> triangulation;
	string tri_loc;
	if(boost::filesystem::exists(path("model/tris_68_full.txt")))
	{
		std::ifstream triangulation_file("model/tris_68_full.txt");
		LandmarkDetector::ReadMat(triangulation_file, triangulation);
		tri_loc = "model/tris_68_full.txt";
	}
	else
	{
		path loc = path(arguments[0]).parent_path() / "model/tris_68_full.txt";
		tri_loc = loc.string();

		if(exists(loc))
		{
			std::ifstream triangulation_file(loc.string());
			LandmarkDetector::ReadMat(triangulation_file, triangulation);
		}
		else
		{
			cout << "Can't find triangulation files, exiting" << endl;
			return 0;
		}
	}	

	// Will warp to scaled mean shape
	cv::Mat_<double> similarity_normalised_shape = face_model.pdm.mean_shape * sim_scale;
	// Discard the z component
	similarity_normalised_shape = similarity_normalised_shape(cv::Rect(0, 0, 1, 2*similarity_normalised_shape.rows/3)).clone();

	// If multiple video files are tracked, use this to indicate if we are done
	bool done = false;	
	int f_n = -1;
	int curr_img = -1;

	string au_loc;
	if(boost::filesystem::exists(path("AU_predictors/AU_all_best.txt")))
	{
		au_loc = "AU_predictors/AU_all_best.txt";
	}
	else
	{
		path loc = path(arguments[0]).parent_path() / "AU_predictors/AU_all_best.txt";

		if(exists(loc))
		{
			au_loc = loc.string();
		}
		else
		{
			cout << "Can't find AU prediction files, exiting" << endl;
			return 0;
		}
	}	

	// Creating a  face analyser that will be used for AU extraction
	FaceAnalysis::FaceAnalyser face_analyser(vector<cv::Vec3d>(), 0.7, 112, 112, au_loc, tri_loc);
		
	while(!done) // this is not a for loop as we might also be reading from a webcam
	{
		
		string current_file;
		
		cv::VideoCapture video_capture;
		
		cv::Mat captured_image;
		int total_frames = -1;
		int reported_completion = 0;

		double fps_vid_in = -1.0;

		if(video_input)
		{
			// We might specify multiple video files as arguments
			if(input_files.size() > 0)
			{
				f_n++;

				current_file = input_files[f_n];
				if(current_file=="." || current_file == ".."){
					continue;
				}
			}
			else
			{
				// If we want to write out from webcam
				f_n = 0;
			}
			// Do some grabbing
			if( current_file.size() > 0 )
			{
				INFO_STREAM( "Attempting to read from file: " << current_file );
				video_capture = cv::VideoCapture(current_file);
				total_frames = (int)video_capture.get(CV_CAP_PROP_FRAME_COUNT);
				fps_vid_in = video_capture.get(CV_CAP_PROP_FPS);

				// Check if fps is nan or less than 0
				if (fps_vid_in != fps_vid_in || fps_vid_in <= 0)
				{
					INFO_STREAM("FPS of the video file cannot be determined, assuming 30");
					fps_vid_in = 30;
				}
			}
			else{
                video_capture = cv::VideoCapture(0);
			}

			if (!video_capture.isOpened())
			{
				FATAL_STREAM("Failed to open video source, exiting");
				return 1;
			}
			else
			{
				INFO_STREAM("Device or file opened");
			}

			video_capture >> captured_image;	
		}
		else
		{
			f_n++;	
			curr_img++;
			if(!input_image_files[f_n].empty())
			{
				string curr_img_file = input_image_files[f_n][curr_img];
				captured_image = cv::imread(curr_img_file, -1);
			}
			else
			{
				FATAL_STREAM( "No .jpg or .png images in a specified drectory, exiting" );
				return 1;
			}

		}	
		
		// If optical centers are not defined just use center of image
		if(cx_undefined)
		{
			cx = captured_image.cols / 2.0f;
			cy = captured_image.rows / 2.0f;
		}
		// Use a rough guess-timate of focal length
		if (fx_undefined)
		{
			fx = 500 * (captured_image.cols / 640.0);
			fy = 500 * (captured_image.rows / 480.0);

			fx = (fx + fy) / 2.0;
			fy = fx;
		}
	
		// Creating output files
		std::ofstream output_file;

		std::ofstream label_file;

		stringstream convert;
	    convert<<f_n;
	    string id;
	    id =convert.str();

		label_file.open("cropedtext/file"+id+".txt", ios::out | ios::trunc );
		
		label_file << "Frame  Phone"<<endl;
		if (!output_files.empty())
		{
			output_file.open(output_files[f_n], ios_base::out);
			prepareOutputFile(&output_file, output_2D_landmarks, output_3D_landmarks, output_model_params, output_pose, output_AUs, output_gaze, face_model.pdm.NumberOfPoints(), face_model.pdm.NumberOfModes(), face_analyser.GetAUClassNames(), face_analyser.GetAURegNames());
		}

		// Saving the HOG features
		std::ofstream hog_output_file;
		if(!output_hog_align_files.empty())
		{
			hog_output_file.open(output_hog_align_files[f_n], ios_base::out | ios_base::binary);
		}

		// saving the videos
		cv::VideoWriter writerFace;
		if(!tracked_videos_output.empty())
		{
			writerFace = cv::VideoWriter(tracked_videos_output[f_n], CV_FOURCC('D', 'I', 'V', 'X'), fps_vid_in, captured_image.size(), true);
		}

		int frame_count = 0;
		
		// This is useful for a second pass run (if want AU predictions)
		vector<cv::Vec6d> params_global_video;
		vector<bool> successes_video;
		vector<cv::Mat_<double>> params_local_video;
		vector<cv::Mat_<double>> detected_landmarks_video;
				
		// Use for timestamping if using a webcam
		int64 t_initial = cv::getTickCount();

		bool visualise_hog = verbose;

		// Timestamp in seconds of current processing
		double time_stamp = 0;

		INFO_STREAM( "Starting tracking");
		while(!captured_image.empty())
		{		

			// Grab the timestamp first
			//cout<<"INSIDE"<<endl;
			if (video_input)
			{
				time_stamp = (double)frame_count * (1.0 / fps_vid_in);				
			}
			else
			{
				// if loading images assume 30fps
				time_stamp = (double)frame_count * (1.0 / 30.0);
			}

			// Reading the images
			cv::Mat_<uchar> grayscale_image;

			if(captured_image.channels() == 3)
			{
				cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);				
			}
			else
			{
				grayscale_image = captured_image.clone();				
			}
		
			// The actual facial landmark detection / tracking
			bool detection_success;
			
			if(video_input || images_as_video)
			{
				detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_model, det_parameters);
			}
			else
			{
				detection_success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, face_model, det_parameters);
			}
			
			// Gaze tracking, absolute gaze direction
			cv::Point3f gazeDirection0(0, 0, -1);
			cv::Point3f gazeDirection1(0, 0, -1);

			if (det_parameters.track_gaze && detection_success && face_model.eye_model)
			{
				FaceAnalysis::EstimateGaze(face_model, gazeDirection0, fx, fy, cx, cy, true);
				FaceAnalysis::EstimateGaze(face_model, gazeDirection1, fx, fy, cx, cy, false);
			}



			// Do face alignment
			cv::Mat sim_warped_img;
			cv::Mat_<double> hog_descriptor;

			// But only if needed in output
			if(!output_similarity_align.empty() || hog_output_file.is_open() || output_AUs)
			{
				face_analyser.AddNextFrame(captured_image, face_model, time_stamp, false, !det_parameters.quiet_mode);
				face_analyser.GetLatestAlignedFace(sim_warped_img);

				if(!det_parameters.quiet_mode)
				{
					//cv::imshow("sim_warp", sim_warped_img);			
				}
				if(hog_output_file.is_open())
				{
					FaceAnalysis::Extract_FHOG_descriptor(hog_descriptor, sim_warped_img, num_hog_rows, num_hog_cols);						

					if(visualise_hog && !det_parameters.quiet_mode)
					{
						cv::Mat_<double> hog_descriptor_vis;
						FaceAnalysis::Visualise_FHOG(hog_descriptor, num_hog_rows, num_hog_cols, hog_descriptor_vis);
                        cv::imshow("hog", hog_descriptor_vis);
					}
				}
			}



			// Work out the pose of the head from the tracked model
			cv::Vec6d pose_estimate;
			if(use_world_coordinates)
			{
				pose_estimate = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);
			}
			else
			{
				pose_estimate = LandmarkDetector::GetCorrectedPoseCamera(face_model, fx, fy, cx, cy);
			}

			if(hog_output_file.is_open())
			{
				output_HOG_frame(&hog_output_file, detection_success, hog_descriptor, num_hog_rows, num_hog_cols);
			}
			//cout<<"Inside2"<<endl;
			// Write the similarity normalised output
			/*if(!output_similarity_align.empty())
			{
				//cout<<"ab\n";
				if (sim_warped_img.channels() == 3 && grayscale)
				{
					cvtColor(sim_warped_img, sim_warped_img, CV_BGR2GRAY);
				}
				//cout<<"cd\n";
				char name[100];
					
				// output the frame number
				std::sprintf(name, "frame_det_%06d.png", frame_count);

				// Construct the output filename
				boost::filesystem::path slash("/");

				//cout<<"ef\n";
					
				std::string preferredSlash = slash.make_preferred().string();
				
				string out_file = output_similarity_align[f_n] + preferredSlash + string(name);
				//imwrite(out_file, sim_warped_img);

			}*/

			//cout<<"Inside4"<<endl;

			// Visualising the tracker
			char k = waitKey(2);
			if(k>0){
				a=k;
			}
			else{
				k=a;
			}




			visualise_tracking(&label_file,a,f_n,output_similarity_align,captured_image, face_model, det_parameters, gazeDirection0, gazeDirection1, frame_count, fx, fy, cx, cy);

			// Output the landmarks, pose, gaze, parameters and AUs
			outputAllFeatures(&output_file, output_2D_landmarks, output_3D_landmarks, output_model_params, output_pose, output_AUs, output_gaze,
				face_model, frame_count, time_stamp, detection_success, gazeDirection0, gazeDirection1,
				pose_estimate, fx, fy, cx, cy, face_analyser);

			// output the tracked video
			if(!tracked_videos_output.empty())
			{		
				writerFace << captured_image;
			}

			if(video_input)
			{
				video_capture >> captured_image;
			}
			else
			{
				curr_img++;
				if(curr_img < (int)input_image_files[f_n].size())
				{
					string curr_img_file = input_image_files[f_n][curr_img];
					captured_image = cv::imread(curr_img_file, -1);
				}
				else
				{
					captured_image = cv::Mat();
				}
			}
			// detect key presses
			char character_press = cv::waitKey(1);
			
			// restart the tracker
			if(character_press == 'r')
			{
				face_model.Reset();
			}
			// quit the application
			else if(character_press=='q')
			{
				return(0);
			}

			// Update the frame count
			frame_count++;

			if(total_frames != -1)
			{
				if((double)frame_count/(double)total_frames >= reported_completion / 10.0)
				{
					cout << reported_completion * 10 << "% ";
					reported_completion = reported_completion + 1;
				}
			}

		}
		
		output_file.close();

		if(output_files.size() > 0)
		{
		
			// If the video is long enough post-process it for AUs
			if (output_AUs && frame_count > 100)
			{
				post_process_output_file(face_analyser, output_files[f_n]);
			}
		}
		// Reset the models for the next video
		face_analyser.Reset();
		face_model.Reset();

		frame_count = 0;
		curr_img = -1;

		if (total_frames != -1)
		{
			cout << endl;
		}

		// break out of the loop if done with all the files (or using a webcam)
		if(f_n == input_files.size() -1 || input_files.empty())
		{
			done = true;
		}
	}

	return 0;
}

// Allows for post processing of the AU signal
void post_process_output_file(FaceAnalysis::FaceAnalyser& face_analyser, string output_file)
{

	vector<double> certainties;
	vector<bool> successes;
	vector<double> timestamps;
	vector<std::pair<std::string, vector<double>>> predictions_reg;
	vector<std::pair<std::string, vector<double>>> predictions_class;

	// Construct the new values to overwrite the output file with
	face_analyser.ExtractAllPredictionsOfflineReg(predictions_reg, certainties, successes, timestamps);
	face_analyser.ExtractAllPredictionsOfflineClass(predictions_class, certainties, successes, timestamps);

	int num_class = predictions_class.size();
	int num_reg = predictions_reg.size();

	// Extract the indices of writing out first
	vector<string> au_reg_names = face_analyser.GetAURegNames();
	std::sort(au_reg_names.begin(), au_reg_names.end());
	vector<int> inds_reg;

	// write out ar the correct index
	for (string au_name : au_reg_names)
	{
		for (int i = 0; i < num_reg; ++i)
		{
			if (au_name.compare(predictions_reg[i].first) == 0)
			{
				inds_reg.push_back(i);
				break;
			}
		}
	}

	vector<string> au_class_names = face_analyser.GetAUClassNames();
	std::sort(au_class_names.begin(), au_class_names.end());
	vector<int> inds_class;

	// write out ar the correct index
	for (string au_name : au_class_names)
	{
		for (int i = 0; i < num_class; ++i)
		{
			if (au_name.compare(predictions_class[i].first) == 0)
			{
				inds_class.push_back(i);
				break;
			}
		}
	}
	// Read all of the output file in
	vector<string> output_file_contents;

	std::ifstream infile(output_file);
	string line;

	while (std::getline(infile, line))
		output_file_contents.push_back(line);

	infile.close();

	// Read the header and find all _r and _c parts in a file and use their indices
	std::vector<std::string> tokens;
	boost::split(tokens, output_file_contents[0], boost::is_any_of(","));

	int begin_ind = -1;
	
	for (int i = 0; i < tokens.size(); ++i)
	{
		if (tokens[i].find("AU") != string::npos && begin_ind == -1)
		{
			begin_ind = i;
			break;
		}
	}
	int end_ind = begin_ind + num_class + num_reg;

	// Now overwrite the whole file
	std::ofstream outfile(output_file, ios_base::out);
	// Write the header
	outfile << output_file_contents[0].c_str() << endl;
	
	// Write the contents
	for (int i = 1; i < output_file_contents.size(); ++i)
	{
		std::vector<std::string> tokens;
		boost::split(tokens, output_file_contents[i], boost::is_any_of(","));

		outfile << tokens[0];

		for (int t = 1; t < tokens.size(); ++t)
		{
			if (t >= begin_ind && t < end_ind)
			{
				if(t - begin_ind < num_reg)
				{
					outfile << "," << predictions_reg[inds_reg[t - begin_ind]].second[i - 1];
				}
				else
				{
					outfile << "," << predictions_class[inds_class[t - begin_ind - num_reg]].second[i - 1];
				}
			}
			else
			{
				outfile << "," << tokens[t];
			}
		}
		outfile << endl;
	}
		

}

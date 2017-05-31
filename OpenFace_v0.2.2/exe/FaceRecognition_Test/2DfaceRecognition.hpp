#ifndef _2DFaceRecognition_hpp_
#define _2DFaceRecognition_hpp_
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

//lbp histogram headers
#include "lbp.hpp"
#include "histogram.hpp"

using namespace std;
using namespace cv;

class FaceRecognition{



string name="";
float th_knwn=0.5;
float th_uknwn=0.5;
string MODEL_FILE = "";
string TRAIN_FEATURE_FILE = "";
string TEST_FEATURE_FILE = "";
string OUTPUT_FILE = "";
int numTrainFaces=1;
int numTestFaces=1;
int radius = 1;
int neighbors= 8;
int gridx=8;
int gridy=8;
int bins=16;
public :
FaceRecognition();

void reset(int code);
void generate_features(Mat frame, string data, int label);

void  preprocess_image(Mat img, Mat& processed_img);
    
int Train(Mat input_frame, string subjectname,int numSamples,int count);

int Test(Mat input_frame, string subjectname,string FRModelFile,int numSamples,int count);

int  probabilies(string OUTPUT_FILE);

Mat norm_0_255(const Mat& src);
};

#endif


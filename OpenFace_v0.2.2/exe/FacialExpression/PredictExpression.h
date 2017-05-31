#ifndef _PREDICTEXPRESSION_H
#define _PREDICTEXPRESSION_H

#define _CRT_SECURE_NO_WARNINGS_
#include <sstream>
#include <fstream>
#include <string.h>
#include <opencv2/opencv.hpp>


class PredictExpression
{
	float probabilityThreshold = 0.3;

	public:
	std::string expressionNames[4] = { "Smile","Anger","Surprise","Neutral" };
	PredictExpression();
	int RunPrediction(std::string feature_vector,std::string probability_output);
	int PredictClass(std::string outputfilename);
};
#endif

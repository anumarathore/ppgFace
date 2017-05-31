#include "stdafx.h"

#include "PredictExpression.h"
 PredictExpression::PredictExpression()
{
	//Reading threshold values;
	std::string filename = "/bin/ExpressionPredictionThresholds.xml";
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	fs["PredictionProbability"] >> probabilityThreshold;
	fs.release();
}
int PredictExpression::RunPrediction(std::string feature_vector, std::string probablity_output)
{

	std::string scaleData = "./bin/svm-scale -r bin/emoscale "+feature_vector+" > "+feature_vector+"_scaled.txt";
	int k=system(scaleData.c_str());
//	int k = svm_scale_restore_main("emoscale", "emo_result.txt", "emo_result_scaled.txt");
	std::string predict_emo ="./bin/svm-predict -b 1 -q "+feature_vector+"_scaled.txt bin/emo_model.txt "+probablity_output;
 	k=system(predict_emo.c_str());
	//int p = svm_predict_main("emo_result_scaled.txt", "emo_model.txt", "emo_output.txt");
	return PredictClass(probablity_output);
}

int PredictExpression::PredictClass(std::string outputfilename)
{
	std::ifstream f1;
	f1.open(outputfilename.c_str(), std::fstream::in);
	std::string line;
	const char* arr = NULL;
	std::vector<std::string> probs;

	while (!f1.eof())
	{
		if (f1.peek() == f1.eof())
			break;
		getline(f1, line);
		//std::stringstream ss(line);
		//ss >> label >> value1 >> value2>>value3>>value4;
		probs.push_back(line);
	}

	f1.close();
	probs.erase(probs.begin());
	int label = -1;
	int labelcounter[4] = { 0,0,0,0 };
	float labelvalues[4] = { 0,0,0,0 };
	float avg_prob_0 = 0, avg_prob_1 = 0;
	for (int i = 0; i < probs.size(); i++)
	{
		std::stringstream ss(probs[i]);
	ss>> label >> labelvalues[0] >> labelvalues[1] >> labelvalues[2] >> labelvalues[3];

	//std::cout << probs[i] << std::endl;
		}

	//int PredictedClassLabel = 3;
	if (labelvalues[label -1] > probabilityThreshold) 
		return label;
	else return 3;
	// // The dominant class label should be present for atleast 50% of the frames collected.
	// //std::cout << "  Max element : " << labelcounter[0] << " "<<labelcounter[1] << " " << labelcounter[2] << " " << labelcounter[3] << std::endl;
	// if(std::max_element(labelcounter,labelcounter+N) > 0.5*emo_frames_num)
	// 	PredictedClassLabel= std::distance(labelcounter, std::max_element(labelcounter, labelcounter + N));
	
	// return PredictedClassLabel+1;
	
}

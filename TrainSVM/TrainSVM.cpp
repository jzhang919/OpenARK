// TrainSVM.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <opencv2/ml.hpp>
#include <string.h>

std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

void trainSVM() {
	const char * data_path = "C:/Users/jzhan299/Downloads/eyeblink8/data.yaml";
	const char * label_path = "C:/Users/jzhan299/Downloads/eyeblink8/labels.yaml";
	cv::FileStorage data_file = cv::FileStorage(data_path, cv::FileStorage::READ);
	cv::FileStorage label_file = cv::FileStorage(label_path, cv::FileStorage::READ);
	cv::Mat data, labels;
	data_file["data"] >> data;
	data.convertTo(data, CV_32F);
	label_file["labels"] >> labels;
	labels.convertTo(labels, CV_32S);
	cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(data, cv::ml::SampleTypes::ROW_SAMPLE, labels);
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::ONE_CLASS);
	svm->setKernel(cv::ml::SVM::RBF);
	svm->setNu(0.4);
	svm->trainAuto(tData, 5);
	std::cout << "Done training." << std::endl;
	svm->save("C:/Users/jzhan299/Downloads/eyeblink8/svm_small.xml");
}

int main()
{
	trainSVM();
    return 0;
}


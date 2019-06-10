// blink_detector.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include "RS2Camera.h"
#include <string>
#include <vector>

const std::string lbfmodel_path = "C:/Program Files/OpenARK/config/face/haarcascade_frontalface_alt2.xml";
const std::string face_haarcascade_path = "C:/Program Files/OpenARK/config/face/lbfmodel.yaml";
float EYE_AR_THRESH = 0.3;
float EYE_AR_CONSEC_FRAMES = 3;

std::string path = "tbd";
float eye_aspect_ratio(std::vector<cv::Point2d> eye) {
	float p2p6 = cv::norm(eye[1] - eye[5]);
	float p3p5 = cv::norm(eye[2] - eye[4]);
	float p1p4 = cv::norm(eye[0] - eye[3]);

	float ear = (p2p6 + p3p5) / (2.0 * p1p4);
	return ear;
}

bool find_blinks(ark::DepthCamera::Ptr camera) {
	std::ofstream ear_file;
	ear_file.open(path);
	int counter = 0, total = 0;

	cv::CascadeClassifier faceDetector;
	cv::Ptr<cv::face::Facemark> facemark = cv::face::FaceMarkLBF::create();

	facemark->loadModel(lbfmodel_path);
	faceDetector.load(face_haarcascade_path);
	cv::Mat gray;
	std::vector<cv::Rect> faces;
	std::vector<std::vector<cv::Point2f>> landmarks;

	camera->beginCapture();
	while (true) {
		cv::Mat rgbMap = camera->getRGBMap();
		cv::cvtColor(rgbMap, gray, cv::color_BGR2GRAY);
		faceDetector.detectMultiScale(gray, faces);
		bool success = facemark->fit(gray, faces, landmarks);
		std::vector<cv::Point2d> l_eye_pts;
		std::vector<cv::Point2d> r_eye_pts;
		if (success && landmarks[0].size() == 68) {
			for (int i = 36; i < 42; i++) {
				l_eye_pts.push_back(landmarks[0][i]);
			}
			for (int j = 42; j < 48; j++) {
				r_eye_pts.push_back(landmarks[0][j]);
			}
		}
		float left_EAR = eye_aspect_ratio(l_eye_pts);
		float right_EAR = eye_aspect_ratio(r_eye_pts);
		float ear = (left_EAR + right_EAR) / 2.0;
		//ear_file << ear << "\n";
		std::cout << "Eye aspect ratio: " << ear << std::endl;
		if (ear < EYE_AR_THRESH) {
			counter++;
		}
		else {
			if (counter >= EYE_AR_CONSEC_FRAMES) {
				total++;
			}
			else {
				counter = 0;
			}
		}

		int key = cv::waitKey(1) & 0xFF;
		if (key == 'Q' || key == 27) {
			break;
		}

		cv::Mat rgbVis = rgbMap.clone();
		for (int k = 0; k < l_eye_pts.size(); k++) {
			cv::circle(rgbVis, l_eye_pts[k], 1, cv::Scalar(0, 0, 255));
		}
		for (int l = 0; l < r_eye_pts.size(); l++) {
			cv::circle(rgbVis, r_eye_pts[l], 1, cv::Scalar(0, 0, 255));
		}
	}
	ear_file.close();
	cv::destroyAllWindows();
	return 0;
}

int main()
{
	ark::DepthCamera::Ptr camera = std::make_shared<ark::RS2Camera>(true);
	find_blinks(camera);
	return 0;
}


// blink_detector.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "BlinkDetector.h"
#include "HumanDetector.h"

//std::string path = "tbd";

namespace ark {
	BlinkDetector::BlinkDetector(DetectionParams::Ptr params) {
		// Since we have seen no humans previously, we set this to default value
		BlinkDetector::facemark = cv::face::FacemarkLBF::create();
		facemark->loadModel(HumanDetector::FACE_LBFMODEL_FILE_PATH);
		faceDetector.load(HumanDetector::FACE_HAARCASCADE_FILE_PATH);

	}

	float BlinkDetector::getEyeAspectRatio(std::vector<cv::Point2d> eye) {
		float p2p6 = cv::norm(eye[1] - eye[5]);
		float p3p5 = cv::norm(eye[2] - eye[4]);
		float p1p4 = cv::norm(eye[0] - eye[3]);

		float ear = (p2p6 + p3p5) / (2.0 * p1p4); // EAR formula suggested by Cech 2016.
		return ear;
	}

	void BlinkDetector::update(cv::Mat &rgbMap) {
		detect(rgbMap);
	}

	void BlinkDetector::detect(cv::Mat &image) {
		detectBlink(image);
	}

	float BlinkDetector::getEar() {
		return BlinkDetector::ear;
	}

	std::vector<cv::Point2d> BlinkDetector::getLeftEyePts () {
		return BlinkDetector::l_eye_pts;
	}

	std::vector<cv::Point2d> BlinkDetector::getRightEyePts() {
		return BlinkDetector::r_eye_pts;
	}

	bool BlinkDetector::detectBlink(cv::Mat &rgbMap) {
		BlinkDetector::l_eye_pts.clear();
		BlinkDetector::r_eye_pts.clear();

		cv::Mat gray;
		std::vector<cv::Rect> faces;
		std::vector<std::vector<cv::Point2f>> landmarks;
		bool blink_found;

		cv::cvtColor(rgbMap, gray, cv::COLOR_BGR2GRAY);
		faceDetector.detectMultiScale(gray, faces);
		if (!(facemark->fit(gray, faces, landmarks))) {
			return false;
		}
		
		if (landmarks[0].size() == 68) {
			for (int i = 36; i < 42; i++) {
				BlinkDetector::l_eye_pts.push_back(landmarks[0][i]);
			}
			for (int j = 42; j < 48; j++) {
				BlinkDetector::r_eye_pts.push_back(landmarks[0][j]);
			}
		}
		float left_EAR = BlinkDetector::getEyeAspectRatio(l_eye_pts);
		float right_EAR = BlinkDetector::getEyeAspectRatio(r_eye_pts);
		BlinkDetector::ear = (left_EAR + right_EAR) / 2.0;
		std::cout << "Eye aspect ratio: " << BlinkDetector::ear << std::endl;
		if (BlinkDetector::ear < EYE_AR_THRESH) {
			BlinkDetector::consecBlinkCounter = 0;
			return false;
		}
		else {
			if (BlinkDetector::consecBlinkCounter >= EYE_AR_CONSEC_FRAMES) {
				return true;
			}
			else {
				BlinkDetector::consecBlinkCounter++;
				return false;
			}
		}
	}
}

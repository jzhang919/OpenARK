#pragma once 
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include "RS2Camera.h"
#include <string>
#include <vector>
#include "Detector.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/pixel.h>
namespace ark {
	class BlinkDetector : public Detector {
	public:
		BlinkDetector(DetectionParams::Ptr params = nullptr);

		float EYE_AR_THRESH = 0.25;
		float EYE_AR_CONSEC_FRAMES = 3;

		void update(cv::Mat &rgbMap);
		void detectFace(const cv::Mat& frame);
		void detectHumanHOG(const cv::Mat& frame);
		void visualizeBlink(cv::Mat & rgbMap);
		int getTotal(void);
		float getEar(void);
		std::vector<cv::Point2d> getLeftEyePts();
		std::vector<cv::Point2d> getRightEyePts();
	protected:
		void detect(cv::Mat & image) override;

	private:
		cv::CascadeClassifier faceDetector;
		cv::Ptr<cv::face::Facemark> facemark;
		int total;
		float ear;
		dlib::frontal_face_detector faceHOG;
		cv::Rect humanDetectionBox;
		std::vector<cv::Point2d> l_eye_pts;
		std::vector<cv::Point2d> r_eye_pts;
		void detectBlink(cv::Mat &rgbMap);
		float getEyeAspectRatio(std::vector<cv::Point2d> eye);
		cv::Rect BlinkDetector::find_max_rec(const std::vector<cv::Rect>& found_filtered);
		int consecBlinkCounter;
	};

}
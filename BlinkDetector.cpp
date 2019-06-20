// blink_detector.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "BlinkDetector.h"
#include "HumanDetector.h"

namespace ark {
	static std::string type2str(int type) {
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

	BlinkDetector::BlinkDetector(DetectionParams::Ptr params) {
		// Since we have seen no humans previously, we set this to default value
		BlinkDetector::total = 0;
		BlinkDetector::ear = 0;
		loadSVM("C:/Users/jzhan299/Downloads/eyeblink8");
		dlib::frontal_face_detector faceHOG = dlib::get_frontal_face_detector();
		BlinkDetector::facemark = cv::face::FacemarkLBF::create();
		facemark->loadModel(HumanDetector::FACE_LBFMODEL_FILE_PATH);
		faceDetector.load(HumanDetector::FACE_HAARCASCADE_FILE_PATH);
		BlinkDetector::fDetector = dlib::get_frontal_face_detector();
		dlib::deserialize(util::resolveRootPath("/config/face/eye_eyebrows_22.dat")) >> BlinkDetector::eyeDetector;
	}

	bool BlinkDetector::loadSVM(const std::string & ipath) {
		using namespace boost::filesystem;

		const char * FILE_NAME = "svm_small.xml";

		std::string loadPath = ipath + "/" + FILE_NAME;

		std::ifstream testIfs(loadPath);
		if (!testIfs) {
			cout << "NOT TRAINED" << endl;
			return trained = false;
		}

		svm = cv::ml::SVM::load(loadPath);
		trained = true;
		return trained;
	}

	float BlinkDetector::classify(const cv::Mat & features) const {
		if (!trained) {
			printf("ERROR: SVM not trained. Please initialize using loadFile.\n");
			return -1;
		}
		// if no fingers, predict not hand
		if (!features.data || features.cols == 0) return 0.0;
	
		features.convertTo(features, CV_32F);
		/*string ty = type2str(features.type());
		printf("Matrix: %s %dx%d \n", ty.c_str(), features.cols, features.rows);*/

		double result = svm->predict(features);
		cout << "SVM Prediction: " << result << endl;
		// range [0, 1]
		return std::max(std::min(1.0, result), 0.0);
	}

	static void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame)
	{
		rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0));

		std::string label = cv::format("%.2f", conf);

		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		top = max(top, labelSize.height);
		cv::rectangle(frame, cv::Point(left, top - labelSize.height),
			cv::Point(left + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
		putText(frame, label, Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
	}

	//Detects the singular, most likely face in picture.
	void BlinkDetector::detectFace(const cv::Mat &frame) {
		cv::Mat img = frame.clone();
		const std::string configFile = util::resolveRootPath("./config/face/deploy.prototxt");
		const std::string weightFile = util::resolveRootPath("./config/face/res10_300x300_ssd_iter_140000.caffemodel");
		cv::dnn::Net net = cv::dnn::readNetFromCaffe(configFile, weightFile);
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_MYRIAD);
		cv::Mat inputBlob = cv::dnn::blobFromImage(img, 1, cv::Size(160,120), cv::Scalar(104.0, 177.0, 123.0),true, false);

		net.setInput(inputBlob, "data");
		cv::Mat output = net.forward("detection_out");

		cv::Mat faces(output.size[2], output.size[3], CV_32F, output.ptr<float>());

		float confidence = 0.0;
		for (int i = 0; i < faces.rows; i++)
		{
			float *data = faces.ptr<float>(i);
			if (data[2] > confidence && data[2] > 0.5) {
				confidence = faces.at<float>(i, 2);
				int x1 = static_cast<int>(data[3] * frame.cols) - frame.cols/20;
				int y1 = static_cast<int>(data[4] * frame.rows) - frame.rows/20;
				int x2 = static_cast<int>(data[5] * frame.cols) + frame.cols/20;
				int y2 = static_cast<int>(data[6] * frame.rows) + frame.rows/20;

				BlinkDetector::humanDetectionBox = cv::Rect(
					cv::Point(std::max(0, x1), std::max(0,y1)), 
					cv::Point(std::min(frame.cols, x2), std::min(frame.rows, y2))
				);
				//drawPred(confidence, x1, y1, x2, y2, img);
			}
		}
		//cout << humanDetectionBox << endl;
		//cv::imshow("image", img);
	}

	static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
	{
		return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
	}

	static void dlibPointToOpenCV(dlib::full_object_detection& S, std::vector<cv::Point2f>& L, double scale = 1.0)
	{
		for (unsigned int i = 0; i<S.num_parts(); ++i)
		{
			L.push_back(cv::Point2f(S.part(i).x()*(1 / scale), S.part(i).y()*(1 / scale)));
		}
	}


	cv::Rect BlinkDetector::find_max_rec(const std::vector<cv::Rect>& found_filtered) {
		int max_size = 0;
		cv::Rect max_rect;
		for (int i = 0; i < found_filtered.size(); i++) {
			cv::Rect r = found_filtered[i];
			if (r.area() > max_size) {
				max_rect = found_filtered[i];
			}

		}
		return max_rect;
	}

	//TODO: Fix; currently <20% accuracy for unknown reasons.
	void BlinkDetector::detectHumanHOG(const cv::Mat& frame) {
		cv::Mat img, original;

		// copy the rgb image where we'll applied the rectangles
		img = frame.clone();

		std::vector<dlib::rectangle> found;
		std::vector<cv::Rect> found_filtered;
		dlib::array2d<dlib::bgr_pixel> dlibImg;
		dlib::assign_image(dlibImg, dlib::cv_image <dlib::bgr_pixel> (img));
		found = faceHOG(dlibImg);
		dlib::assign_image(dlibImg, dlib::cv_image<dlib::bgr_pixel>(img));
		size_t i, j;
		for (i = 0; i < found.size(); i++) {
			cv::Rect r = dlibRectangleToOpenCV(found[i]);
			for (j = 0; j < found.size(); j++) {
				if (j != i && (r & dlibRectangleToOpenCV(found[j])) == r) {
					break;
				}
			}
			if (j == found.size()) {
				found_filtered.push_back(r);
			}
		}

		cv::Rect max_rect;
		max_rect = find_max_rec(found_filtered);

		if (max_rect.area() > 0) {
			max_rect.x += cvRound(max_rect.width*0.1);
			max_rect.width = cvRound(max_rect.width*0.8);
			max_rect.y += cvRound(max_rect.height*0.06);
			max_rect.height = cvRound(max_rect.height*0.9);
			cv::rectangle(original, max_rect.tl(), max_rect.br(), cv::Scalar(0, 255, 0), 2);

		}

		BlinkDetector::humanDetectionBox = max_rect;
	}

	void BlinkDetector::visualizeBlink(cv::Mat & rgbMap){
		for (int k = 0; k < l_eye_pts.size(); k++) {
			cv::circle(rgbMap, l_eye_pts[k], 1, cv::Scalar(0, 0, 255));
		}

		for (int l = 0; l < r_eye_pts.size(); l++) {
			cv::circle(rgbMap, r_eye_pts[l], 1, cv::Scalar(0, 0, 255));
		}
		char blink_str[20], ear_str[20];
		sprintf(blink_str, "Total # blinks: %i\n", BlinkDetector::total);
		sprintf(ear_str, "EAR: %.2f\n", BlinkDetector::ear);
		cv::putText(rgbMap, blink_str, cv::Point2f(10, 30),
			cv::FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);
		cv::putText(rgbMap, ear_str, cv::Point2f(300, 30),
			cv::FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);
		cv::imshow("RGB", rgbMap);
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
		detectBlinkOpenCV(image);
	}

	int BlinkDetector::getTotal() {
		return BlinkDetector::total;
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

	void BlinkDetector::detectBlinkDLib(cv::Mat &rgbMap) {
		BlinkDetector::l_eye_pts.clear();
		BlinkDetector::r_eye_pts.clear();
		dlib::array2d<dlib::bgr_pixel> img;
		dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(rgbMap));
		
		std::vector<dlib::rectangle> faces = BlinkDetector::fDetector(img);
		if (faces.size() == 0)
			return;
		dlib::full_object_detection shapes = BlinkDetector::eyeDetector(img, faces[0]);
		if (shapes.num_parts() == 22) {
			for (int i = 2; i < 8; i++) {
				BlinkDetector::l_eye_pts.push_back(cv::Point2f(shapes.part(i).x(), shapes.part(i).y()));
			}
			for (int j = 8; j < 15; j++) {
				if (j != 12) // Skip eyebrow index.
					BlinkDetector::r_eye_pts.push_back(cv::Point2f(shapes.part(j).x(), shapes.part(j).y()));
			}
		}
		else {
			throw("Landmark detection failed!\n");
		}
		float left_EAR = BlinkDetector::getEyeAspectRatio(l_eye_pts);
		float right_EAR = BlinkDetector::getEyeAspectRatio(r_eye_pts);
		BlinkDetector::ear = (left_EAR + right_EAR) / 2.0;
		std::cout << "Eye aspect ratio: " << BlinkDetector::ear << std::endl;
		ears.push_back(ear);
		if (ears.size() > 13) {
			ears.erase(ears.begin());
			cv::Mat ear_window = cv::Mat(ears);
			cv::resize(ear_window, ear_window, cv::Size(13, 1));
			if (BlinkDetector::classify(ear_window)) {
				BlinkDetector::total++;
			}
		}
		//OBSOLETE: Thresholding eyeblink detection
		//if (BlinkDetector::ear < EYE_AR_THRESH) {
		//	BlinkDetector::consecBlinkCounter++;
		//}
		//else {
		//	if (BlinkDetector::consecBlinkCounter >= EYE_AR_CONSEC_FRAMES) {
		//		BlinkDetector::total++;
		//	}
		//	BlinkDetector::consecBlinkCounter = 0;
		//}
	}

	void BlinkDetector::detectBlinkOpenCV(cv::Mat &rgbMap) {
		BlinkDetector::l_eye_pts.clear();
		BlinkDetector::r_eye_pts.clear();
		BlinkDetector::detectFace(rgbMap);
		cv::Mat frame = rgbMap.clone();
		if (BlinkDetector::humanDetectionBox.area() <= 0 || !((BlinkDetector::humanDetectionBox & cv::Rect(0, 0, frame.cols, frame.rows)) == BlinkDetector::humanDetectionBox)) {
			return;
		}
		frame = frame(BlinkDetector::humanDetectionBox);
		cv::rectangle(rgbMap, BlinkDetector::humanDetectionBox.tl(), BlinkDetector::humanDetectionBox.br(), cv::Scalar(0, 255, 0), 2, 4);
		cv::Mat gray;
		std::vector<cv::Rect> faces;
		std::vector<std::vector<cv::Point2f>> landmarks;

		cv::Size wholesize;
		cv::Point offset;
		frame.locateROI(wholesize, offset);
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		equalizeHist(gray, gray);
		faceDetector.detectMultiScale(gray, faces);
		if (!(facemark->fit(gray, faces, landmarks))) {
			return;
		}
		
		if (landmarks[0].size() == 68) {
			for (int i = 36; i < 42; i++) {
				BlinkDetector::l_eye_pts.push_back(landmarks[0][i] + cv::Point2f(offset));
			}
			for (int j = 42; j < 48; j++) {
				BlinkDetector::r_eye_pts.push_back(landmarks[0][j] + cv::Point2f(offset));
			}
		}
		else {
			throw("Landmark detection failed!\n");
		}
		float left_EAR = BlinkDetector::getEyeAspectRatio(l_eye_pts);
		float right_EAR = BlinkDetector::getEyeAspectRatio(r_eye_pts);
		BlinkDetector::ear = (left_EAR + right_EAR) / 2.0;
		std::cout << "Eye aspect ratio: " << BlinkDetector::ear << std::endl;
	/*	if (BlinkDetector::ear < EYE_AR_THRESH) {
			BlinkDetector::consecBlinkCounter++;
		}
		else {
			if (BlinkDetector::consecBlinkCounter >= EYE_AR_CONSEC_FRAMES) {
				BlinkDetector::total++;
			}
			BlinkDetector::consecBlinkCounter = 0;
		}*/
		ears.push_back(ear);
		if (ears.size() > 13) {
			ears.erase(ears.begin());
			cv::Mat ear_window = cv::Mat(ears);
			cv::resize(ear_window, ear_window, cv::Size(13, 1));
			if (BlinkDetector::classify(ear_window)) {
				ears.clear();
				BlinkDetector::total++;
			}
		}
	}
}

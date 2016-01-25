#include <opencv2\opencv.hpp>
#include "SVMLight\SvmLightLib.h"

#ifndef __linux
#include <io.h> 
#define access _access_s
#else
#include <unistd.h>
#endif

#define POSITIVE_TRAINING_SET_PATH "DATASET\\POSITIVE\\"
#define NEGATIVE_TRAINING_SET_PATH "DATASET\\NEGATIVE\\"
#define CLASSIFIER_FILE "classifier.dat"
#define FEATURE_VECTORS_FILE "features.dat"
#define WINDOW_NAME "WINDOW"
#define TRAFFIC_VIDEO_FILE "traffic.avi"

using namespace std; 
using namespace cv;
using namespace SVMLight;

// Reference: http://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
bool FileExists(const std::string &Filename)
{
	return access(Filename.c_str(), 0) == 0;
}

// Reference: https://github.com/smart-make/opencv/blob/master/samples/cpp/peopledetect.cpp
void filter_rects(const std::vector<cv::Rect>& candidates, std::vector<cv::Rect>& objects)
{
	size_t i, j;
	for (i = 0; i < candidates.size(); ++i)
	{
		cv::Rect r = candidates[i];

		for (j = 0; j < candidates.size(); ++j)
			if (j != i && (r & candidates[j]) == r)
				break;

		if (j == candidates.size()) {			
			objects.push_back(r);
		}
	}
}

// Reference: https://github.com/Itseez/opencv/blob/master/samples/cpp/train_HOG.cpp
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color)
{
	if (!locations.empty())
	{
		vector< Rect >::const_iterator loc = locations.begin();
		vector< Rect >::const_iterator end = locations.end();
		for (; loc != end; ++loc)
		{
			rectangle(img, *loc, color, 2);
		}
	}
}

vector<string> GetFilePathsInDirectory(const char* directory)
{
	vector<string> vFiles;
	FILE* pipe = NULL;
	string full_path = "dir /B /S " + string(directory);
	char buf[256];

	if (pipe = _popen(full_path.c_str(), "rt"))
		while (!feof(pipe))
			if (fgets(buf, 256, pipe) != NULL) {
				buf[strlen(buf) - 1] = '\0';
				vFiles.push_back(string(buf));
			}

	_pclose(pipe);
	return vFiles;
}

int main() {

	HOGDescriptor hog;
	hog.winSize = Size(96, 160);

	/******************************TRAIN SVM******************************/
	vector<float> vFeatures;
	vector<const char*> vDirectories;
	vector<string> vFiles;
	Mat mImage;

	if (!FileExists(FEATURE_VECTORS_FILE) || !FileExists(CLASSIFIER_FILE))
	{
		// Feed SVM with feature vectors obtained by HOG
		SVMTrainer svm(FEATURE_VECTORS_FILE);
		vDirectories.push_back(POSITIVE_TRAINING_SET_PATH);
		vDirectories.push_back(NEGATIVE_TRAINING_SET_PATH);

		for each (const char* dir in vDirectories)
		{
			// true = positive sample, false = negative sample
			bool bSample = (dir == POSITIVE_TRAINING_SET_PATH) ? true : false;
			vFiles = GetFilePathsInDirectory(dir);

			for each (string file in vFiles) {

				mImage = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
				hog.compute(mImage, vFeatures, Size(8, 8), Size(0, 0));
				svm.writeFeatureVectorToFile(vFeatures, bSample);
				vFeatures.clear();
				mImage.release();
			}
		}

		svm.trainAndSaveModel(CLASSIFIER_FILE);
	}
	/************************************************************/

	vector<Rect> vFound, vFiltered;
	Mat mRawTraffic, mProcessedTraffic;

	SVMClassifier classifier(CLASSIFIER_FILE);
	vector<float> vDescriptor = classifier.getDescriptorVector();
	hog.setSVMDetector(vDescriptor);

	VideoCapture cap(TRAFFIC_VIDEO_FILE);

	while (true) {

		cap >> mRawTraffic;
		GaussianBlur(mRawTraffic, mProcessedTraffic, Size(3, 3), 2, 2);

		// Eliminate ingoing traffic
		for (int pi = 0; pi < mProcessedTraffic.rows; ++pi)
			for (int pj = 0; pj < mProcessedTraffic.cols; ++pj)
				if (pj > mProcessedTraffic.cols / 2) {
					mProcessedTraffic.at<Vec3b>(pi, pj)[0] = 0;
					mProcessedTraffic.at<Vec3b>(pi, pj)[1] = 0;
					mProcessedTraffic.at<Vec3b>(pi, pj)[2] = 0;
				}
		cvtColor(mProcessedTraffic, mProcessedTraffic, CV_BGR2GRAY);

		vFound.clear();
		vFiltered.clear();

		hog.detectMultiScale(mProcessedTraffic, vFound, 0, Size(0, 0), Size(2, 2), 1.05, 2);
		filter_rects(vFound, vFiltered);
		draw_locations(mRawTraffic, vFiltered, Scalar(0,255,0));

		imshow(WINDOW_NAME, mRawTraffic);
		waitKey(10);
	}

	return 0;
}

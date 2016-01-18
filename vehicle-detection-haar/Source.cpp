#include <opencv2\opencv.hpp>
using namespace cv;

// Reference: https://github.com/Itseez/opencv/blob/master/samples/cpp/train_HOG.cpp
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color);

#define VIDEO_FILE_NAME "xxx.mp4"
#define CASCADE_FILE_NAME "cars3.xml"
#define WINDOW_NAME "WINDOW"

int main()
{
	VideoCapture cap;
	Mat mFrame, mGray;
	CascadeClassifier classifier;
	vector<Rect> vFound;

	classifier.load(CASCADE_FILE_NAME);
	cap.open(VIDEO_FILE_NAME);

	while (cap.read(mFrame))
	{
		// Apply the classifier to the frame
		cvtColor(mFrame, mGray, COLOR_BGR2GRAY);
		equalizeHist(mGray, mGray);
		classifier.detectMultiScale(mGray, vFound, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		draw_locations(mFrame, vFound, Scalar(0, 255, 0));
		imshow(WINDOW_NAME, mFrame);

		waitKey(10);
	}

	return 0;
}


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
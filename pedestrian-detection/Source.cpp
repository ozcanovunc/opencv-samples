#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define INPUT_VIDEO_FILE "pedestrian.mp4"
#define WINDOW_NAME "WINDOW"

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

int main(int argc, char** argv)
{
	Mat mSrc;
	vector<Rect> vDetected;
	HOGDescriptor hog;

	static vector<float> detector = HOGDescriptor::getDefaultPeopleDetector();
	hog.setSVMDetector(detector);

	VideoCapture cap(INPUT_VIDEO_FILE);

	while (true) {

		cap >> mSrc;
		hog.detectMultiScale(mSrc, vDetected, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		draw_locations(mSrc, vDetected, Scalar(0, 255, 0));

		imshow(WINDOW_NAME, mSrc);
		waitKey(10);
	}

	return 0;
}
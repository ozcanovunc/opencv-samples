#include <opencv2\opencv.hpp>
#include <iostream>

#define _CRT_SECURE_NO_WARNINGS
#define MIN_DIFFERENCE_TO_BE_CORNER 10
#define WINDOW_NAME "CAM"

using namespace cv;
using namespace std;

int	iFiducialType1 = 0,
	iFiducialType2 = 0,
	iFiducialType3 = 0;

static void WindowClickedEvent(int event, int x, int y, int flags, void* userdata) {

	Mat	image;

	if (event == EVENT_LBUTTONDOWN) {

		image = *(Mat*)userdata;
		imshow("", image);
		cout << "Type 1 fiducial: " << iFiducialType1 << endl;
		cout << "Type 2 fiducial: " << iFiducialType2 << endl;
		cout << "Type 3 fiducial: " << iFiducialType3 << endl << endl;
	}
}

static bool IsCircularContour(vector<Point> contour) {

	bool	bIsCircular = true;
	int	iPointCount = contour.size();;

	if (!contour.empty()) {

		for (int pi = 0; pi < iPointCount - 1; ++pi)
			if (abs(contour[pi].x - contour[pi + 1].x) > MIN_DIFFERENCE_TO_BE_CORNER
				|| abs(contour[pi].y - contour[pi + 1].y) > MIN_DIFFERENCE_TO_BE_CORNER)
				bIsCircular = false;
	}

	return bIsCircular;
}

// Very dangerous to change. Functionality depends on the fudicial we're working on.
static int GetInnermostRectContourIndex
(vector<vector<Point> > contours, vector<Vec4i> hierarchy, int parent_contour_index) {

	int	iFirstInnerRect,
		iSecondInnerRect,
		iThirdInnerRect,
		iInnermostRect = -1;

	if (hierarchy.empty())
		return iInnermostRect;

	if (hierarchy[parent_contour_index][2] != -1) {
		iFirstInnerRect = hierarchy[hierarchy[parent_contour_index][2]][2];
		if (iFirstInnerRect != -1 && hierarchy[iFirstInnerRect][2] != -1) {
			iSecondInnerRect = hierarchy[hierarchy[iFirstInnerRect][2]][2];
			if (iSecondInnerRect != -1 && hierarchy[iSecondInnerRect][2] != -1) {
				iThirdInnerRect = hierarchy[hierarchy[iSecondInnerRect][2]][2];
				if (iThirdInnerRect != -1 && hierarchy[iThirdInnerRect][2] != -1
					&& !IsCircularContour(contours[hierarchy[iThirdInnerRect][2]])) {
					iInnermostRect = hierarchy[iThirdInnerRect][2];
				}
			}
		}
	}

	return iInnermostRect;
}

int main(int argc, char** argv)
{
	Mat	mSrc,
		mGray,
		mBin,
		mEdges;

	vector<vector<Point> > vContours;
	vector<Vec4i> vHierarchy;

	VideoCapture cap(0);

	namedWindow("CAM", CV_WINDOW_AUTOSIZE);
	setMouseCallback("CAM", WindowClickedEvent, &mSrc);

	while (true) {

		cap >> mSrc;

		cvtColor(mSrc, mGray, CV_BGR2GRAY);
		threshold(mGray, mBin, 100, 255, 0);
		GaussianBlur(mBin, mBin, Size(9, 9), 2, 2);
		Canny(mBin, mEdges, 100, 400);

		// RETR_TREE: Creates a full family hierarchy list.
		// CV_CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, 
		// and diagonal segments and leaves only their end points.
		// CV_CHAIN_APPROX_NONE: stores absolutely all the contour points.
		// Hierarchy: [Next, Previous, First_Child, Parent]
		findContours(mEdges.clone(), vContours, vHierarchy, RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

		iFiducialType1 = iFiducialType2 = iFiducialType3 = 0;

		for (int ci = 0; ci < vContours.size(); ++ci)
		{
			// Find the outermost box iff there is no parent.
			if (vHierarchy[ci][3] == -1) {

				int iInnermostRect = GetInnermostRectContourIndex(vContours, vHierarchy, ci);

				// Check if the current contour is the type of fiducial we're looking for.
				// iInnermostRect denotes innermost rectangle's index
				if (iInnermostRect != -1) {

					// Type 1 fiducial - No circles' been detected. (No child as well)
					if (vHierarchy[iInnermostRect][2] == -1) {
						drawContours(mSrc, vContours, ci, Scalar(0, 0, 255), 2);
						++iFiducialType1;
					}

					// There are some contours inside the innermost quadrilateral, check if it's a circle
					else if (IsCircularContour(vContours[vHierarchy[iInnermostRect][2]])) {

						// Type 2 fiducial - only one circle's been detected.
						// vHierarchy[iInnermostRect][2] denotes circular contour's index
						if (vHierarchy[vHierarchy[iInnermostRect][2]][0] == -1) {
							drawContours(mSrc, vContours, ci, Scalar(0, 255, 0), 2);
							++iFiducialType2;
						}

						// Type 3 fiducial - Two circles' been detected.
						// vHierarchy[iInnermostRect][2] and vHierarchy[vHierarchy[iInnermostRect][2]][0] 
						// denote circular contours' indexes
						else if (IsCircularContour(vContours[vHierarchy[vHierarchy[iInnermostRect][2]][0]])) {
							drawContours(mSrc, vContours, ci, Scalar(255, 0, 0), 2);
							++iFiducialType3;
						}
					}
				}
			}
		}

		imshow(WINDOW_NAME, mSrc);
		waitKey(20);
	}

	return 0;
}

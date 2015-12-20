#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdlib.h>

#define _CRT_SECURE_NO_WARNINGS
#define MIN_DIFFERENCE_TO_BE_CORNER 10
#define HEIGHT_OF_3D_CONTOUR 0.5
#define MAX_CHESS_SCENE_FOR_CALIBRATION 10

using namespace cv;
using namespace std;

int	iFiducialType1 = 0,
	iFiducialType2 = 0,
	iFiducialType3 = 0;

static void WindowClickedEvent(int event, int x, int y, int flags, void* userdata) {
	
	Mat	image;

	if (event == EVENT_LBUTTONDOWN) {

		image = *(Mat*)userdata;
		imshow(NULL, image);
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

vector<Point2d> GetCornerPointsOfContour(vector<Point> contour) {

	if (contour.empty())
		return vector<Point2d>();

	Rect rect = boundingRect(contour);
	vector<Point2d> corners;

	// Top left, top right, bottom left, bottom right
	corners.push_back(rect.tl());
	corners.push_back(Point2d(rect.tl().x + rect.width, rect.tl().y));
	corners.push_back(rect.br());
	corners.push_back(Point2d(rect.br().x - rect.width, rect.br().y));

	return corners;
}

bool DrawCubeBetweenContours(InputOutputArray &image, vector<Point2d> rhs, vector<Point2d> lhs, Scalar color, int thickness) {

	if (rhs.empty() || lhs.empty() || rhs.size() != 4 || lhs.size() != 4)
		return false;

	line(image, rhs[0], rhs[1], color, thickness);
	line(image, rhs[1], rhs[2], color, thickness);
	line(image, rhs[2], rhs[3], color, thickness);
	line(image, rhs[3], rhs[0], color, thickness);
	line(image, lhs[0], lhs[1], color, thickness);
	line(image, lhs[1], lhs[2], color, thickness);
	line(image, lhs[2], lhs[3], color, thickness);
	line(image, lhs[3], lhs[0], color, thickness);
	line(image, rhs[0], lhs[0], color, thickness);
	line(image, rhs[1], lhs[1], color, thickness);
	line(image, rhs[2], lhs[2], color, thickness);
	line(image, rhs[3], lhs[3], color, thickness);

	return true;
}

int main(int argc, char** argv)
{
	/***** PARAMETERS FOR CAMERA CALIBRATION AND 3D POSE ESTIMATION *****/

	int iIteration = 0,
		iBoardHeight = 6,
		iBoardWidth = 9,
		iNumSquares = iBoardHeight * iBoardWidth;
	Size board_size = Size(iBoardHeight, iBoardWidth);

	Mat mDistortion(1, 5, DataType<double>::type);
	Mat mIntrinsic = Mat(3, 3, CV_64FC1);
	Mat rvec = Mat(Size(3, 1), CV_64F);
	Mat tvec = Mat(Size(3, 1), CV_64F);

	vector<Point2d> vProjectedCorners;
	vector<Point3d> vContourCorners;
	vector<Point3d> v3DContourCorners;

	vector<vector<Point3f> > vObjectPoints;
	vector<vector<Point2f> > vImagePoints;
	vector<Point2f> vCornerPoints;
	vector<Point3f> vObj;
	vector<Mat> vRvecs;
	vector<Mat> vTvecs;

	/*****************************************************************/

	Mat	mSrc,
		mGray,
		mBin,
		mEdges;

	vector<vector<Point> > vContours;
	vector<Vec4i> vHierarchy;

	VideoCapture cap(0);

	namedWindow("CAM", CV_WINDOW_AUTOSIZE);
	namedWindow("CALIBRATION", CV_WINDOW_AUTOSIZE);
	setMouseCallback("CAM", WindowClickedEvent, &mSrc);

	vContourCorners.push_back(Point3d(0.0, 1.0, 0.0));
	vContourCorners.push_back(Point3d(1.0, 1.0, 0.0));
	vContourCorners.push_back(Point3d(1.0, 0.0, 0.0));
	vContourCorners.push_back(Point3d(0.0, 0.0, 0.0));

	v3DContourCorners.push_back(Point3d(0.0, 1.0, HEIGHT_OF_3D_CONTOUR));
	v3DContourCorners.push_back(Point3d(1.0, 1.0, HEIGHT_OF_3D_CONTOUR));
	v3DContourCorners.push_back(Point3d(1.0, 0.0, HEIGHT_OF_3D_CONTOUR));
	v3DContourCorners.push_back(Point3d(0.0, 0.0, HEIGHT_OF_3D_CONTOUR));

	/************************* CAMERA CALIBRATION *************************/

	for (int i = 0; i < iBoardWidth; ++i)
		for (int j = 0; j < iBoardHeight; ++j)
			vObj.push_back(Point3f(float(j * iNumSquares), float(i * iNumSquares), 0.0));

	cout << endl << "CAMERA CALIBRATION" << endl << endl;

	while (true)
	{
		cap >> mSrc;
		cvtColor(mSrc, mGray, CV_BGR2GRAY);

		bool bFound = findChessboardCorners
			(mSrc, board_size, vCornerPoints,
				CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		if (bFound)
		{
			cornerSubPix(mGray, vCornerPoints, Size(11, 11), Size(-1, -1),
				TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
			vImagePoints.push_back(vCornerPoints);
			vObjectPoints.push_back(vObj);

			if (++iIteration == MAX_CHESS_SCENE_FOR_CALIBRATION) {
				destroyWindow("CALIBRATION");
				break;
			}

			cout << MAX_CHESS_SCENE_FOR_CALIBRATION - iIteration << " LEFT" << endl;
		}

		imshow("CALIBRATION", mSrc);
		waitKey(20);
	}

	calibrateCamera(vObjectPoints, vImagePoints, mSrc.size(), 
		mIntrinsic, mDistortion, vRvecs, vTvecs);

	cout << endl << "K MATRIX" << endl << endl << mIntrinsic << endl;

	/************************* FIDUCIAL DETECTION *************************/

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

						solvePnP(vContourCorners, GetCornerPointsOfContour(
							vContours[ci]), mIntrinsic, mDistortion, rvec, tvec, false);
						projectPoints(v3DContourCorners, rvec, tvec, mIntrinsic,
							mDistortion, vProjectedCorners);
						DrawCubeBetweenContours(mSrc, GetCornerPointsOfContour(
							vContours[ci]), vProjectedCorners, Scalar(0, 0, 255), 2);
						++iFiducialType1;
					}

					// There are some contours inside the innermost quadrilateral, check if it's a circle
					else if (IsCircularContour(
						vContours[vHierarchy[iInnermostRect][2]])) {

						// Type 2 fiducial - only one circle's been detected.
						// vHierarchy[iInnermostRect][2] denotes circular contour's index
						if (vHierarchy[vHierarchy[iInnermostRect][2]][0] == -1) {

							solvePnP(vContourCorners, GetCornerPointsOfContour(
								vContours[ci]), mIntrinsic, mDistortion, rvec, tvec, false);
							projectPoints(v3DContourCorners, rvec, tvec, mIntrinsic, 
								mDistortion, vProjectedCorners);
							DrawCubeBetweenContours(mSrc, GetCornerPointsOfContour(
								vContours[ci]), vProjectedCorners, Scalar(0, 255, 0), 2);
							++iFiducialType2;
						}

						// Type 3 fiducial - Two circles' been detected.
						// vHierarchy[iInnermostRect][2] and vHierarchy[vHierarchy[iInnermostRect][2]][0] 
						// denote circular contours' indexes
						else if (IsCircularContour(
							vContours[vHierarchy[vHierarchy[iInnermostRect][2]][0]])){

							solvePnP(vContourCorners, GetCornerPointsOfContour(
								vContours[ci]), mIntrinsic, mDistortion, rvec, tvec, false);
							projectPoints(v3DContourCorners, rvec, tvec, mIntrinsic, 
								mDistortion, vProjectedCorners);
							DrawCubeBetweenContours(mSrc, GetCornerPointsOfContour(
								vContours[ci]), vProjectedCorners, Scalar(255, 0, 0), 2);
							++iFiducialType3;
						}
					}
				}
			}
		}

		imshow("CAM", mSrc);
		waitKey(20);
	}

	return 0;
}

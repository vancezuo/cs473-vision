/*
 * cs473vision.cpp
 *
 *  Created on: Feb 18, 2014
 *      Author: Vance Zuo
 */

#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

// Main
int main(int argc, char* argv[]) {
	Mat image;
	Mat blurred;
	Mat gray_blurred;
	Mat thres_otsu;
	Mat watershed_fg;
	Mat watershed_bg;
	Mat watershed_marker;
	Mat watershed_marker32;
	Mat watershed_result;
	Mat watershed_outside;
	Mat watershed_segment;
	Mat watershed_object;
	Mat grabcut_bg;
	Mat grabcut_fg;
	Mat grabcut_result;

	// Load Image
	image = imread(argv[1]);
	if (!image.data) {
		cout << "No image data" << endl;
		return -1;
	}

	// Blur Image
	medianBlur(image, blurred, 5);

	// Create Watershed Algorithm Marker
	cvtColor(blurred, gray_blurred, COLOR_BGR2GRAY);
	threshold(gray_blurred, thres_otsu, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

	erode(thres_otsu, watershed_fg, Mat(), Point(-1, -1), 2);

	dilate(thres_otsu, watershed_bg, Mat(), Point(-1, -1), 3);
	threshold(watershed_bg, watershed_bg, 0, 128, THRESH_BINARY_INV);

	add(watershed_fg, watershed_bg, watershed_marker);

	// Apply Watershed Algorithm
	watershed_marker.convertTo(watershed_marker32, CV_32SC1);
	watershed(blurred, watershed_marker32);
	convertScaleAbs(watershed_marker32, watershed_result);

	// Extract Center Segment of Watershed Result
	watershed_outside = watershed_result.clone();
	floodFill(watershed_outside, Point(image.cols/2, image.rows/2), Scalar(0.0, 0.0, 0.0));
	bitwise_not(watershed_outside, watershed_outside);
	bitwise_and(watershed_outside, watershed_result, watershed_segment);

	bitwise_and(image, image, watershed_object, watershed_segment);

	// Create Bounding Rectangles for Watershed Result
	vector<Point> watershed_points;
	Mat_<uchar>::iterator watershed_it = watershed_segment.begin<uchar>();
	Mat_<uchar>::iterator watershed_end = watershed_segment.end<uchar>();
	for (; watershed_it != watershed_end; ++watershed_it) {
		if (*watershed_it) watershed_points.push_back(watershed_it.pos());
	}
	RotatedRect watershed_box_min = minAreaRect(Mat(watershed_points));
	Rect watershed_box = boundingRect(Mat(watershed_points));

	// Apply Grabcut Algorithm (using Watershed Result as parameter)
	grabCut(image, grabcut_result, watershed_box, grabcut_bg, grabcut_fg, 1, 0);

	compare(grabcut_result, GC_PR_FGD, grabcut_result, CMP_EQ);

	Mat grabcut_object(image.size(), CV_8UC3, Scalar(0, 0, 0));
	image.copyTo(grabcut_object, grabcut_result);

	// Create Bounding Rectangles for Grabcut Result
	vector<Point> grabcut_points;
	Mat_<uchar>::iterator grabcut_it = grabcut_result.begin<uchar>();
	Mat_<uchar>::iterator grabcut_end = grabcut_result.end<uchar>();
	for (; grabcut_it != grabcut_end; ++grabcut_it) {
		if (*grabcut_it) grabcut_points.push_back(grabcut_it.pos());
	}
	RotatedRect grabcut_box_min = minAreaRect(Mat(grabcut_points));
	Rect grabcut_box = boundingRect(Mat(grabcut_points));

	// Draw Bounding Rectangles
	Point2f watershed_rect_points[4];
	Point2f grabcut_rect_points[4];
	watershed_box_min.points(watershed_rect_points);
	grabcut_box_min.points(grabcut_rect_points);
	Scalar color = Scalar(196.0, 2.0, 51.0);
	for( int j = 0; j < 4; j++ ) {
		line(watershed_object, watershed_rect_points[j], watershed_rect_points[(j+1)%4], color, 1, 8);
		line(grabcut_object, grabcut_rect_points[j], grabcut_rect_points[(j+1)%4], color, 1, 8);
	}
	rectangle(watershed_object, watershed_box, color);
	rectangle(grabcut_object, grabcut_box, color);

	// Print Bounding Rectangle Sizes
	cout << "Image: " << image.cols << " x " << image.rows << endl;
	cout << "---------" << endl;
	cout << "Upright Box: " << watershed_box.width << " x " << watershed_box.height << endl;
	cout << "Min Box: " << watershed_box_min.size.width << " x " << watershed_box_min.size.height << endl;
	cout << "---------" << endl;
	cout << "Upright Box: " << grabcut_box.width << " x " << grabcut_box.height << endl;
	cout << "Min Box: " << grabcut_box_min.size.width << " x " << grabcut_box_min.size.height << endl;

	// Display Steps
	namedWindow("Image", WINDOW_AUTOSIZE);
	imshow("Image", image);
	waitKey(0);
	imshow("Image", blurred);
	waitKey(0);
	imshow("Image", gray_blurred);
	waitKey(0);
	imshow("Image", thres_otsu);
	waitKey(0);
	imshow("Image", watershed_fg);
	waitKey(0);
	imshow("Image", watershed_bg);
	waitKey(0);
	imshow("Image", watershed_marker);
	waitKey(0);
	imshow("Image", watershed_result);
	waitKey(0);
	imshow("Image", watershed_segment);
	waitKey(0);
	imshow("Image", watershed_object);
	waitKey(0);
	imshow("Image", grabcut_result);
	waitKey(0);
	imshow("Image", grabcut_object);
	waitKey(0);

	return 0;
}

/*
 * watershed.cpp
 *
 *  Created on: Feb 18, 2014
 *      Author: Vance Zuo
 */

#include "opencv2/opencv.hpp"
#include <iostream>

// Main
int main(int argc, char* argv[]) {
	cv::Mat image = cv::imread(argv[1]);
	if (!image.data) {
		std::cout << "No image data" << std::endl;
		return -1;
	}

	cv::Mat smooth;
	cv::medianBlur(image, smooth, 5);

	cv::Mat gray, thres, foreground, background, marker, marker32, m;

	cv::cvtColor(smooth, gray, cv::COLOR_BGR2GRAY);
	cv::threshold(gray, thres, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

	cv::erode(thres, foreground, cv::Mat(), cv::Point(-1, -1), 2);

	cv::dilate(thres, background, cv::Mat(), cv::Point(-1, -1), 3);
	cv::threshold(background, background, 0, 128, cv::THRESH_BINARY_INV);

	cv::add(foreground, background, marker);

	marker.convertTo(marker32, CV_32SC1);
	cv::watershed(smooth, marker32);
	cv::convertScaleAbs(marker32, m);

	cv::Mat center, outside;
	outside = m.clone();
	cv::floodFill(outside, cv::Point(image.cols / 2, image.rows / 2), cv::Scalar(0.0, 0.0, 0.0));
	cv::bitwise_not(outside, outside);
	cv::bitwise_and(outside, m, center);

	cv::Mat result;
	cv::bitwise_and(image, image, result, center);

	std::vector<cv::Point> points;
	cv::Mat_<uchar>::iterator it = center.begin<uchar>();
	cv::Mat_<uchar>::iterator end = center.end<uchar>();
	for (; it != end; ++it)
	{
		if (*it) points.push_back(it.pos());
	}
	cv::RotatedRect box_min = cv::minAreaRect(cv::Mat(points));
	cv::Rect box = cv::boundingRect(cv::Mat(points));

	cv::Point2f rect_points[4];
	box_min.points(rect_points);
	cv::Scalar color = cv::Scalar(196.0, 2.0, 51.0);
	for( int j = 0; j < 4; j++ ) {
		cv::line(result, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
	}
	cv::rectangle(result, box, color);

	std::cout << "Image: " << image.size().width << " x " << image.size().height << std::endl;
	std::cout << "Upright Box: " << box.width << " x " << box.height << std::endl;
	std::cout << "Min Box: " << box_min.size.width << " x " << box_min.size.height << std::endl;

	cv::Mat result_grabcut, bg_grabcut, fg_grabcut;
	cv::grabCut(image, result_grabcut, box, bg_grabcut, fg_grabcut, 1, 0);

	cv::compare(result_grabcut, cv::GC_PR_FGD, result_grabcut, cv::CMP_EQ);

	cv::Mat foreground_grabcut(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	image.copyTo(foreground_grabcut, result_grabcut);

	cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Image", image);
	cv::waitKey(0);
	cv::imshow("Image", smooth);
	cv::waitKey(0);
	cv::imshow("Image", gray);
	cv::waitKey(0);
	cv::imshow("Image", foreground);
	cv::waitKey(0);
	cv::imshow("Image", background);
	cv::waitKey(0);
	cv::imshow("Image", marker);
	cv::waitKey(0);
	cv::imshow("Image", m);
	cv::waitKey(0);
	cv::imshow("Image", center);
	cv::waitKey(0);
	cv::imshow("Image", result);
	cv::waitKey(0);
	cv::imshow("Image", result_grabcut);
	cv::waitKey(0);
	cv::imshow("Image", foreground_grabcut);
	cv::waitKey(0);

	return 0;
}

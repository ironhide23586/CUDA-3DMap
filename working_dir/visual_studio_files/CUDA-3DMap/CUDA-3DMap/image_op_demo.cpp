#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void show(Mat& im) {
	cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
	cv::imshow("image", im);
	cv::waitKey();
}

int main() {
	Mat im;
	im = cv::imread("img.jpg");
	show(im);
}
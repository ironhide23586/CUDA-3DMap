#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SHMatrix/SHMatrix.h"
#include "SHMage.h"

using namespace std;
using namespace cv;

//void show(cv::Mat& im) {
//	cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
//	cv::imshow("image", im);
//	cv::waitKey();
//}

int main() {
	Mat im;
	im = cv::imread("img.jpg");

	cublasHandle_t cublasHandle;
	CublasSafeCall(cublasCreate_v2(&cublasHandle));

	SHMage a(cublasHandle, im, GPU);
	*a.im += 50;
	a.show();
}
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SHMatrix/SHMatrix.h"

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

	cublasHandle_t cublasHandle;
	CublasSafeCall(cublasCreate_v2(&cublasHandle));

	int n = im.rows * im.cols * im.channels();
	float *x = (float *)malloc(sizeof(float) * n);
	for (int i = 0; i < n; i++) {
		x[i] = (float)im.data[i];
	}


	std::vector<int> a_shp = { im.rows, im.cols, im.channels() };
	SHMatrix a(cublasHandle, x, a_shp, CPU);

	a.Move2GPU();
	a *= 20;
	a.Move2CPU();

	for (int i = 0; i < n; i++) {
		im.data[i] = (uchar)a.data[i];
	}

	show(im);
}
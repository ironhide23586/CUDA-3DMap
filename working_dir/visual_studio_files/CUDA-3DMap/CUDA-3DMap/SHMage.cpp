#include "SHMage.h"

SHMage::SHMage(const cublasHandle_t &cublasHandle, cv::Mat &cv_im,
	mem_location loc)
	: cublasHandle_local(cublasHandle), cv_im_ref(cv_im), data_loc(loc) {
	std::vector<int> shp = { cv_im.channels(), cv_im.rows, cv_im.cols };
	int n = cv_im.channels() * cv_im.rows * cv_im.cols;
	float *x = (float *)malloc(sizeof(float) * n);
	for (int i = 0; i < n; i++) {
		x[i] = (float)cv_im_ref.data[i];
	}
	im = new SHMatrix(cublasHandle_local, x, shp, CPU);
	if (data_loc == GPU) {
		im->Move2GPU();
	}
}

void SHMage::show() {
	im->Move2CPU();
	for (int i = 0; i < im->num_elems; i++) {
		cv_im_ref.data[i] = (uchar)im->data[i];
	}
	if (data_loc == GPU) {
		im->Move2GPU();
	}
	cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
	cv::imshow("image", cv_im_ref);
	cv::waitKey();
}
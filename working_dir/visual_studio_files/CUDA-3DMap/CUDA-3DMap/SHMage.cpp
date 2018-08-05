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

void SHMage::GenerateGaussianKernel(float oneside_dim, float stddev) {
	int side_dim = oneside_dim * 2 + 1;
	int n = side_dim * side_dim;
	float *kernel = (float *)malloc(sizeof(float) * n);
	int idx = 0;
	float sum = 0.0f;
	for (int y = -oneside_dim; y <= oneside_dim; y++) {
		for (int x = -oneside_dim; x <= oneside_dim; x++) {
			float v = (1.0f / (2.0f * M_PI * stddev * stddev))
				* exp(-((x * x) + (y * y)) / (2 * stddev * stddev));
			kernel[idx] = v;
			sum += v;
			idx++;
		} 
	}
	
	for (int idx = 0; idx < n; idx++) {
		kernel[idx] /= sum;
	}
	std::vector<int> shp = { side_dim, side_dim };
	filter_kernel = new SHMatrix(cublasHandle_local, kernel, shp, CPU);
	if (data_loc == GPU) {
		filter_kernel->Move2GPU();
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
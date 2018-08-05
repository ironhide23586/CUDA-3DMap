#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

#include "SHMatrix/SHMatrix.h"

# define M_PI           3.14159265358979323846  /* pi */

using namespace cv;

class SHMage {

public:
	SHMatrix *im, *filter_kernel;
	cublasHandle_t cublasHandle_local;
	mem_location data_loc;
	cv::Mat& cv_im_ref;


	SHMage(const cublasHandle_t &cublasHandle, 
		   cv::Mat &cv_im, mem_location = GPU);

	void GenerateGaussianKernel(float side_dim = 3, float stddev = .9);

	void show();
};


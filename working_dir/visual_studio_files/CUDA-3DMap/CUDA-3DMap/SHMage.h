#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SHMatrix/SHMatrix.h"

using namespace cv;

class SHMage {

public:
	SHMatrix *im;
	cublasHandle_t cublasHandle_local;
	mem_location data_loc;
	cv::Mat& cv_im_ref;

	SHMage(const cublasHandle_t &cublasHandle, 
		   cv::Mat &cv_im, mem_location = GPU);

	void show();
};


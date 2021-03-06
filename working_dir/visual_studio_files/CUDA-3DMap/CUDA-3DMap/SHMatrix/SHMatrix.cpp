//#include "stdafx.h"
#include "SHMatrix.h"

SHMatrix::SHMatrix(const cublasHandle_t &cublas_handle_arg,
                   float *mat_data, std::vector<int> &dims,
                   mem_location loc)
  : cublas_handle(cublas_handle_arg),
    data(mat_data),
    data_dims(dims),
    data_loc(loc) {
  init();
  load_dims(dims);
  allocated = true;
}

SHMatrix::SHMatrix(const cublasHandle_t &cublas_handle_arg,
                   std::vector<int> &dims,
                   mem_location loc, bool default_init,
                   float init_val)
  : cublas_handle(cublas_handle_arg), 
    data_dims(dims),
    data_loc(loc) {
  init();
  load_dims(dims);
  data = allocate_memory();
  if (default_init) {
    init_with_default_value(data, data_loc, init_val);
  }
}

SHMatrix::SHMatrix(const cublasHandle_t &cublas_handle_arg,
                   SHMatrix &src_shmatrix, mem_location loc)
  : cublas_handle(cublas_handle_arg),
    data_loc(loc) {
  init();
  duplicate_shmatrix(src_shmatrix);
}

void SHMatrix::Equate(SHMatrix &src_shmatrix) {
  bool mem_alloc_needed = !(num_elems == src_shmatrix.num_elems);
  if (!mem_alloc_needed)
    reset_metadata();
  else
    Clear();
  duplicate_shmatrix(src_shmatrix, mem_alloc_needed);
}

void SHMatrix::Reallocate(std::vector<int> &dims, mem_location mem_loc,
                          bool copy_original, bool default_init,
                          float init_val) {
  int desired_num_elems = dims[0], original_num_elems = num_elems;
  mem_location original_mem_loc = data_loc;
  float *new_data_ptr;
  for (int i = 1; i < dims.size(); i++) {
    desired_num_elems *= dims[i];
  }
  bool deallocation_flag = false;
  if (num_elems != desired_num_elems || data_loc != mem_loc) {
    deallocation_flag = true;
  }
  init_list_properties();
  init_value_properties();
  data_loc = mem_loc;
  data_dims = dims;
  load_dims(dims);
  if (deallocation_flag) {
    new_data_ptr = allocate_memory();
    if (default_init)
      init_with_default_value(new_data_ptr, mem_loc, init_val);
    if (copy_original) {
      int copy_size = num_elems < original_num_elems
        ? num_elems : original_num_elems;
      copy_data_from(new_data_ptr, data, 
                     mem_loc, original_mem_loc, copy_size);
    }
    deallocate_memory(data, original_mem_loc);
    data = new_data_ptr;
  }
  if (default_init && !deallocation_flag) {
    init_with_default_value(data, data_loc, init_val);
  }
  allocated = true;
}

void SHMatrix::Print(bool print_elems) {
  CommitUnaryOps();
  float *h_v;
  if (data_loc == GPU) {
    h_v = (float *)malloc(sizeof(float) * rows * cols);
    CudaSafeCall(cudaMemcpy(h_v, data, sizeof(float) * rows * cols,
                            cudaMemcpyDeviceToHost));
  }
  else if (data_loc == CPU) {
    h_v = data;
  }
  print_h_var(h_v, rows, cols, print_elems);
  if (data_loc == GPU)
    free(h_v);
}

void SHMatrix::Move2GPU() {
  if (data_loc == GPU)
    return;
  float *d_data;
  CudaSafeCall(cudaMalloc((void **)&d_data, sizeof(float) * num_elems));
  CudaSafeCall(cudaMemcpy(d_data, data, sizeof(float) * num_elems,
                          cudaMemcpyHostToDevice));
  if (allocated)
    free(data);
  data = d_data;
  data_loc = GPU;
}

void SHMatrix::Move2CPU() {
  if (data_loc == CPU)
    return;
  CommitUnaryOps();
  float *h_data = (float *)malloc(sizeof(float) * num_elems);
  CudaSafeCall(cudaMemcpy(h_data, data, sizeof(float) * num_elems,
                          cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaFree(data));
  data = h_data;
  data_loc = CPU;
}

void SHMatrix::Clear() {
  deallocate_memory(data, data_loc);
  allocated = true;
  reset_metadata();
}

void SHMatrix::GaussianInit(float mean, float stddev) {
  if (data_loc == GPU) {
    gaussian_init_gpu(mean, stddev);
  }
  else if (data_loc == CPU) {
    gaussian_init_cpu(mean, stddev);
  }
  allocated = true;
}

void SHMatrix::UniformInit(float mean, float stddev) {
  if (data_loc == GPU) {
    uniform_init_gpu(mean, stddev);
  }
  else if (data_loc == CPU) {
    uniform_init_cpu(mean, stddev);
  }
  allocated = true;
}

float SHMatrix::GetGaussianNum(float mean, float stddev) {
  static std::default_random_engine re;
  static std::normal_distribution<float> dist(mean, stddev);
  return dist(re);
}

float SHMatrix::GetUniformNum(float lower, float higher) {
  static std::default_random_engine re;
  static std::uniform_real_distribution<float> dist(lower, higher);
  return dist(re);
}

void SHMatrix::CommitUnaryOps() {
  CommitTranspose();
  CommitScale();
}

void SHMatrix::CommitTranspose() {
  if (transpose_called && !transpose_done) {
    if (scale_called && !scale_done) {
      transpose_worker(scalar);
      scale_done = true;
      scalar = 1.0f;
    }
    else
      transpose_worker();
    transpose_done = true;
  }
}

void SHMatrix::CommitScale() {
  if (scale_called && !scale_done) {
    scale_worker();
    scale_done = true;
  }
}

// Transpose operation : speeds up computation by postponing T operations
SHMatrix& SHMatrix::T() { //mini_idx & maxi_idx computation pending
  if (!transpose_done)
    transpose_called ^= true; //toggling boolean
  else
    transpose_called = true;
  transpose_done = false;
  int tmp = cols;
  std::reverse(data_dims.begin(), data_dims.end());
  load_dims(data_dims);
  return *this;
}

SHMatrix& SHMatrix::Scale(float scale_arg) {
  scale_called = true;
  scalar *= scale_arg;
  scale_done = false;
  return *this;
}

void SHMatrix::Dot(cublasHandle_t cublas_handle, SHMatrix &A,
                   SHMatrix &B, SHMatrix &C) {
  if (C.data_loc == GPU) {
    gpu2any_dotproduct(cublas_handle, A, B, C);
  }
  else if (C.data_loc == CPU) {
    cpu2any_dotproduct(A, B, C);
  }
}

float* SHMatrix::DataPointerAtLoc(SHMatrix &arg,
                                  mem_location desired_loc) {
  float *ret_ptr;
  if (arg.data_loc == desired_loc)
    ret_ptr = arg.data;
  else {
    if (desired_loc == GPU) {
      CudaSafeCall(cudaMalloc((void **)&ret_ptr,
                              sizeof(float) * arg.num_elems));
      CudaSafeCall(cudaMemcpy(ret_ptr, arg.data,
                              sizeof(float) * arg.num_elems,
                              cudaMemcpyHostToDevice));
    }
    else if (desired_loc == CPU) {
      ret_ptr = (float *)malloc(sizeof(float) * arg.num_elems);
      CudaSafeCall(cudaMemcpy(ret_ptr, arg.data,
                              sizeof(float) * arg.num_elems,
                              cudaMemcpyDeviceToHost));
    }
  }
  return ret_ptr;
}

void SHMatrix::operator*=(SHMatrix &arg) {
  if (data_dims.size() > 2) {
    CommitUnaryOps();
    arg.CommitUnaryOps();
  }
  scalar *= arg.scalar;
  if (data_loc == GPU) {
    gpu2any_elemwise_mult(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_mult(arg);
  }
}

void SHMatrix::operator+=(SHMatrix &arg) {
  CommitScale();
  arg.CommitScale();
  if (data_loc == GPU) {
    gpu2any_elemwise_add(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_add(arg);
  }
}

void SHMatrix::operator-=(SHMatrix &arg) {
  CommitScale();
  arg.CommitScale();
  if (data_loc == GPU) {
    gpu2any_elemwise_subtract(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_subtract(arg);
  }
}

void SHMatrix::operator/=(SHMatrix &arg) {
  if (data_dims.size() > 2) {
    CommitUnaryOps();
    arg.CommitUnaryOps();
  }
  scalar /= arg.scalar;
  if (data_loc == GPU) {
    gpu2any_elemwise_divide(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_divide(arg);
  }
}

void SHMatrix::operator*=(float arg) {
  Scale(arg);
}

void SHMatrix::operator+=(float arg) {
  CommitScale();
  if (data_loc == GPU) {
    gpu2any_elemwise_add(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_add(arg);
  }
}

void SHMatrix::operator-=(float arg) {
  CommitScale();
  if (data_loc == GPU) {
    gpu2any_elemwise_subtract(arg);
  }
  else if (data_loc == CPU) {
    cpu2any_elemwise_subtract(arg);
  }
}

void SHMatrix::operator/=(float arg) {
  Scale(1.0f / arg);
}

void SHMatrix::gaussian_init_gpu(float mean, float stddev) {
  curandGenerator_t rng;
  CurandSafeCall(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW));
  CurandSafeCall(curandGenerateNormal(rng, data, sizeof(float) * num_elems,
                                      mean, stddev));
  CurandSafeCall(curandDestroyGenerator(rng));
}

void SHMatrix::gaussian_init_cpu(float mean, float stddev) {
  for (int i = 0; i < num_elems; i++) {
    data[i] = GetGaussianNum(mean, stddev);
  }
}

void SHMatrix::uniform_init_gpu(float lower, float higher) {
  curandGenerator_t rng;
  CurandSafeCall(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_XORWOW));
  CurandSafeCall(curandGenerateUniform(rng, data, sizeof(float) * num_elems));
  CurandSafeCall(curandDestroyGenerator(rng));
  ScaleUniformSHMatrix(data, num_elems, lower, higher);
}

void SHMatrix::uniform_init_cpu(float lower, float higher) {
  for (int i = 0; i < num_elems; i++) {
    data[i] = GetUniformNum(lower, higher);
  }
}

SHMatrix::~SHMatrix() { Clear(); }

void SHMatrix::load_dims(std::vector<int> &dims) {
  cols = 1;
  rows = dims[0];
  for (int i = 1; i < dims.size(); i++) {
    cols *= dims[i];
  }
  num_elems = rows * cols;
}

void SHMatrix::print_h_var(float *h_v, int r, int c, bool print_elem) {
  std::cout << "-------------------------" << std::endl;
  if (!allocated) {
    std::cout << "<NULL_SHMATRIX>" << std::endl;
  }
  mini = h_v[0];
  maxi = h_v[0];
  float sum = 0.0f;
  mini_idx = 0;
  maxi_idx = 0;
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      if (print_elem)
        std::cout << h_v[j + i * c] << "\t";
      if (h_v[j + i * c] < mini) {
        mini = h_v[j + i * c];
        mini_idx = j + i * c;
      }
      if (h_v[j + i * c] > maxi) {
        maxi = h_v[j + i * c];
        maxi_idx = j + i * c;
      }
      sum += h_v[j + i * c];
    }
    if (print_elem)
      std::cout << std::endl;
  }
  mean = sum / (r * c);
  if (allocated) {
    std::cout << "Shape = (";
    for (int i = 0; i < data_dims.size() - 1; i++) {
      std::cout << data_dims[i] << ", ";
    }
    std::cout << data_dims[data_dims.size() - 1] << ")" << std::endl;
    std::cout << "Minimum at index " << mini_idx << " = " << mini << std::endl;
    std::cout << "Maximum at index " << maxi_idx << " = " << maxi << std::endl;
    std::cout << "Average of all elements = " << mean << std::endl;
  }
  std::cout << "Number of elements = " << num_elems << std::endl;
  if (data_loc == GPU) {
    std::cout << "Location = GPU" << std::endl;
  }
  else if (data_loc == CPU) {
    std::cout << "Location = CPU" << std::endl;
  }
}

void SHMatrix::gpu2any_elemwise_mult(SHMatrix &arg) {
  gpu2any_elemwise_op_worker(arg, MULT);
}

void SHMatrix::gpu2any_elemwise_divide(SHMatrix &arg) {
  gpu2any_elemwise_op_worker(arg, DIV);
}

void SHMatrix::gpu2any_elemwise_add(SHMatrix &arg) {
  gpu2any_elemwise_op_worker(arg, ADD);
}

void SHMatrix::gpu2any_elemwise_subtract(SHMatrix &arg) {
  gpu2any_elemwise_op_worker(arg, SUB);
}

void SHMatrix::gpu2any_dotproduct(cublasHandle_t cublas_handle, SHMatrix &A,
                                  SHMatrix &B, SHMatrix &C) {
  int new_rows = A.rows, new_cols = B.cols;
  int new_num_elems = new_rows * new_cols;
  std::vector<int> new_dims(2);
  new_dims[0] = new_rows;
  new_dims[1] = new_cols;
  C.Reallocate(new_dims, GPU);
  float *d_A = A.data, *d_B = B.data;
  if (A.data_loc == CPU) {
    d_A = DataPointerAtLoc(A, GPU);
  }
  if (B.data_loc == CPU) {
    d_B = DataPointerAtLoc(B, GPU);
  }
  float coeff = A.scalar * B.scalar, beta = 0;

  if (transpose_decider(A.transpose_called, A.transpose_done)) {
    if (transpose_decider(B.transpose_called, B.transpose_done)) {
      CublasSafeCall(cublasSgemm_v2(cublas_handle, CUBLAS_OP_N,
                                    CUBLAS_OP_N, A.rows, B.cols, B.rows,
                                    &coeff, d_A, A.rows, d_B,
                                    B.rows, &beta, C.data, C.rows));
    }
    else {
      CublasSafeCall(cublasSgemm_v2(cublas_handle, CUBLAS_OP_N,
                                    CUBLAS_OP_T, A.rows, B.cols, B.rows,
                                    &coeff, d_A, A.rows, d_B,
                                    B.cols, &beta, C.data, C.rows));
    }
  }
  else {
    if (transpose_decider(B.transpose_called, B.transpose_done)) {
      CublasSafeCall(cublasSgemm_v2(cublas_handle, CUBLAS_OP_T,
                                    CUBLAS_OP_N, A.rows, B.cols, B.rows,
                                    &coeff, d_A, A.cols, d_B,
                                    B.rows, &beta, C.data, C.rows));
    }
    else {
      CublasSafeCall(cublasSgemm_v2(cublas_handle, CUBLAS_OP_T,
                                    CUBLAS_OP_T, A.rows, B.cols, B.rows,
                                    &coeff, d_A, A.cols, d_B, 
                                    B.cols, &beta, C.data, C.rows));
    }
  }
  C.transpose_called = true;
  C.transpose_done = false;
  if (A.data_loc == CPU) {
    CudaSafeCall(cudaFree(d_A));
  }
  if (B.data_loc == CPU) {
    CudaSafeCall(cudaFree(d_B));
  }
}

void SHMatrix::gpu2any_elemwise_op_worker(SHMatrix &arg, ELEM_OP elem_op) {
  float *d_arg_data = DataPointerAtLoc(arg, GPU);
  int ld_data_real = transpose_decider(transpose_called, transpose_done) ? rows
    : cols;
  int ld_arg_data_real = transpose_decider(arg.transpose_called,
                                           arg.transpose_done)
    ? arg.rows : arg.cols;
  if (elem_op == MULT)
    ElemwiseMultiplyInPlaceGPU(data, d_arg_data, ld_data_real,
                               ld_arg_data_real, num_elems,
                               transpose_decider(transpose_called,
                                                 transpose_done),
                               transpose_decider(arg.transpose_called,
                                                 arg.transpose_done));
  else if (elem_op == DIV)
    ElemwiseDivideInPlaceGPU(data, d_arg_data, ld_data_real,
                             ld_arg_data_real, num_elems,
                             transpose_decider(transpose_called,
                                               transpose_done),
                             transpose_decider(arg.transpose_called,
                                               arg.transpose_done));
  else if (elem_op == ADD)
    ElemwiseAddInPlaceGPU(data, d_arg_data, ld_data_real,
                          ld_arg_data_real, num_elems,
                          transpose_decider(transpose_called,
                                            transpose_done),
                          transpose_decider(arg.transpose_called,
                                            arg.transpose_done));
  else if (elem_op == SUB)
    ElemwiseSubtractInPlaceGPU(data, d_arg_data, ld_data_real,
                               ld_arg_data_real, num_elems,
                               transpose_decider(transpose_called,
                                                 transpose_done),
                               transpose_decider(arg.transpose_called,
                                                 arg.transpose_done));

  if (arg.data_loc == CPU) {
    CudaSafeCall(cudaFree(d_arg_data));
  }
}

void SHMatrix::gpu2any_elemwise_add(float arg) {
  ElemwiseAddInPlaceGPU_Scalar(data, arg, num_elems);
}

void SHMatrix::gpu2any_elemwise_subtract(float arg) {
  ElemwiseSubtractInPlaceGPU_Scalar(data, arg, num_elems);
}

void SHMatrix::cpu2any_elemwise_mult(SHMatrix &arg) {
  cpu2any_elemwise_op_worker(arg, MULT);
}

void SHMatrix::cpu2any_elemwise_divide(SHMatrix &arg) {
  cpu2any_elemwise_op_worker(arg, DIV);
}

void SHMatrix::cpu2any_elemwise_add(SHMatrix &arg) {
  cpu2any_elemwise_op_worker(arg, ADD);
}

void SHMatrix::cpu2any_elemwise_subtract(SHMatrix &arg) {
  cpu2any_elemwise_op_worker(arg, SUB);
}

void SHMatrix::cpu2any_dotproduct(SHMatrix &A, SHMatrix &B, SHMatrix &C) {
  int new_rows = A.rows, new_cols = B.cols;
  int new_num_elems = new_rows * new_cols;
  std::vector<int> new_dims(2);
  new_dims[0] = new_rows;
  new_dims[1] = new_cols;
  C.Reallocate(new_dims, CPU);
  float *h_A = A.data, *h_B = B.data;
  if (A.data_loc == GPU) {
    h_A = DataPointerAtLoc(A, CPU);
  }
  if (B.data_loc == GPU) {
    h_B = DataPointerAtLoc(B, CPU);
  }
  float val;
  int vert_idx[2], hor_idx[2], k = B.rows;
  int a_lin_idx, b_lin_idx;
  for (int i = 0; i < new_rows; i++) {
    for (int j = 0; j < new_cols; j++) {
      vert_idx[0] = 0;
      vert_idx[1] = j;
      hor_idx[0] = i;
      hor_idx[1] = 0;
      val = 0;
      for (int cnt = 0; cnt < k; cnt++) {
        if (transpose_decider(A.transpose_called, A.transpose_done))
          a_lin_idx = hor_idx[0] + hor_idx[1] * A.rows;
        else
          a_lin_idx = hor_idx[1] + hor_idx[0] * A.cols;
        if (transpose_decider(B.transpose_called, B.transpose_done))
          b_lin_idx = vert_idx[0] + vert_idx[1] * B.rows;
        else
          b_lin_idx = vert_idx[1] + vert_idx[0] * B.cols;
        val += h_A[a_lin_idx] * h_B[b_lin_idx];
        vert_idx[0]++;
        hor_idx[1]++;
      }
      C.data[j + i * new_cols] = val;
    }
  }
  if (A.data_loc == GPU) {
    free(h_A);
  }
  if (B.data_loc == GPU) {
    free(h_B);
  }
}

void SHMatrix::cpu2any_elemwise_op_worker(SHMatrix &arg,
                                          ELEM_OP elem_op) {
  float *h_arg_data = DataPointerAtLoc(arg, CPU);
  int ld_data_real = transpose_decider(transpose_called, transpose_done) ? rows
    : cols;
  int ld_arg_data_real = transpose_decider(arg.transpose_called,
                                           arg.transpose_done)
    ? arg.rows : arg.cols;
  if (elem_op == MULT)
    ElemwiseMultiplyInPlaceCPU(data, h_arg_data, ld_data_real,
                               ld_arg_data_real, num_elems,
                               transpose_decider(transpose_called,
                                                 transpose_done),
                               transpose_decider(arg.transpose_called,
                                                 arg.transpose_done));
  else if (elem_op == DIV)
    ElemwiseDivideInPlaceCPU(data, h_arg_data, ld_data_real,
                             ld_arg_data_real, num_elems,
                             transpose_decider(transpose_called,
                                               transpose_done),
                             transpose_decider(arg.transpose_called,
                                               arg.transpose_done));
  else if (elem_op == ADD)
    ElemwiseAddInPlaceCPU(data, h_arg_data, ld_data_real,
                          ld_arg_data_real, num_elems,
                          transpose_decider(transpose_called,
                                            transpose_done),
                          transpose_decider(arg.transpose_called,
                                            arg.transpose_done));
  else if (elem_op == SUB)
    ElemwiseSubtractInPlaceCPU(data, h_arg_data, ld_data_real,
                               ld_arg_data_real, num_elems,
                               transpose_decider(transpose_called,
                                                 transpose_done),
                               transpose_decider(arg.transpose_called,
                                                 arg.transpose_done));
  if (arg.data_loc == GPU) {
    free(h_arg_data);
  }
}

void SHMatrix::cpu2any_elemwise_add(float arg) {
  for (int i = 0; i < num_elems; i++) {
    data[i] += arg;
  }
}

void SHMatrix::cpu2any_elemwise_subtract(float arg) {
  for (int i = 0; i < num_elems; i++) {
    data[i] -= arg;
  }
}

void SHMatrix::duplicate_shmatrix(SHMatrix &src_shmatrix, 
                                  bool mem_alloc_needed) {
  for (int i = 0; i < src_shmatrix.data_dims.size(); i++) {
    data_dims.push_back(src_shmatrix.data_dims[i]);
  }
  load_dims(data_dims);
  if (mem_alloc_needed)
    data = allocate_memory();
  copy_data_from(data, src_shmatrix.data, data_loc,
                 src_shmatrix.data_loc, num_elems);
  scalar = src_shmatrix.scalar;
  transpose_called = src_shmatrix.transpose_called;
  transpose_done = src_shmatrix.transpose_done;
  scale_called = src_shmatrix.scale_called;
  scale_done = src_shmatrix.scale_done;
  allocated = true;
}

void SHMatrix::copy_data_from(float *dst_ptr, float *src_ptr,
                              mem_location dst_loc, mem_location src_loc,
                              int copy_length) {
  if (src_loc == GPU) {
    if (dst_loc == GPU) {
      CudaSafeCall(cudaMemcpy(dst_ptr, src_ptr,
                              sizeof(float) * copy_length,
                              cudaMemcpyDeviceToDevice));
    }
    else if (dst_loc == CPU) {
      CudaSafeCall(cudaMemcpy(dst_ptr, src_ptr,
                              sizeof(float) * copy_length,
                              cudaMemcpyDeviceToHost));
    }
  }
  else if (src_loc == CPU) {
    if (dst_loc == GPU) {
      CudaSafeCall(cudaMemcpy(dst_ptr, src_ptr,
                              sizeof(float) * copy_length,
                              cudaMemcpyHostToDevice));
    }
    else if (dst_loc == CPU) {
      memcpy(dst_ptr, src_ptr, sizeof(float) * copy_length);
    }
  }
}

void SHMatrix::deallocate_memory(float *mem_ptr, mem_location mem_loc) {
  if (mem_loc == GPU) {
    if (allocated)
      CudaSafeCall(cudaFree(mem_ptr));
  }
  else if (mem_loc == CPU) {
    if (allocated)
      free(mem_ptr);
  }
  allocated = false;
}

void SHMatrix::transpose_worker_gpu(float coeff) {
  float *d_data_T;
  CudaSafeCall(cudaMalloc((void **)&d_data_T,
                          sizeof(float) * num_elems));
  CublasSafeCall(cublasSgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             cols, rows, &coeff, data, rows, &beta,
                             d_data_T, cols, d_data_T, cols));
  CudaSafeCall(cudaFree(data));
  data = d_data_T;
}

//Operation currently offloaded to CPU; GPU Kernel for this pending!
void SHMatrix::transpose_worker_ndim_gpu(float coeff) {
  float *h_data_tmp = (float *)malloc(sizeof(float) * num_elems);
  CudaSafeCall(cudaMemcpy(h_data_tmp, data,
                          sizeof(float) * num_elems,
                          cudaMemcpyDeviceToHost));
  float *tmp = data;
  data = h_data_tmp;
  transpose_worker_cpu(coeff);
  h_data_tmp = data;
  data = tmp;
  CudaSafeCall(cudaMemcpy(data, h_data_tmp,
                          sizeof(float) * num_elems,
                          cudaMemcpyHostToDevice));
  free(h_data_tmp);
}

void SHMatrix::transpose_worker_cpu(float coeff) {
  float *h_data_T = (float *)malloc(sizeof(float) * num_elems);
  float tmp;
  int read_lin_idx = 0, write_lin_idx = 0;
  std::vector<int> read_idx_vect(data_dims.size(), 0);
  std::vector<int> write_idx_vect(data_dims.size(), 0);
  std::vector<int> read_data_dims(data_dims.size());
  std::vector<int> write_data_dims(data_dims.size());
  std::copy(data_dims.begin(), data_dims.end(),
            read_data_dims.begin());
  std::copy(data_dims.begin(), data_dims.end(),
            write_data_dims.begin());
  std::reverse(read_data_dims.begin(), read_data_dims.end());
  int dim = data_dims.size() - 1;

  while (true) {
    read_lin_idx = vect_to_lin_idx(read_idx_vect, read_data_dims);
    std::copy(read_idx_vect.begin(), read_idx_vect.end(),
              write_idx_vect.begin());
    std::reverse(write_idx_vect.begin(), write_idx_vect.end());
    write_lin_idx = vect_to_lin_idx(write_idx_vect, write_data_dims);
    h_data_T[write_lin_idx] = coeff * data[read_lin_idx];
    if (read_lin_idx >= num_elems - 1)
      break;
    next_vect_idx(read_idx_vect, read_data_dims);
  }
  free(data);
  data = h_data_T;
  read_idx_vect.clear();
  write_idx_vect.clear();
  read_data_dims.clear();
  write_data_dims.clear();
}

void SHMatrix::scale_worker_gpu(float coeff) {
  CublasSafeCall(cublasSscal_v2(cublas_handle, num_elems,
                                &coeff, data, 1));
}

void SHMatrix::scale_worker_cpu(float coeff) {
  for (int i = 0; i < num_elems; i++) {
    data[i] *= coeff;
  }
}

void SHMatrix::transpose_worker(float coeff) {
  if (data_loc == GPU) {
    if (data_dims.size() < 3)
      transpose_worker_gpu(coeff);
    else { 
      transpose_worker_ndim_gpu(coeff);
    }
  }
  else if (data_loc == CPU) {
    transpose_worker_cpu(coeff);
  }
}

void SHMatrix::scale_worker() {
  if (data_loc == GPU) {
    scale_worker_gpu(scalar);
  }
  else if (data_loc == CPU) {
    scale_worker_cpu(scalar);
  }
  scalar = 1.0f;
}

void SHMatrix::next_vect_idx(std::vector<int> &vect_idx,
                             std::vector<int> &vect_dims) {
  int alter_dim = vect_idx.size() - 1;
  while (vect_idx[alter_dim] >= vect_dims[alter_dim] - 1) {
    alter_dim--;
    if (alter_dim < 0)
      return;
  }
  if (alter_dim < vect_idx.size() - 1
      && vect_idx[alter_dim + 1] >= vect_dims[alter_dim + 1] - 1) {
    for (int i = alter_dim + 1; i < vect_idx.size(); i++) {
      vect_idx[i] = 0;
    }
  }
  vect_idx[alter_dim]++;
}

int SHMatrix::vect_to_lin_idx(std::vector<int> &vect_idx,
                              std::vector<int> &vect_dims) {
  int lin_idx = vect_idx[vect_idx.size() - 1];
  int m = vect_dims[vect_dims.size() - 1];
  for (int dim = vect_idx.size() - 2; dim >= 0; dim--) {
    lin_idx += vect_idx[dim] * m;
    if (dim > 0)
      m *= vect_dims[dim];
  }
  return lin_idx;
}

std::vector<int> SHMatrix::lin_to_vect_idx(int lin_idx,
                                           std::vector<int> &vect_dims) {
  std::vector<int> vect_idx(vect_dims.size(), 0);
  int curr_dim_sz = 1, curr_lin_idx = lin_idx;
  for (int i = 1; i < vect_dims.size(); i++) {
    curr_dim_sz *= vect_dims[i];
  }
  for (int dim = 0; dim < vect_dims.size(); dim++) {
    vect_idx[dim] = curr_lin_idx / curr_dim_sz;
    curr_lin_idx -= vect_idx[dim] * curr_dim_sz;
    if (dim + 1 < vect_dims.size())
      curr_dim_sz /= vect_dims[dim + 1];
  }
  return vect_idx;
}

void SHMatrix::reset_metadata() {
  if (allocated) {
    init_list_properties();
    init_value_properties();
  }
}

float* SHMatrix::allocate_memory() {
  float *data_ptr;
  if (data_loc == GPU) {
    CudaSafeCall(cudaMalloc((void **)&data_ptr, sizeof(float) * num_elems));
  }
  else if (data_loc == CPU) {
    data_ptr = (float *)malloc(sizeof(float) * num_elems);
  }
  allocated = true;
  return data_ptr;
}

//Truth table -
// 11 - 0
// 10 - 1
// 00 - 0
// 01 - 0
bool SHMatrix::transpose_decider(bool t_called, bool t_done) {
  return (t_called ^ t_done) & t_called; //same as t_called && !t_done
}

void SHMatrix::init() {
  init_value_properties();
}

void SHMatrix::init_value_properties() {
  rows = 0;
  cols = 0;
  num_elems = 0;
  mini_idx = 0;
  maxi_idx = 0;
  transpose_called = false;
  transpose_done = false;
  scale_called = false;
  scale_done = false;
  scalar = 1.0f;
  alpha = 1.0f;
  beta = 0.0f;
  allocated = false;
  mean = 0.0f;
  mini = 0.0f;
  maxi = 0.0f;
}

void SHMatrix::init_list_properties() {
  data_dims.clear();
  name.clear();
}

void SHMatrix::init_with_default_value(float *mem_ptr, mem_location mem_loc,
                                       float init_val) {
  if (mem_loc == GPU) {
    FloatCUDAMemset(mem_ptr, num_elems, init_val);
    CudaCheckError();
  }
  else if (mem_loc == CPU) {
    for (int i = 0; i < num_elems; i++) {
      mem_ptr[i] = init_val;
    }
  }
}
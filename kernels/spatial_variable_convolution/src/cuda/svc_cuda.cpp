#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
torch::Tensor spatial_variable_convolution_cuda_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor dilation_map,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, 
    int dilation_h, int dilation_w, int groups, int channels_per_group, 
    int height_out, int width_out, bool use_dilation_map);

std::vector<torch::Tensor> spatial_variable_convolution_cuda_backward(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, torch::Tensor dilation_map,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, 
    int dilation_h, int dilation_w, int groups, int channels_per_group, 
    int height_out, int width_out, bool use_dilation_map);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor spatial_variable_convolution_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor dilation_map,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, 
    int dilation_h, int dilation_w, int groups, int channels_per_group, 
    int height_out, int width_out, bool use_dilation_map) {
  
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  
  if (use_dilation_map) {
    CHECK_INPUT(dilation_map);
  }
  
  return spatial_variable_convolution_cuda_forward(
      input, weight, dilation_map, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, 
      dilation_h, dilation_w, groups, channels_per_group, height_out, width_out, use_dilation_map);
}

std::vector<torch::Tensor> spatial_variable_convolution_backward(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, torch::Tensor dilation_map,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, 
    int dilation_h, int dilation_w, int groups, int channels_per_group, 
    int height_out, int width_out, bool use_dilation_map) {
  
  CHECK_INPUT(grad_output);
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  
  if (use_dilation_map) {
    CHECK_INPUT(dilation_map);
  }
  
  return spatial_variable_convolution_cuda_backward(
      grad_output, input, weight, dilation_map, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
      dilation_h, dilation_w, groups, channels_per_group, height_out, width_out, use_dilation_map);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &spatial_variable_convolution_forward, 
        "Spatial Variable Convolution forward (CUDA)");
  m.def("backward", &spatial_variable_convolution_backward, 
        "Spatial Variable Convolution backward (CUDA)");
}
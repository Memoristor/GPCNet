#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N) {
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

// Input gradients accumulate in FP32 for FP16/BF16, or FP64 for FP64 inputs.
// Overload for float
__device__ __forceinline__ void atomic_add(float* address, float val) { atomicAdd(address, val); }

// Overload for double
__device__ __forceinline__ void atomic_add(double* address, double val) { atomicAdd(address, val); }

// Coordinates, interpolation and reductions use FP32 for FP16/BF16 inputs.
// FP64 inputs retain FP64 arithmetic. Tensor storage keeps the input dtype.
// Bilinear interpolation function
template <typename scalar_t, typename acc_t>
__device__ acc_t bilinear_interpolate(const scalar_t* input, const int height, const int width,
                                     acc_t h, acc_t w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return static_cast<acc_t>(0);
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  acc_t lh = h - h_low;
  acc_t lw = w - w_low;
  acc_t hh = static_cast<acc_t>(1) - lh;
  acc_t hw = static_cast<acc_t>(1) - lw;

  acc_t v1 = static_cast<acc_t>(0);
  if (h_low >= 0 && w_low >= 0) v1 = input[h_low * width + w_low];

  acc_t v2 = static_cast<acc_t>(0);
  if (h_low >= 0 && w_high < width) v2 = input[h_low * width + w_high];

  acc_t v3 = static_cast<acc_t>(0);
  if (h_high < height && w_low >= 0) v3 = input[h_high * width + w_low];

  acc_t v4 = static_cast<acc_t>(0);
  if (h_high < height && w_high < width) v4 = input[h_high * width + w_high];

  return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
}

// Forward kernel implementation
template <typename scalar_t, bool UseDilationMap>
__global__ void spatial_variable_convolution_forward_kernel(
    const int n, const scalar_t* input, const scalar_t* weight, const scalar_t* dilation_map,
    const int batch_size, const int channels, const int height, const int width,
    const int height_out, const int width_out, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w, 
    const int dilation_h, const int dilation_w, const int groups, const int channels_per_group, 
    scalar_t* output) {
  using acc_t = at::acc_type<scalar_t, true>;
  
  const int kernel_h_center = (kernel_h - 1) >> 1;
  const int kernel_w_center = (kernel_w - 1) >> 1;
  const int kernel_h_ref = kernel_h_center * dilation_h;
  const int kernel_w_ref = kernel_w_center * dilation_w;

  CUDA_KERNEL_LOOP(index, n) {
    // Decode index to get output position
    const int w_out = index % width_out;
    const int h_out = (index / width_out) % height_out;
    const int c_out = (index / (width_out * height_out)) % channels;
    const int b = index / (channels * height_out * width_out);
    
    // Calculate group index
    const int group_idx = c_out / channels_per_group;
    
    // Initialize output value
    acc_t output_value = static_cast<acc_t>(0);
    
    // Input channel is the same as output channel
    const int c_in = c_out;
    const scalar_t* input_ptr = input + (b * channels + c_in) * height * width;
    
    // For each kernel position
    for (int kh = 0; kh < kernel_h; kh++) {
      for (int kw = 0; kw < kernel_w; kw++) {
        acc_t input_value;
        
        // Base position in input (without kernel offsets)
        const int h_in_base = h_out * stride_h - pad_h;
        const int w_in_base = w_out * stride_w - pad_w;
        
        if (UseDilationMap) {
          // With dilation map - use bilinear interpolation
          const int dilation_map_idx = b * groups * height_out * width_out +
                                     group_idx * height_out * width_out + 
                                     h_out * width_out + w_out;
          const acc_t dilation_value = dilation_map[dilation_map_idx];
          
          // Calculate center-relative offset for correct alignment
          const acc_t h_offset = (kh - kernel_h_center) * dilation_h * dilation_value;
          const acc_t w_offset = (kw - kernel_w_center) * dilation_w * dilation_value;
          
          // Final sampling position (center-aligned)
          const acc_t h_in = h_in_base + h_offset + kernel_h_ref;
          const acc_t w_in = w_in_base + w_offset + kernel_w_ref;
          
          input_value = bilinear_interpolate(input_ptr, height, width, h_in, w_in);
        } else {
          // Without dilation map - use direct indexing with center alignment
          const int h_offset = kh - kernel_h_center;
          const int w_offset = kw - kernel_w_center;
          
          const int h_in = h_in_base + h_offset * dilation_h + kernel_h_ref;
          const int w_in = w_in_base + w_offset * dilation_w + kernel_w_ref;
          
          // Check boundaries
          if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
            input_value = input_ptr[h_in * width + w_in];
          } else {
            input_value = static_cast<acc_t>(0); // Zero padding
          }
        }
        
        // Get weight value
        const int k_idx = kh * kernel_w + kw;
        const int weight_idx = b * (groups * kernel_h * kernel_w * height_out * width_out) +
                             group_idx * (kernel_h * kernel_w * height_out * width_out) +
                             k_idx * (height_out * width_out) +
                             h_out * width_out + w_out;
        
        const acc_t weight_value = weight[weight_idx];
        
        // Accumulate output
        output_value += input_value * weight_value;
      }
    }
    
    // Store output
    output[index] = static_cast<scalar_t>(output_value);
  }
}

// Backward kernel for input gradients
template <typename scalar_t, bool UseDilationMap>
__global__ void spatial_variable_convolution_backward_input_kernel(
    const int n, const scalar_t* grad_output, const scalar_t* weight, const scalar_t* dilation_map,
    const int batch_size, const int channels, const int height, const int width,
    const int height_out, const int width_out, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w, 
    const int dilation_h, const int dilation_w, const int groups, const int channels_per_group, 
    at::acc_type<scalar_t, true>* grad_input) {
  using acc_t = at::acc_type<scalar_t, true>;
  
  const int kernel_h_center = (kernel_h - 1) >> 1;
  const int kernel_w_center = (kernel_w - 1) >> 1;
  const int kernel_h_ref = kernel_h_center * dilation_h;
  const int kernel_w_ref = kernel_w_center * dilation_w;

  CUDA_KERNEL_LOOP(index, n) {
    const int w_out = index % width_out;
    const int h_out = (index / width_out) % height_out;
    const int c_out = (index / (width_out * height_out)) % channels;
    const int b = index / (channels * height_out * width_out);
    
    // Get group index
    const int group_idx = c_out / channels_per_group;
    
    // Get gradient value for this output
    const acc_t grad_val = grad_output[index];
    
    // If gradient is zero, skip computation
    if (grad_val == static_cast<acc_t>(0)) {
      continue;
    }
    
    // Input channel matches output channel
    const int c_in = c_out;
    
    // Base position in input (without kernel offsets)
    const int h_in_base = h_out * stride_h - pad_h;
    const int w_in_base = w_out * stride_w - pad_w;
    
    // For each kernel position
    for (int kh = 0; kh < kernel_h; kh++) {
      for (int kw = 0; kw < kernel_w; kw++) {
        // Get weight value
        const int k_idx = kh * kernel_w + kw;
        const int weight_idx = b * (groups * kernel_h * kernel_w * height_out * width_out) +
                             group_idx * (kernel_h * kernel_w * height_out * width_out) +
                             k_idx * (height_out * width_out) +
                             h_out * width_out + w_out;
        const acc_t weight_val = weight[weight_idx];
        
        // If weight is zero, skip computation
        if (weight_val == static_cast<acc_t>(0)) {
          continue;
        }
        
        if (UseDilationMap) {
          // Get dilation value
          const int dilation_map_idx = b * groups * height_out * width_out +
                                     group_idx * height_out * width_out + 
                                     h_out * width_out + w_out;
          const acc_t dilation_value = dilation_map[dilation_map_idx];
          
          // Calculate center-relative offset
          const acc_t h_offset = (kh - kernel_h_center) * dilation_h * dilation_value;
          const acc_t w_offset = (kw - kernel_w_center) * dilation_w * dilation_value;
          
          // Final sampling position (center-aligned)
          const acc_t h_in = h_in_base + h_offset + kernel_h_ref;
          const acc_t w_in = w_in_base + w_offset + kernel_w_ref;
          
          // Skip if outside input boundaries
          if (h_in <= -1 || h_in >= height || w_in <= -1 || w_in >= width) {
            continue;
          }
          
          // Calculate bilinear interpolation coordinates
          int h_low = floor(h_in);
          int w_low = floor(w_in);
          int h_high = h_low + 1;
          int w_high = w_low + 1;
          
          acc_t lh = h_in - h_low;
          acc_t lw = w_in - w_low;
          acc_t hh = static_cast<acc_t>(1) - lh;
          acc_t hw = static_cast<acc_t>(1) - lw;
          
          // Calculate gradient contribution to each corner
          const acc_t grad_product = grad_val * weight_val;
          const acc_t grad_h0w0 = grad_product * hh * hw;
          const acc_t grad_h0w1 = grad_product * hh * lw;
          const acc_t grad_h1w0 = grad_product * lh * hw;
          const acc_t grad_h1w1 = grad_product * lh * lw;
          
          // Get base pointer for input gradient
          acc_t* grad_input_ptr = grad_input + (b * channels + c_in) * height * width;
          
          // Accumulate gradients at each corner point
          if (h_low >= 0 && h_low < height && w_low >= 0 && w_low < width) {
            atomic_add(&grad_input_ptr[h_low * width + w_low], grad_h0w0);
          }
          
          if (h_low >= 0 && h_low < height && w_high >= 0 && w_high < width) {
            atomic_add(&grad_input_ptr[h_low * width + w_high], grad_h0w1);
          }
          
          if (h_high >= 0 && h_high < height && w_low >= 0 && w_low < width) {
            atomic_add(&grad_input_ptr[h_high * width + w_low], grad_h1w0);
          }
          
          if (h_high >= 0 && h_high < height && w_high >= 0 && w_high < width) {
            atomic_add(&grad_input_ptr[h_high * width + w_high], grad_h1w1);
          }
        } else {
          // Without dilation map - direct integer indexing with center alignment
          const int h_offset = kh - kernel_h_center;
          const int w_offset = kw - kernel_w_center;
          
          const int h_in = h_in_base + h_offset * dilation_h + kernel_h_ref;
          const int w_in = w_in_base + w_offset * dilation_w + kernel_w_ref;
          
          // Skip if outside input boundaries
          if (h_in < 0 || h_in >= height || w_in < 0 || w_in >= width) {
            continue;
          }
          
          // Calculate input gradient index and update
          const int input_idx = (b * channels + c_in) * height * width + h_in * width + w_in;
          atomic_add(&grad_input[input_idx], grad_val * weight_val);
        }
      }
    }
  }
}

// Weight gradient kernel
template <typename scalar_t, bool UseDilationMap>
__global__ void spatial_variable_convolution_backward_weight_kernel(
    const int n, const scalar_t* grad_output, const scalar_t* input, const scalar_t* dilation_map,
    const int batch_size, const int channels, const int height, const int width,
    const int height_out, const int width_out, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w, 
    const int dilation_h, const int dilation_w, const int groups, const int channels_per_group, 
    scalar_t* grad_weight) {
  using acc_t = at::acc_type<scalar_t, true>;
      
  const int kernel_h_center = (kernel_h - 1) >> 1;
  const int kernel_w_center = (kernel_w - 1) >> 1;
  const int kernel_h_ref = kernel_h_center * dilation_h;
  const int kernel_w_ref = kernel_w_center * dilation_w;
  
  CUDA_KERNEL_LOOP(index, n) {
    // Decode weight gradient index
    const int w_out = index % width_out;
    const int h_out = (index / width_out) % height_out;
    const int k_idx = (index / (width_out * height_out)) % (kernel_h * kernel_w);
    const int g = (index / (width_out * height_out * kernel_h * kernel_w)) % groups;
    const int b = index / (width_out * height_out * kernel_h * kernel_w * groups);
    
    const int kh = k_idx / kernel_w;
    const int kw = k_idx % kernel_w;
    
    // Base position in input (without kernel offsets)
    const int h_in_base = h_out * stride_h - pad_h;
    const int w_in_base = w_out * stride_w - pad_w;
    
    // Calculate channel range for this group
    const int c_start = g * channels_per_group;
    const int c_end = (g + 1) * channels_per_group;
    
    acc_t grad_weight_val = static_cast<acc_t>(0);
    
    if (UseDilationMap) {
      // Get dilation value for this location
      const int dilation_map_idx = b * groups * height_out * width_out +
                                 g * height_out * width_out + 
                                 h_out * width_out + w_out;
      const acc_t dilation_value = dilation_map[dilation_map_idx];
      
      // Calculate center-relative offset
      const acc_t h_offset = (kh - kernel_h_center) * dilation_h * dilation_value;
      const acc_t w_offset = (kw - kernel_w_center) * dilation_w * dilation_value;
      
      // Final input coordinates (center-aligned)
      const acc_t h_in = h_in_base + h_offset + kernel_h_ref;
      const acc_t w_in = w_in_base + w_offset + kernel_w_ref;
      
      // Skip if completely outside input boundaries
      if (h_in <= -1 || h_in >= height || w_in <= -1 || w_in >= width) {
        grad_weight_val = static_cast<acc_t>(0);
      } else {
        // Accumulate gradient from all channels in this group
        for (int c = c_start; c < c_end; c++) {
          // Get input value using bilinear interpolation
          const scalar_t* input_ptr = input + (b * channels + c) * height * width;
          const acc_t input_val = bilinear_interpolate(input_ptr, height, width, h_in, w_in);
          
          // Get output gradient
          const int grad_out_idx = b * (channels * height_out * width_out) +
                                   c * (height_out * width_out) +
                                   h_out * width_out + w_out;
          const acc_t grad_out_val = grad_output[grad_out_idx];
          
          // Accumulate contribution
          grad_weight_val += input_val * grad_out_val;
        }
      }
    } else {
      // Without dilation map - direct integer indexing with center alignment
      const int h_offset = kh - kernel_h_center;
      const int w_offset = kw - kernel_w_center;
      
      const int h_in = h_in_base + h_offset * dilation_h + kernel_h_ref;
      const int w_in = w_in_base + w_offset * dilation_w + kernel_w_ref;
      
      // Skip if outside valid region
      if (h_in < 0 || h_in >= height || w_in < 0 || w_in >= width) {
        grad_weight_val = static_cast<acc_t>(0);
      } else {
        // Accumulate from all channels in this group
        for (int c = c_start; c < c_end; c++) {
          // Get input value directly
          const acc_t input_val = input[(b * channels + c) * height * width + h_in * width + w_in];
          
          // Get output gradient
          const int grad_out_idx = b * (channels * height_out * width_out) +
                                   c * (height_out * width_out) +
                                   h_out * width_out + w_out;
          const acc_t grad_out_val = grad_output[grad_out_idx];
          
          // Accumulate contribution
          grad_weight_val += input_val * grad_out_val;
        }
      }
    }
    
    // Store weight gradient
    grad_weight[index] = static_cast<scalar_t>(grad_weight_val);
  }
}

// Dilation map gradient kernel
template <typename scalar_t>
__global__ void spatial_variable_convolution_backward_dilation_kernel(
    const int n, const scalar_t* grad_output, const scalar_t* input, const scalar_t* weight,
    const scalar_t* dilation_map, const int batch_size, const int channels, const int height,
    const int width, const int height_out, const int width_out, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, 
    const int dilation_h, const int dilation_w, const int groups, const int channels_per_group, 
    scalar_t* grad_dilation_map) {
  using acc_t = at::acc_type<scalar_t, true>;
      
  const int kernel_h_center = (kernel_h - 1) >> 1;
  const int kernel_w_center = (kernel_w - 1) >> 1;
  const int kernel_h_ref = kernel_h_center * dilation_h;
  const int kernel_w_ref = kernel_w_center * dilation_w;
  
  CUDA_KERNEL_LOOP(index, n) {
    // Decode dilation map index
    const int w_out = index % width_out;
    const int h_out = (index / width_out) % height_out;
    const int g = (index / (width_out * height_out)) % groups;
    const int b = index / (width_out * height_out * groups);
    
    // Base position in input (without kernel offsets)
    const int h_in_base = h_out * stride_h - pad_h;
    const int w_in_base = w_out * stride_w - pad_w;
    
    // Get channel range for this group
    const int c_start = g * channels_per_group;
    const int c_end = (g + 1) * channels_per_group;
    
    // Get current dilation value
    const acc_t dilation_value = dilation_map[index];
    
    // Initialize gradient
    acc_t grad_dilation = static_cast<acc_t>(0);
    
    // Accumulate gradient from all channels in the group
    for (int c = c_start; c < c_end; c++) {
      // Get output gradient for this position
      const int grad_out_idx = b * (channels * height_out * width_out) +
                               c * (height_out * width_out) + 
                               h_out * width_out + w_out;
      const acc_t grad_out_val = grad_output[grad_out_idx];
      
      // Skip if gradient is zero
      if (grad_out_val == static_cast<acc_t>(0)) {
        continue;
      }
      
      // Get input pointer for this batch and channel
      const scalar_t* input_ptr = input + (b * channels + c) * height * width;
      
      // For each kernel position
      for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
          // Get weight value
          const int k_idx = kh * kernel_w + kw;
          const int weight_idx = b * (groups * kernel_h * kernel_w * height_out * width_out) +
                                g * (kernel_h * kernel_w * height_out * width_out) +
                                k_idx * (height_out * width_out) +
                                h_out * width_out + w_out;
          const acc_t weight_val = weight[weight_idx];
          
          // Skip if weight is zero
          if (weight_val == static_cast<acc_t>(0)) {
            continue;
          }
          
          // Calculate offset relative to center
          const acc_t h_offset_factor = (kh - kernel_h_center) * dilation_h;
          const acc_t w_offset_factor = (kw - kernel_w_center) * dilation_w;
          
          // Calculate input position with current dilation
          const acc_t h_in = h_in_base + h_offset_factor * dilation_value + kernel_h_ref;
          const acc_t w_in = w_in_base + w_offset_factor * dilation_value + kernel_w_ref;
          
          // Skip if completely outside input
          if (h_in <= -1 || h_in >= height || w_in <= -1 || w_in >= width) {
            continue;
          }
          
          // Calculate bilinear interpolation coordinates
          int h_low = floor(h_in);
          int w_low = floor(w_in);
          int h_high = h_low + 1;
          int w_high = w_low + 1;
          
          acc_t lh = h_in - h_low;
          acc_t lw = w_in - w_low;
          
          // Get input values with boundary checking
          acc_t v1 = static_cast<acc_t>(0);
          acc_t v2 = static_cast<acc_t>(0);
          acc_t v3 = static_cast<acc_t>(0);
          acc_t v4 = static_cast<acc_t>(0);
          
          bool valid_v1 = false;
          bool valid_v2 = false;
          bool valid_v3 = false;
          bool valid_v4 = false;
          
          if (h_low >= 0 && w_low >= 0) {
            v1 = input_ptr[h_low * width + w_low];
            valid_v1 = true;
          }
          
          if (h_low >= 0 && w_high < width) {
            v2 = input_ptr[h_low * width + w_high];
            valid_v2 = true;
          }
          
          if (h_high < height && w_low >= 0) {
            v3 = input_ptr[h_high * width + w_low];
            valid_v3 = true;
          }
          
          if (h_high < height && w_high < width) {
            v4 = input_ptr[h_high * width + w_high];
            valid_v4 = true;
          }
          
          // Calculate gradients with respect to h and w
          acc_t grad_h = static_cast<acc_t>(0);
          acc_t grad_w = static_cast<acc_t>(0);
          
          if (valid_v1 || valid_v3) {
            grad_h += (v3 - v1) * (static_cast<acc_t>(1) - lw);
          }
          if (valid_v2 || valid_v4) {
            grad_h += (v4 - v2) * lw;
          }
          if (valid_v1 || valid_v2) {
            grad_w += (v2 - v1) * (static_cast<acc_t>(1) - lh);
          }
          if (valid_v3 || valid_v4) {
            grad_w += (v4 - v3) * lh;
          }
          
          // Scale by offset factors to get gradient w.r.t. dilation
          const acc_t grad_dilation_contribution =
              (grad_h * h_offset_factor + grad_w * w_offset_factor) * 
              weight_val * grad_out_val;
          
          // Accumulate gradient
          grad_dilation += grad_dilation_contribution;
        }
      }
    }
    
    // Store gradient
    grad_dilation_map[index] = static_cast<scalar_t>(grad_dilation);
  }
}

// Forward function
torch::Tensor spatial_variable_convolution_cuda_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor dilation_map,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w, int groups, int channels_per_group,
    int height_out, int width_out, bool use_dilation_map) {
  
  const int batch_size = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);
  
  // Create output tensor
  auto output = torch::zeros({batch_size, channels, height_out, width_out}, input.options());
  const int num_kernels = batch_size * channels * height_out * width_out;
  
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "spatial_variable_convolution_forward_cuda", ([&] {
        if (use_dilation_map) {
          spatial_variable_convolution_forward_kernel<scalar_t, true>
              <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  num_kernels, input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), 
                  dilation_map.data_ptr<scalar_t>(), batch_size, channels, height, width,
                  height_out, width_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                  dilation_h, dilation_w, groups, channels_per_group, output.data_ptr<scalar_t>());
        } else {
          spatial_variable_convolution_forward_kernel<scalar_t, false>
              <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  num_kernels, input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), 
                  nullptr, batch_size, channels, height, width,
                  height_out, width_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                  dilation_h, dilation_w, groups, channels_per_group, output.data_ptr<scalar_t>());
        }
      }));
  
  return output;
}

// Backward function
std::vector<torch::Tensor> spatial_variable_convolution_cuda_backward(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, torch::Tensor dilation_map,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w, int groups, int channels_per_group,
    int height_out, int width_out, bool use_dilation_map) {
  
  const int batch_size = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);
  
  // Atomic updates must not round each contribution to FP16/BF16.
  auto grad_input = torch::zeros_like(
      input, input.options().dtype(at::toAccumulateType(input.scalar_type(), true)));
  auto grad_weight = torch::zeros_like(weight);
  auto grad_dilation_map = use_dilation_map ? torch::zeros_like(dilation_map) : torch::zeros({1, 1, 1}, dilation_map.options());
  
  // Number of elements for each kernel
  const int num_kernels_output = batch_size * channels * height_out * width_out;
  const int num_kernels_weight = batch_size * groups * kernel_h * kernel_w * height_out * width_out;
  const int num_kernels_dilation = batch_size * groups * height_out * width_out;
  
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "spatial_variable_convolution_backward_cuda", ([&] {
        using acc_t = at::acc_type<scalar_t, true>;
        if (use_dilation_map) {
          // With dilation map
          spatial_variable_convolution_backward_input_kernel<scalar_t, true>
              <<<GET_BLOCKS(num_kernels_output), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  num_kernels_output, grad_output.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                  dilation_map.data_ptr<scalar_t>(), batch_size, channels, height, width,
                  height_out, width_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                  dilation_h, dilation_w, groups, channels_per_group, grad_input.data_ptr<acc_t>());
          
          spatial_variable_convolution_backward_weight_kernel<scalar_t, true>
              <<<GET_BLOCKS(num_kernels_weight), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  num_kernels_weight, grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  dilation_map.data_ptr<scalar_t>(), batch_size, channels, height, width,
                  height_out, width_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                  dilation_h, dilation_w, groups, channels_per_group, grad_weight.data_ptr<scalar_t>());
          
          spatial_variable_convolution_backward_dilation_kernel<scalar_t>
              <<<GET_BLOCKS(num_kernels_dilation), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  num_kernels_dilation, grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  weight.data_ptr<scalar_t>(), dilation_map.data_ptr<scalar_t>(),
                  batch_size, channels, height, width, height_out, width_out, kernel_h, kernel_w,
                  stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, channels_per_group,
                  grad_dilation_map.data_ptr<scalar_t>());
        } else {
          // Without dilation map
          spatial_variable_convolution_backward_input_kernel<scalar_t, false>
              <<<GET_BLOCKS(num_kernels_output), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  num_kernels_output, grad_output.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                  nullptr, batch_size, channels, height, width,
                  height_out, width_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                  dilation_h, dilation_w, groups, channels_per_group, grad_input.data_ptr<acc_t>());
          
          spatial_variable_convolution_backward_weight_kernel<scalar_t, false>
              <<<GET_BLOCKS(num_kernels_weight), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                  num_kernels_weight, grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                  nullptr, batch_size, channels, height, width,
                  height_out, width_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                  dilation_h, dilation_w, groups, channels_per_group, grad_weight.data_ptr<scalar_t>());
        }
      }));
  
  return {grad_input.to(input.scalar_type()), grad_weight, grad_dilation_map};
}

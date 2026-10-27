# Spatial Variable Convolution (SVC)

A CUDA-accelerated implementation of spatial variable convolution that efficiently handles both regular convolutions and dynamic dilation convolutions without creating memory-intensive intermediate tensors.

## Features

- Memory-efficient replacement for `F.unfold` + `einsum` convolutions
- Support for dynamic dilation maps with bilinear interpolation
- Group convolution functionality
- No large intermediate tensors created during computation
- Fast forward and backward passes
- Support for half, float, and double precision

## Installation

```bash
# Install dependencies
pip install torch einops

# Install the package
pip install -e .
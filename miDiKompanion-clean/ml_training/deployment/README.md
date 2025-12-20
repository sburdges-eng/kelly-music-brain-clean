# ML Models Deployment Package

## Overview

This package contains trained ONNX models for miDiKompanion plugin.

**Deployment Date**: 2025-12-18T04:34:56.304174
**Total Size**: 0.00 MB

## Models


## Installation

1. Copy models to plugin Resources directory:
   ```
   cp models/*.onnx /path/to/plugin/Resources/models/
   ```

2. Enable ONNX Runtime in CMake:
   ```cmake
   cmake -DENABLE_ONNX_RUNTIME=ON ..
   ```

3. Rebuild plugin

## Usage

Models are automatically loaded by `ONNXInference` class when available.
See `src/ml/ONNXInference.h` for API documentation.

## Validation

All models have been validated:
- ONNX model structure check passed
- Input/output shapes verified
- Test inference successful

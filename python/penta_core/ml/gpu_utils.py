"""
GPU Utilities - Device detection and management.

Provides unified GPU/accelerator detection across:
- NVIDIA CUDA
- Apple Metal (MPS)
- AMD ROCm
- Intel oneAPI
- OpenCL
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import platform
import importlib


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal
    ROCM = "rocm"  # AMD
    ONEAPI = "oneapi"  # Intel
    OPENCL = "opencl"


@dataclass
class GPUDevice:
    """Information about a compute device."""
    device_type: DeviceType
    name: str
    index: int = 0
    memory_total_mb: int = 0
    memory_free_mb: int = 0
    compute_capability: Optional[str] = None
    cuda_driver_version: Optional[str] = None
    cuda_runtime_version: Optional[str] = None

    # Performance info
    fp32_tflops: Optional[float] = None
    fp16_tflops: Optional[float] = None
    bf16_supported: Optional[bool] = None
    int8_supported: Optional[bool] = None
    tensor_cores_available: Optional[bool] = None

    # Backend-specific
    backend_device_id: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.device_type.value}:{self.index} - {self.name} ({self.memory_total_mb}MB)"


def get_available_devices() -> List[GPUDevice]:
    """
    Detect all available compute devices.

    Returns:
        List of available devices
    """
    devices = []

    # Always add CPU
    devices.append(GPUDevice(
        device_type=DeviceType.CPU,
        name=platform.processor() or "CPU",
        index=0,
    ))

    # Check for CUDA
    cuda_devices = _detect_cuda_devices()
    devices.extend(cuda_devices)

    # Check for MPS (Apple Silicon)
    mps_devices = _detect_mps_devices()
    devices.extend(mps_devices)

    # Check for ROCm (AMD)
    rocm_devices = _detect_rocm_devices()
    devices.extend(rocm_devices)

    return devices


def _detect_cuda_devices() -> List[GPUDevice]:
    """Detect NVIDIA CUDA devices."""
    devices = []

    if importlib.util.find_spec("torch"):
        import torch
        if torch.cuda.is_available():
            cuda_info = get_cuda_runtime_info()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory // (1024 * 1024)

                # Get free memory
                torch.cuda.set_device(i)
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                    memory_free = free_bytes // (1024 * 1024)
                    memory_total = total_bytes // (1024 * 1024)
                except AttributeError:
                    memory_free = memory_total - (torch.cuda.memory_reserved(i) // (1024 * 1024))

                capability = f"{props.major}.{props.minor}"
                tensor_cores = _has_tensor_cores(props.major, props.minor)
                bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
                bf16_available = bf16_supported() if bf16_supported else None
                int8_supported = _has_int8_support(props.major, props.minor)

                devices.append(GPUDevice(
                    device_type=DeviceType.CUDA,
                    name=props.name,
                    index=i,
                    memory_total_mb=memory_total,
                    memory_free_mb=memory_total - memory_free,
                    compute_capability=capability,
                    cuda_driver_version=cuda_info.get("driver"),
                    cuda_runtime_version=cuda_info.get("runtime"),
                    bf16_supported=bf16_available,
                    int8_supported=int8_supported,
                    tensor_cores_available=tensor_cores,
                    backend_device_id=f"cuda:{i}",
                ))

    if not devices and importlib.util.find_spec("pycuda"):
        import pycuda.driver as cuda
        import pycuda.autoinit

        cuda_info = get_cuda_runtime_info()
        for i in range(cuda.Device.count()):
            dev = cuda.Device(i)
            major, minor = dev.compute_capability()
            capability = f"{major}.{minor}"
            devices.append(GPUDevice(
                device_type=DeviceType.CUDA,
                name=dev.name(),
                index=i,
                memory_total_mb=dev.total_memory() // (1024 * 1024),
                compute_capability=capability,
                cuda_driver_version=cuda_info.get("driver"),
                cuda_runtime_version=cuda_info.get("runtime"),
                bf16_supported=_has_bf16_support(major, minor),
                int8_supported=_has_int8_support(major, minor),
                tensor_cores_available=_has_tensor_cores(major, minor),
                backend_device_id=f"cuda:{i}",
            ))

    return devices


def _detect_mps_devices() -> List[GPUDevice]:
    """Detect Apple Metal Performance Shaders devices."""
    devices = []

    if platform.system() != "Darwin":
        return devices

    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Get GPU name from system
            import subprocess
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
            )
            gpu_name = "Apple GPU"
            for line in result.stdout.split('\n'):
                if "Chipset Model:" in line:
                    gpu_name = line.split(":")[-1].strip()
                    break

            devices.append(GPUDevice(
                device_type=DeviceType.MPS,
                name=gpu_name,
                index=0,
                backend_device_id="mps",
            ))

    except ImportError:
        pass

    return devices


def _detect_rocm_devices() -> List[GPUDevice]:
    """Detect AMD ROCm devices."""
    devices = []

    try:
        import torch
        if hasattr(torch, "hip") and torch.hip.is_available():
            for i in range(torch.hip.device_count()):
                props = torch.hip.get_device_properties(i)
                devices.append(GPUDevice(
                    device_type=DeviceType.ROCM,
                    name=props.name,
                    index=i,
                    memory_total_mb=props.total_memory // (1024 * 1024),
                    backend_device_id=f"hip:{i}",
                ))
    except (ImportError, AttributeError):
        pass

    return devices


def get_cuda_runtime_info() -> Dict[str, Optional[str]]:
    """
    Get CUDA runtime and driver versions (if available).

    Returns:
        Dict with keys: driver, runtime
    """
    info: Dict[str, Optional[str]] = {"driver": None, "runtime": None}
    if importlib.util.find_spec("torch"):
        import torch
        info["runtime"] = torch.version.cuda
        if hasattr(torch, "_C") and hasattr(torch._C, "_cuda_getDriverVersion"):
            driver_version = torch._C._cuda_getDriverVersion()
            if driver_version:
                major = driver_version // 1000
                minor = (driver_version % 1000) // 10
                info["driver"] = f"{major}.{minor}"
    return info


def get_cuda_capability_map() -> Dict[str, Dict[str, Any]]:
    """
    Map CUDA device capabilities by index.

    Returns:
        Dict of device_index -> capability info
    """
    capability_map: Dict[str, Dict[str, Any]] = {}
    devices = _detect_cuda_devices()
    for device in devices:
        capability_map[str(device.index)] = {
            "name": device.name,
            "compute_capability": device.compute_capability,
            "memory_total_mb": device.memory_total_mb,
            "memory_free_mb": device.memory_free_mb,
            "tensor_cores_available": device.tensor_cores_available,
            "bf16_supported": device.bf16_supported,
            "int8_supported": device.int8_supported,
            "cuda_driver_version": device.cuda_driver_version,
            "cuda_runtime_version": device.cuda_runtime_version,
        }
    return capability_map


def _has_tensor_cores(major: int, minor: int) -> bool:
    """Determine Tensor Core availability from compute capability."""
    return (major, minor) >= (7, 0)


def _has_bf16_support(major: int, minor: int) -> bool:
    """Determine BF16 support from compute capability."""
    return (major, minor) >= (8, 0)


def _has_int8_support(major: int, minor: int) -> bool:
    """Determine INT8 support from compute capability."""
    return (major, minor) >= (6, 1)


def select_best_device(
    prefer_gpu: bool = True,
    min_memory_mb: int = 0,
) -> GPUDevice:
    """
    Select the best available compute device.

    Args:
        prefer_gpu: Prefer GPU over CPU if available
        min_memory_mb: Minimum required memory

    Returns:
        Best available device
    """
    devices = get_available_devices()

    if not prefer_gpu:
        # Return CPU
        return next(d for d in devices if d.device_type == DeviceType.CPU)

    # Filter by memory
    gpu_devices = [
        d for d in devices
        if d.device_type != DeviceType.CPU
        and d.memory_total_mb >= min_memory_mb
    ]

    if not gpu_devices:
        # Fall back to CPU
        return next(d for d in devices if d.device_type == DeviceType.CPU)

    # Prefer in order: CUDA > MPS > ROCm
    priority = [DeviceType.CUDA, DeviceType.MPS, DeviceType.ROCM]

    for device_type in priority:
        matches = [d for d in gpu_devices if d.device_type == device_type]
        if matches:
            # Return device with most memory
            return max(matches, key=lambda d: d.memory_total_mb)

    return gpu_devices[0]


def get_device_for_backend(backend: str) -> Optional[GPUDevice]:
    """
    Get device info for a specific backend.

    Args:
        backend: Backend name (pytorch, onnx, tensorflow, coreml)

    Returns:
        Recommended device for backend, or None
    """
    devices = get_available_devices()

    if backend in ["pytorch", "torch"]:
        # PyTorch: CUDA > MPS > CPU
        for device_type in [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]:
            for device in devices:
                if device.device_type == device_type:
                    return device

    elif backend in ["onnx", "onnxruntime"]:
        # ONNX Runtime: CUDA > DirectML > CPU
        for device_type in [DeviceType.CUDA, DeviceType.CPU]:
            for device in devices:
                if device.device_type == device_type:
                    return device

    elif backend in ["tensorflow", "tf", "tflite"]:
        # TensorFlow: CUDA > CPU
        for device_type in [DeviceType.CUDA, DeviceType.CPU]:
            for device in devices:
                if device.device_type == device_type:
                    return device

    elif backend in ["coreml", "mlmodel"]:
        # CoreML: MPS only on Apple, otherwise None
        if platform.system() == "Darwin":
            for device in devices:
                if device.device_type == DeviceType.MPS:
                    return device

    return None


def get_torch_device() -> str:
    """
    Get PyTorch device string.

    Returns:
        Device string like "cuda:0", "mps", or "cpu"
    """
    device = select_best_device()

    if device.device_type == DeviceType.CUDA:
        return f"cuda:{device.index}"
    elif device.device_type == DeviceType.MPS:
        return "mps"
    elif device.device_type == DeviceType.ROCM:
        return f"cuda:{device.index}"  # ROCm uses CUDA API
    else:
        return "cpu"


def get_onnx_providers() -> List[str]:
    """
    Get ONNX Runtime execution providers.

    Returns:
        List of provider names in priority order
    """
    providers = []
    devices = get_available_devices()

    # Check for GPU devices
    for device in devices:
        if device.device_type == DeviceType.CUDA:
            providers.append("CUDAExecutionProvider")
        elif device.device_type == DeviceType.MPS:
            providers.append("CoreMLExecutionProvider")
        elif device.device_type == DeviceType.ROCM:
            providers.append("ROCMExecutionProvider")

    # Windows DirectML
    if platform.system() == "Windows":
        try:
            import onnxruntime
            if "DmlExecutionProvider" in onnxruntime.get_available_providers():
                providers.append("DmlExecutionProvider")
        except ImportError:
            pass

    # Always include CPU fallback
    providers.append("CPUExecutionProvider")

    return providers


def estimate_inference_memory(
    model_size_mb: float,
    batch_size: int = 1,
    input_size_mb: float = 0.1,
) -> float:
    """
    Estimate memory required for inference.

    Args:
        model_size_mb: Model file size in MB
        batch_size: Inference batch size
        input_size_mb: Input tensor size per sample

    Returns:
        Estimated memory in MB
    """
    # Model weights
    memory = model_size_mb

    # Activations (rough estimate: 2x model size)
    memory += model_size_mb * 2

    # Input/output buffers
    memory += input_size_mb * batch_size * 2

    # Workspace (25% overhead)
    memory *= 1.25

    return memory


def check_memory_available(required_mb: float) -> bool:
    """
    Check if enough GPU memory is available.

    Args:
        required_mb: Required memory in MB

    Returns:
        True if enough memory available
    """
    device = select_best_device()

    if device.device_type == DeviceType.CPU:
        return True  # CPU always has "enough" memory

    if device.memory_free_mb >= required_mb:
        return True

    return False

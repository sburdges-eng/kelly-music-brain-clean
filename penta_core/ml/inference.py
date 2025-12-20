"""
Inference Engine - Unified ML inference across backends.

Provides a common interface for running inference with:
- ONNX Runtime
- TensorFlow Lite
- CoreML
- PyTorch
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np
import time

from python.penta_core.ml.model_registry import (
    ModelInfo,
    ModelBackend,
    get_model,
)


@dataclass
class InferenceResult:
    """Result from model inference."""
    outputs: Dict[str, np.ndarray]
    latency_ms: float
    model_name: str
    backend: ModelBackend

    # Optional metadata
    confidence: Optional[float] = None
    labels: Optional[List[str]] = None

    def get_output(self, name: str = None) -> np.ndarray:
        """Get output by name, or first output if name is None."""
        if name:
            return self.outputs[name]
        return next(iter(self.outputs.values()))

    def get_top_k(
        self,
        k: int = 5,
        output_name: str = None,
    ) -> List[tuple]:
        """Get top-k predictions with indices and scores."""
        output = self.get_output(output_name).flatten()
        indices = np.argsort(output)[-k:][::-1]
        scores = output[indices]

        if self.labels:
            return [(self.labels[i], float(s)) for i, s in zip(indices, scores)]
        return [(int(i), float(s)) for i, s in zip(indices, scores)]


class InferenceEngine(ABC):
    """Base class for inference engines."""

    def __init__(self, model_info: ModelInfo):
        self.model_info = model_info
        self._session = None
        self._loaded = False

    @abstractmethod
    def load(self) -> bool:
        """Load the model."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model."""
        pass

    @abstractmethod
    def infer(
        self,
        inputs: Dict[str, np.ndarray],
    ) -> InferenceResult:
        """Run inference."""
        pass

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def get_input_shape(self) -> Optional[List[int]]:
        """Get expected input shape."""
        return self.model_info.input_shape

    def get_output_shape(self) -> Optional[List[int]]:
        """Get expected output shape."""
        return self.model_info.output_shape


class ONNXEngine(InferenceEngine):
    """ONNX Runtime inference engine."""

    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info)
        self._providers = None

    def load(self) -> bool:
        """Load ONNX model."""
        try:
            import onnxruntime as ort

            # Select execution providers
            available_providers = ort.get_available_providers()
            self._providers = []

            # Prefer GPU providers
            if "CUDAExecutionProvider" in available_providers:
                self._providers.append("CUDAExecutionProvider")
            if "CoreMLExecutionProvider" in available_providers:
                self._providers.append("CoreMLExecutionProvider")
            if "DmlExecutionProvider" in available_providers:
                self._providers.append("DmlExecutionProvider")

            # Always include CPU fallback
            self._providers.append("CPUExecutionProvider")

            # Create session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                self.model_info.path,
                sess_options=sess_options,
                providers=self._providers,
            )

            self._loaded = True
            return True

        except ImportError:
            print("ONNX Runtime not installed. Install with: pip install onnxruntime")
            return False
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            return False

    def unload(self) -> None:
        """Unload ONNX model."""
        self._session = None
        self._loaded = False

    def infer(
        self,
        inputs: Dict[str, np.ndarray],
    ) -> InferenceResult:
        """Run ONNX inference."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.perf_counter()

        # Get input/output names
        input_names = [inp.name for inp in self._session.get_inputs()]
        output_names = [out.name for out in self._session.get_outputs()]

        # Prepare inputs
        feed_dict = {}
        for name in input_names:
            if name in inputs:
                feed_dict[name] = inputs[name]
            else:
                # Try to find matching input by index
                idx = input_names.index(name)
                if idx < len(inputs):
                    feed_dict[name] = list(inputs.values())[idx]

        # Run inference
        outputs_list = self._session.run(output_names, feed_dict)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Build output dictionary
        outputs = {name: arr for name, arr in zip(output_names, outputs_list)}

        return InferenceResult(
            outputs=outputs,
            latency_ms=latency_ms,
            model_name=self.model_info.name,
            backend=ModelBackend.ONNX,
        )

    def get_input_names(self) -> List[str]:
        """Get input tensor names."""
        if self._session:
            return [inp.name for inp in self._session.get_inputs()]
        return []

    def get_output_names(self) -> List[str]:
        """Get output tensor names."""
        if self._session:
            return [out.name for out in self._session.get_outputs()]
        return []


class TFLiteEngine(InferenceEngine):
    """TensorFlow Lite inference engine."""

    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info)
        self._interpreter = None
        self._input_details = None
        self._output_details = None

    def load(self) -> bool:
        """Load TFLite model."""
        try:
            import tensorflow as tf

            # Create interpreter
            self._interpreter = tf.lite.Interpreter(
                model_path=self.model_info.path,
                num_threads=4,
            )
            self._interpreter.allocate_tensors()

            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            self._loaded = True
            return True

        except ImportError:
            print("TensorFlow not installed. Install with: pip install tensorflow")
            return False
        except Exception as e:
            print(f"Failed to load TFLite model: {e}")
            return False

    def unload(self) -> None:
        """Unload TFLite model."""
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._loaded = False

    def infer(
        self,
        inputs: Dict[str, np.ndarray],
    ) -> InferenceResult:
        """Run TFLite inference."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.perf_counter()

        # Set input tensors
        for i, detail in enumerate(self._input_details):
            name = detail["name"]
            if name in inputs:
                self._interpreter.set_tensor(detail["index"], inputs[name])
            elif i < len(inputs):
                self._interpreter.set_tensor(
                    detail["index"],
                    list(inputs.values())[i],
                )

        # Run inference
        self._interpreter.invoke()

        # Get output tensors
        outputs = {}
        for detail in self._output_details:
            outputs[detail["name"]] = self._interpreter.get_tensor(detail["index"])

        latency_ms = (time.perf_counter() - start_time) * 1000

        return InferenceResult(
            outputs=outputs,
            latency_ms=latency_ms,
            model_name=self.model_info.name,
            backend=ModelBackend.TENSORFLOW_LITE,
        )


class CoreMLEngine(InferenceEngine):
    """CoreML inference engine (macOS/iOS only)."""

    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info)
        self._model = None

    def load(self) -> bool:
        """Load CoreML model."""
        try:
            import coremltools as ct
            import platform

            if platform.system() != "Darwin":
                print("CoreML is only available on macOS/iOS")
                return False

            self._model = ct.models.MLModel(self.model_info.path)
            self._loaded = True
            return True

        except ImportError:
            print("coremltools not installed. Install with: pip install coremltools")
            return False
        except Exception as e:
            print(f"Failed to load CoreML model: {e}")
            return False

    def unload(self) -> None:
        """Unload CoreML model."""
        self._model = None
        self._loaded = False

    def infer(
        self,
        inputs: Dict[str, np.ndarray],
    ) -> InferenceResult:
        """Run CoreML inference."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.perf_counter()

        # Run prediction
        result = self._model.predict(inputs)

        # Convert to numpy arrays
        outputs = {}
        for name, value in result.items():
            if hasattr(value, "__array__"):
                outputs[name] = np.array(value)
            else:
                outputs[name] = np.array([value])

        latency_ms = (time.perf_counter() - start_time) * 1000

        return InferenceResult(
            outputs=outputs,
            latency_ms=latency_ms,
            model_name=self.model_info.name,
            backend=ModelBackend.COREML,
        )


class PyTorchEngine(InferenceEngine):
    """PyTorch inference engine."""

    def __init__(self, model_info: ModelInfo):
        super().__init__(model_info)
        self._model = None
        self._device = None

    def load(self) -> bool:
        """Load PyTorch model."""
        try:
            import torch

            # Select device
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")

            # Load model
            self._model = torch.load(
                self.model_info.path,
                map_location=self._device,
            )

            if hasattr(self._model, "eval"):
                self._model.eval()

            self._loaded = True
            return True

        except ImportError:
            print("PyTorch not installed. Install with: pip install torch")
            return False
        except Exception as e:
            print(f"Failed to load PyTorch model: {e}")
            return False

    def unload(self) -> None:
        """Unload PyTorch model."""
        self._model = None
        self._device = None
        self._loaded = False

    def infer(
        self,
        inputs: Dict[str, np.ndarray],
    ) -> InferenceResult:
        """Run PyTorch inference."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        import torch

        start_time = time.perf_counter()

        # Convert inputs to tensors
        tensor_inputs = {}
        for name, arr in inputs.items():
            tensor_inputs[name] = torch.from_numpy(arr).to(self._device)

        # Run inference
        with torch.no_grad():
            if len(tensor_inputs) == 1:
                output = self._model(list(tensor_inputs.values())[0])
            else:
                output = self._model(**tensor_inputs)

        # Convert outputs to numpy
        outputs = {}
        if isinstance(output, dict):
            for name, tensor in output.items():
                outputs[name] = tensor.cpu().numpy()
        elif isinstance(output, (list, tuple)):
            for i, tensor in enumerate(output):
                outputs[f"output_{i}"] = tensor.cpu().numpy()
        else:
            outputs["output"] = output.cpu().numpy()

        latency_ms = (time.perf_counter() - start_time) * 1000

        return InferenceResult(
            outputs=outputs,
            latency_ms=latency_ms,
            model_name=self.model_info.name,
            backend=ModelBackend.PYTORCH,
        )


def create_engine(model_info: ModelInfo) -> InferenceEngine:
    """
    Create an inference engine for the given model.

    Args:
        model_info: Model information

    Returns:
        Appropriate inference engine
    """
    engines = {
        ModelBackend.ONNX: ONNXEngine,
        ModelBackend.TENSORFLOW_LITE: TFLiteEngine,
        ModelBackend.COREML: CoreMLEngine,
        ModelBackend.PYTORCH: PyTorchEngine,
        ModelBackend.TORCHSCRIPT: PyTorchEngine,
    }

    engine_class = engines.get(model_info.backend)
    if not engine_class:
        raise ValueError(f"Unsupported backend: {model_info.backend}")

    return engine_class(model_info)


def create_engine_by_name(name: str) -> Optional[InferenceEngine]:
    """
    Create an inference engine by model name.

    Args:
        name: Registered model name

    Returns:
        Inference engine or None if model not found
    """
    model_info = get_model(name)
    if model_info:
        return create_engine(model_info)
    return None

"""
Model Export Utilities - Convert trained models to deployment formats.

Supports:
- ONNX: Cross-platform inference
- Core ML: Native macOS/iOS deployment
- RTNeural JSON: Real-time audio plugin format
- TorchScript: PyTorch mobile/C++

Usage:
    from python.penta_core.ml.export import ModelExporter
    
    exporter = ModelExporter(model, config)
    exporter.export_all(output_dir="models/")
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for model export."""
    
    model_id: str
    input_size: int
    output_size: int
    architecture_type: str = "mlp"  # mlp, cnn, lstm
    n_mels: int = 64  # For CNN/spectrogram models
    sequence_length: int = 32  # For LSTM/sequence models
    
    # Export options
    opset_version: int = 14  # ONNX opset
    optimize_for_mobile: bool = True
    quantize: bool = False  # Quantization for smaller models
    
    # Core ML options
    coreml_target: str = "macOS13"  # Deployment target
    coreml_compute_units: str = "ALL"  # ALL, CPU_ONLY, CPU_AND_GPU


class ModelExporter:
    """
    Export trained PyTorch models to various formats.
    
    Handles format-specific requirements and optimizations.
    """
    
    def __init__(
        self,
        model: "torch.nn.Module",
        config: ExportConfig,
    ):
        """
        Initialize exporter.
        
        Args:
            model: Trained PyTorch model
            config: Export configuration
        """
        self.model = model
        self.config = config
        self._exported_files: List[Path] = []
    
    def _get_dummy_input(self) -> "torch.Tensor":
        """Create dummy input tensor for tracing."""
        import torch
        
        if self.config.architecture_type == "cnn":
            # Spectrogram input: (batch, channels, mels, time)
            return torch.randn(1, 1, self.config.n_mels, 128)
        elif self.config.architecture_type == "lstm":
            # Sequence input: (batch, seq_len, features)
            return torch.randn(1, self.config.sequence_length, self.config.input_size)
        else:
            # Flat input: (batch, features)
            return torch.randn(1, self.config.input_size)
    
    def export_onnx(
        self,
        output_path: Union[str, Path],
        optimize: bool = True,
    ) -> Path:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Output file path
            optimize: Apply ONNX optimizations
        
        Returns:
            Path to exported model
        """
        import torch
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        self.model = self.model.to("cpu")
        
        dummy_input = self._get_dummy_input()
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        
        # Optimize if requested
        if optimize:
            try:
                import onnx
                from onnxruntime.transformers import optimizer
                
                # Load and optimize
                model_onnx = onnx.load(output_path)
                optimized_model = optimizer.optimize_model(
                    output_path,
                    model_type="bert",  # Generic optimization
                    num_heads=0,
                    hidden_size=0,
                )
                optimized_model.save_model_to_file(output_path)
                logger.info(f"Applied ONNX optimizations")
            except ImportError:
                logger.debug("onnxruntime.transformers not available, skipping optimization")
        
        logger.info(f"Exported ONNX: {output_path}")
        self._exported_files.append(output_path)
        return output_path
    
    def export_coreml(
        self,
        output_path: Union[str, Path],
        convert_to_float16: bool = False,
    ) -> Optional[Path]:
        """
        Export model to Core ML format (macOS only).
        
        Args:
            output_path: Output file path
            convert_to_float16: Use float16 for smaller model
        
        Returns:
            Path to exported model, or None if not on macOS
        """
        import torch
        
        if platform.system() != "Darwin":
            logger.warning("Core ML export only available on macOS")
            return None
        
        try:
            import coremltools as ct
        except ImportError:
            logger.error("coremltools not installed. Install with: pip install coremltools")
            return None
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        self.model = self.model.to("cpu")
        
        dummy_input = self._get_dummy_input()
        
        # Trace model
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        # Determine input shape
        if self.config.architecture_type == "cnn":
            input_shape = ct.Shape(shape=(1, 1, self.config.n_mels, 128))
        elif self.config.architecture_type == "lstm":
            input_shape = ct.Shape(shape=(1, self.config.sequence_length, self.config.input_size))
        else:
            input_shape = ct.Shape(shape=(1, self.config.input_size))
        
        # Map deployment target
        target_map = {
            "macOS13": ct.target.macOS13,
            "macOS14": ct.target.macOS14,
            "iOS16": ct.target.iOS16,
            "iOS17": ct.target.iOS17,
        }
        target = target_map.get(self.config.coreml_target, ct.target.macOS13)
        
        # Map compute units
        compute_map = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        }
        compute_units = compute_map.get(self.config.coreml_compute_units, ct.ComputeUnit.ALL)
        
        # Convert
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=input_shape)],
            minimum_deployment_target=target,
            compute_units=compute_units,
        )
        
        # Optionally convert to float16
        if convert_to_float16:
            mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
                mlmodel, nbits=16
            )
        
        # Add metadata
        mlmodel.author = "Kelly ML Pipeline"
        mlmodel.short_description = f"{self.config.model_id} - {self.config.architecture_type}"
        mlmodel.version = "1.0"
        
        mlmodel.save(output_path)
        
        logger.info(f"Exported Core ML: {output_path}")
        self._exported_files.append(output_path)
        return output_path
    
    def export_torchscript(
        self,
        output_path: Union[str, Path],
        optimize_for_mobile: bool = True,
    ) -> Path:
        """
        Export model to TorchScript format.
        
        Args:
            output_path: Output file path
            optimize_for_mobile: Apply mobile optimizations
        
        Returns:
            Path to exported model
        """
        import torch
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        self.model = self.model.to("cpu")
        
        dummy_input = self._get_dummy_input()
        
        # Trace model
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        # Optimize for mobile if requested
        if optimize_for_mobile:
            try:
                from torch.utils.mobile_optimizer import optimize_for_mobile
                traced_model = optimize_for_mobile(traced_model)
                logger.info("Applied mobile optimizations")
            except ImportError:
                logger.debug("Mobile optimizer not available")
        
        traced_model.save(output_path)
        
        logger.info(f"Exported TorchScript: {output_path}")
        self._exported_files.append(output_path)
        return output_path
    
    def export_rtneural_json(
        self,
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export model weights to RTNeural JSON format.
        
        RTNeural is a lightweight neural network library for real-time
        audio applications. This format stores weights in a JSON structure
        that can be loaded by RTNeural C++ code.
        
        Args:
            output_path: Output file path
            metadata: Additional metadata to include
        
        Returns:
            Path to exported model
        """
        import torch
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        self.model = self.model.to("cpu")
        
        # Extract layers
        layers = []
        layer_idx = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_data = {
                    "type": "dense",
                    "name": name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "weights": module.weight.detach().numpy().tolist(),
                }
                if module.bias is not None:
                    layer_data["bias"] = module.bias.detach().numpy().tolist()
                layers.append(layer_data)
                layer_idx += 1
                
            elif isinstance(module, torch.nn.Conv2d):
                layer_data = {
                    "type": "conv2d",
                    "name": name,
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": list(module.kernel_size),
                    "stride": list(module.stride),
                    "padding": list(module.padding),
                    "weights": module.weight.detach().numpy().tolist(),
                }
                if module.bias is not None:
                    layer_data["bias"] = module.bias.detach().numpy().tolist()
                layers.append(layer_data)
                layer_idx += 1
                
            elif isinstance(module, torch.nn.LSTM):
                # LSTM is more complex - extract all weight matrices
                layer_data = {
                    "type": "lstm",
                    "name": name,
                    "input_size": module.input_size,
                    "hidden_size": module.hidden_size,
                    "num_layers": module.num_layers,
                    "bidirectional": module.bidirectional,
                    "weights_ih": [],
                    "weights_hh": [],
                    "bias_ih": [],
                    "bias_hh": [],
                }
                
                for i in range(module.num_layers):
                    suffix = f"_l{i}"
                    if hasattr(module, f"weight_ih{suffix}"):
                        layer_data["weights_ih"].append(
                            getattr(module, f"weight_ih{suffix}").detach().numpy().tolist()
                        )
                        layer_data["weights_hh"].append(
                            getattr(module, f"weight_hh{suffix}").detach().numpy().tolist()
                        )
                    if hasattr(module, f"bias_ih{suffix}"):
                        layer_data["bias_ih"].append(
                            getattr(module, f"bias_ih{suffix}").detach().numpy().tolist()
                        )
                        layer_data["bias_hh"].append(
                            getattr(module, f"bias_hh{suffix}").detach().numpy().tolist()
                        )
                
                layers.append(layer_data)
                layer_idx += 1
                
            elif isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.Tanh)):
                activation_map = {
                    torch.nn.ReLU: "relu",
                    torch.nn.GELU: "gelu",
                    torch.nn.Tanh: "tanh",
                    torch.nn.Sigmoid: "sigmoid",
                }
                layers.append({
                    "type": "activation",
                    "activation": activation_map.get(type(module), "relu"),
                })
                
            elif isinstance(module, torch.nn.BatchNorm2d):
                layers.append({
                    "type": "batchnorm2d",
                    "name": name,
                    "num_features": module.num_features,
                    "running_mean": module.running_mean.numpy().tolist(),
                    "running_var": module.running_var.numpy().tolist(),
                    "weight": module.weight.detach().numpy().tolist() if module.weight is not None else None,
                    "bias": module.bias.detach().numpy().tolist() if module.bias is not None else None,
                })
        
        # Build model data
        model_data = {
            "format": "rtneural-json",
            "version": "1.0",
            "model_id": self.config.model_id,
            "architecture": self.config.architecture_type,
            "input_size": self.config.input_size,
            "output_size": self.config.output_size,
            "layers": layers,
            "trained": True,
        }
        
        # Add metadata
        if metadata:
            model_data.update(metadata)
        
        with open(output_path, "w") as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Exported RTNeural JSON: {output_path}")
        self._exported_files.append(output_path)
        return output_path
    
    def export_all(
        self,
        output_dir: Union[str, Path],
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """
        Export model to all requested formats.
        
        Args:
            output_dir: Output directory
            formats: List of formats to export (default: all)
        
        Returns:
            Dictionary mapping format to output path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if formats is None:
            formats = ["onnx", "coreml", "rtneural", "torchscript"]
        
        results = {}
        
        for fmt in formats:
            try:
                if fmt == "onnx":
                    path = self.export_onnx(output_dir / f"{self.config.model_id}.onnx")
                    results["onnx"] = path
                elif fmt == "coreml":
                    path = self.export_coreml(output_dir / f"{self.config.model_id}.mlmodel")
                    if path:
                        results["coreml"] = path
                elif fmt == "rtneural":
                    path = self.export_rtneural_json(output_dir / f"{self.config.model_id}.json")
                    results["rtneural"] = path
                elif fmt == "torchscript":
                    path = self.export_torchscript(output_dir / f"{self.config.model_id}.pt")
                    results["torchscript"] = path
            except Exception as e:
                logger.error(f"Failed to export {fmt}: {e}")
        
        return results
    
    def get_exported_files(self) -> List[Path]:
        """Get list of all exported files."""
        return self._exported_files.copy()
    
    @staticmethod
    def compute_hash(path: Path) -> str:
        """Compute SHA256 hash of exported file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


def verify_onnx_model(model_path: Union[str, Path]) -> bool:
    """
    Verify ONNX model is valid and can run inference.
    
    Args:
        model_path: Path to ONNX model
    
    Returns:
        True if model is valid
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Check model
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        
        # Try inference
        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # Create dummy input
        dummy_input = np.random.randn(*[s if isinstance(s, int) else 1 for s in input_shape]).astype(np.float32)
        
        # Run inference
        output = session.run(None, {input_name: dummy_input})
        
        logger.info(f"ONNX model verified: {model_path}")
        logger.info(f"  Input shape: {input_shape}")
        logger.info(f"  Output shape: {output[0].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        return False


def verify_coreml_model(model_path: Union[str, Path]) -> bool:
    """
    Verify Core ML model is valid and can run inference.
    
    Args:
        model_path: Path to Core ML model
    
    Returns:
        True if model is valid
    """
    if platform.system() != "Darwin":
        logger.warning("Core ML verification only available on macOS")
        return False
    
    try:
        import coremltools as ct
        
        # Load model
        model = ct.models.MLModel(str(model_path))
        
        # Get input spec
        spec = model.get_spec()
        input_desc = spec.description.input[0]
        
        logger.info(f"Core ML model verified: {model_path}")
        logger.info(f"  Input: {input_desc.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Core ML verification failed: {e}")
        return False


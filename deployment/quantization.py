"""
Model Quantization Scripts
Convert models to INT8, FP16 for faster inference
"""

import torch
import torch.quantization
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vjepa import VJEPAModel


class ModelQuantizer:
    """Quantize models for production deployment"""
    
    @staticmethod
    def quantize_dynamic_int8(model: torch.nn.Module, save_path: str):
        """
        Dynamic INT8 quantization (CPU)
        Reduces model size by ~4x, speeds up CPU inference ~2-3x
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        
        torch.save(quantized_model.state_dict(), save_path)
        print(f"✅ INT8 model saved to {save_path}")
        
        # Size comparison
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        quant_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1e6
        
        print(f"   Original: {original_size:.2f} MB")
        print(f"   Quantized: {quant_size:.2f} MB")
        print(f"   Compression: {original_size/quant_size:.2f}x")
        
        return quantized_model
    
    @staticmethod
    def convert_fp16(model: torch.nn.Module, save_path: str):
        """
        FP16 conversion (GPU)
        Reduces memory by ~2x, speeds up inference ~1.5-2x on modern GPUs
        """
        model_fp16 = model.half()
        
        torch.save(model_fp16.state_dict(), save_path)
        print(f"✅ FP16 model saved to {save_path}")
        
        return model_fp16
    
    @staticmethod
    def export_onnx(
        model: torch.nn.Module,
        save_path: str,
        input_shape: tuple = (1, 16, 3, 224, 224)
    ):
        """
        Export to ONNX format for cross-platform deployment
        """
        model.eval()
        
        dummy_input = torch.randn(*input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['video'],
            output_names=['embeddings'],
            dynamic_axes={
                'video': {0: 'batch_size', 1: 'num_frames'},
                'embeddings': {0: 'batch_size', 1: 'num_frames'}
            }
        )
        
        print(f"✅ ONNX model saved to {save_path}")
        print(f"   Input shape: {input_shape}")
    
    @staticmethod
    def benchmark_inference(model, device='cuda', num_runs=100):
        """Benchmark inference speed"""
        import time
        
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(1, 16, 3, 224, 224).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model.extract_embeddings(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model.extract_embeddings(dummy_input)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end = time.time()
        
        avg_time = (end - start) / num_runs * 1000  # ms
        fps = 1000 / avg_time
        
        print(f"\n⚡ Inference Benchmark ({device}):")
        print(f"   Average time: {avg_time:.2f} ms")
        print(f"   Throughput: {fps:.2f} FPS")
        
        return avg_time


def main():
    """Run quantization pipeline"""
    
    print("🔧 Model Quantization Pipeline\n")
    
    # Load model
    print("Loading V-JEPA model...")
    model = VJEPAModel(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12
    )
    
    output_dir = Path("checkpoints/quantized")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. INT8 Quantization
    print("\n1️⃣ INT8 Dynamic Quantization")
    quantizer = ModelQuantizer()
    int8_model = quantizer.quantize_dynamic_int8(
        model,
        str(output_dir / "vjepa_int8.pth")
    )
    
    # 2. FP16 Conversion
    print("\n2️⃣ FP16 Conversion")
    fp16_model = quantizer.convert_fp16(
        model,
        str(output_dir / "vjepa_fp16.pth")
    )
    
    # 3. ONNX Export
    print("\n3️⃣ ONNX Export")
    quantizer.export_onnx(
        model,
        str(output_dir / "vjepa.onnx")
    )
    
    # 4. Benchmarks
    if torch.cuda.is_available():
        print("\n4️⃣ Benchmarks")
        
        print("\nOriginal FP32:")
        quantizer.benchmark_inference(model, 'cuda', num_runs=50)
        
        print("\nFP16:")
        quantizer.benchmark_inference(fp16_model.cuda(), 'cuda', num_runs=50)
    
    print("\n✅ Quantization complete!")
    print(f"📁 Models saved to: {output_dir}")


if __name__ == "__main__":
    main()

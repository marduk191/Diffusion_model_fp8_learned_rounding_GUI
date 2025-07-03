import argparse
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import gc
import math

# --- Constants ---
AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", "shared"]
DISTILL_AVOID_KEY_NAMES = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]
TARGET_FP8_DTYPE = torch.float8_e4m3fn
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32
MIN_QUANTIZE_SIZE = 1024
# _NEW: Epsilon for clamping scales to prevent them from becoming zero
SCALE_CLAMP_EPS = 1e-8

class PerChannelLearnedRoundingConverter:
    """
    Robust, VRAM-optimized converter using per-channel scaling to prevent quality degradation.
    """
    def __init__(self, num_iter=100, lr=2e-2, reg_lambda=0.01, beta_start=20, beta_end=2,
                 use_amp=True, batch_size=None, min_size_threshold=1024, optimizer='adagrad'):
        # _MODIFIED: Adjusted hyperparameters for faster per-channel optimization
        self.num_iter = num_iter
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.use_amp = use_amp and torch.cuda.is_available()
        self.batch_size = batch_size
        self.min_size_threshold = min_size_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer_choice = optimizer
        
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max
        self.f8_min_pos = torch.finfo(TARGET_FP8_DTYPE).tiny
        self.beta_schedule = torch.linspace(self.beta_start, self.beta_end, self.num_iter)
        
        print(f"Per-Channel converter initialized on device: {self.device} with optimizer: {self.optimizer_choice}")
        if self.use_amp: print("  - Using Automatic Mixed Precision")

    def should_quantize_tensor(self, tensor: torch.Tensor) -> bool:
        if tensor.numel() < self.min_size_threshold: return False
        if tensor.ndim != 2: return False
        if min(tensor.shape) < 32: return False
        return True

    # _MODIFIED: This function is now per-channel and includes robust clamping
    def _compute_per_channel_scales(self, W: torch.Tensor) -> torch.Tensor:
        """Computes a scale for each output channel (row) of the weight matrix."""
        # Find the max absolute value for each row (dim=1)
        w_max_per_channel = W.abs().max(dim=1).values
        
        # Clamp the max values to prevent division by zero or near-zero
        w_max_per_channel = torch.clamp(w_max_per_channel, min=SCALE_CLAMP_EPS)
        
        # Calculate scales per channel
        scales = self.f8_max_val / w_max_per_channel
        
        return scales.to(dtype=SCALE_DTYPE)

    def _batched_matmul(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        if self.batch_size and X.shape[0] > self.batch_size:
            results = [X[i:min(i + self.batch_size, X.shape[0])] @ W.T for i in range(0, X.shape[0], self.batch_size)]
            return torch.cat(results, dim=0)
        return X @ W.T

    # _MODIFIED: The core conversion logic is updated for per-channel operations
    def convert(self, W_orig: torch.Tensor, X_calib: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.should_quantize_tensor(W_orig):
            # For non-quantized tensors, provide a single unity scale
            return W_orig.to(TARGET_FP8_DTYPE), torch.tensor([1.0], dtype=SCALE_DTYPE)
        
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)
        X_calib = X_calib.to(self.device, dtype=COMPUTE_DTYPE)
        
        # Compute per-channel scales. This will be a vector.
        scales = self._compute_per_channel_scales(W_float32)
        
        # Check for all-zero tensor
        if W_float32.abs().max().item() < 1e-12:
            return torch.zeros_like(W_float32, dtype=TARGET_FP8_DTYPE).cpu(), scales.cpu()

        Y_orig = self._batched_matmul(X_calib, W_float32)
        
        # Reshape scales to (num_channels, 1) for broadcasting with the (num_channels, num_features) weight matrix
        scales_bc = scales.to(COMPUTE_DTYPE).unsqueeze(1)
        
        W_scaled = W_float32 * scales_bc
        floor_scaled = torch.floor(W_scaled / self.f8_min_pos) * self.f8_min_pos
        h_init = W_scaled - floor_scaled
        h = torch.nn.Parameter(h_init)
        
        del W_float32, W_scaled, W_orig
        gc.collect()
        if self.device == 'cuda': torch.cuda.empty_cache()

        if self.optimizer_choice == 'adagrad':
            optimizer = torch.optim.Adagrad([h], lr=self.lr)
        else:
            optimizer = torch.optim.AdamW([h], lr=self.lr, weight_decay=1e-6)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_iter)
        scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        pbar = tqdm(range(self.num_iter), desc="    Optimizing", leave=False)
        for i in pbar:
            beta = self.beta_schedule[i].to(self.device)
            reg_loss, total_loss = torch.tensor(0.0), torch.tensor(0.0)

            def step_body():
                nonlocal reg_loss, total_loss
                W_soft_quant = (floor_scaled + h) / scales_bc
                Y_quant = self._batched_matmul(X_calib, W_soft_quant)
                recon_loss = torch.nn.functional.mse_loss(Y_quant, Y_orig)
                h_normalized = 2 * h / self.f8_min_pos - 1
                reg_loss = self.reg_lambda * torch.sum(1 - torch.abs(h_normalized).pow(beta))
                total_loss = recon_loss + reg_loss
                return total_loss

            if self.use_amp and scaler:
                with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                    loss = step_body()
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = step_body()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            with torch.no_grad(): h.clamp_(0, self.f8_min_pos)
            
            current_loss = total_loss.item()
            pbar.set_postfix({"Loss": f"{current_loss:.2e}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})
            
            if i > 10 and (reg_loss.item() < 1e-7 or current_loss < 1e-7):
                pbar.set_description("    Early stop")
                break

        with torch.no_grad():
            W_f8 = (floor_scaled + h.data).to(dtype=TARGET_FP8_DTYPE)

        del X_calib, h, optimizer, Y_orig, floor_scaled, scales_bc
        gc.collect()
        if self.device == 'cuda': torch.cuda.empty_cache()

        return W_f8.cpu(), scales.cpu()


def should_skip_tensor(key: str, tensor: torch.Tensor, t5xxl: bool, keep_distillation: bool, min_size: int) -> bool:
    if tensor.numel() < min_size: return True
    if tensor.ndim != 2 or min(tensor.shape) < 32: return True
    avoid_keywords = []
    if t5xxl: avoid_keywords.extend(AVOID_KEY_NAMES)
    if keep_distillation: avoid_keywords.extend(DISTILL_AVOID_KEY_NAMES)
    return any(avoid_name in key for avoid_name in avoid_keywords)

def convert_to_fp8_scaled(input_file: str, output_file: str, t5xxl: bool, keep_distillation: bool, 
                         calib_samples: int, min_tensor_size: int, **converter_kwargs):
    print(f"Processing: {input_file} -> {output_file}")
    print(f"FP8 format: {TARGET_FP8_DTYPE}, Scale format: {SCALE_DTYPE}, Per-Channel: True")

    tensors: Dict[str, torch.Tensor] = {}
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in tqdm(keys, desc="Loading tensors"):
                tensors[key] = f.get_tensor(key).cpu()
    except Exception as e:
        print(f"Error loading '{input_file}': {e}"); return

    # _MODIFIED: Use the new PerChannel converter
    converter = PerChannelLearnedRoundingConverter(**converter_kwargs, min_size_threshold=min_tensor_size)

    print("\nScanning model for quantizable dimensions...")
    quantizable_tensors = {k: v for k, v in tensors.items() if k.endswith('.weight') and not should_skip_tensor(k, v, t5xxl, keep_distillation, min_tensor_size)}
    unique_dims = set(t.shape[1] for t in quantizable_tensors.values())
    
    calibration_data_cache = {
        dim: torch.randn(calib_samples, dim, dtype=torch.bfloat16)
        for dim in tqdm(sorted(list(unique_dims)), desc="Generating calibration data")
    }
    
    new_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted(tensors.keys())
    
    for key in tqdm(weight_keys, desc="Converting weights"):
        tensor = tensors[key]
        if key in quantizable_tensors:
            in_features = tensor.shape[1]
            calibration_data = calibration_data_cache.get(in_features)
            if calibration_data is None:
                print(f"  - WARNING: No calib data for dim={in_features}. Skipping {key}.")
                new_tensors[key] = tensor
                continue

            quantized_fp8_tensor, scales = converter.convert(tensor, calibration_data)
            new_tensors[key] = quantized_fp8_tensor
            new_tensors[f"{key[:-len('.weight')]}.scale_weight"] = scales
        else:
            # For non-weight tensors or skipped weights, just copy them
            new_tensors[key] = tensor
            # If it was a skipped weight, add a dummy scale for compatibility
            if key.endswith('.weight'):
                new_tensors[f"{key[:-len('.weight')]}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE)

    new_tensors["scaled_fp8"] = torch.empty((0 if t5xxl else 1), dtype=TARGET_FP8_DTYPE)
    new_tensors["scale_format"] = torch.tensor([32], dtype=torch.uint8) # 32 for float32 scales

    print("-" * 50)
    print(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_file(new_tensors, output_file)
        print("Conversion complete!")
    except Exception as e:
        print(f"Error saving file '{output_file}': {e}"); return
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(
        description=f"Convert safetensors to Robust Per-Channel Scaled {TARGET_FP8_DTYPE} format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input", type=str, required=True, help="Input safetensors file path.")
    parser.add_argument("--output", type=str, help="Output file path. Auto-generated if not provided.")
    parser.add_argument("--keep_distillation", action='store_true', help="Exclude distillation layers from quantization.")
    parser.add_argument("--t5xxl", action='store_true', help="Exclude certain layers for T5XXL model compatibility.")
    parser.add_argument("--optimizer", type=str, default="adagrad", choices=["adamw", "adagrad"], help="Optimizer for learned rounding. 'adagrad' uses less VRAM.")
    parser.add_argument("--calib_samples", type=int, default=128, help="Number of random samples for calibration.")
    parser.add_argument("--num_iter", type=int, default=100, help="Optimization iterations per tensor.")
    parser.add_argument("--lr", type=float, default=2e-2, help="Learning rate for optimization.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for calibration matmul to save VRAM. Try 16 or 32.")
    parser.add_argument("--min_tensor_size", type=int, default=1024, help="Minimum number of elements in a tensor to be quantized.")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}"); return

    try:
        _ = torch.zeros(1, dtype=TARGET_FP8_DTYPE)
    except (RuntimeError, TypeError):
        print("Error: PyTorch or hardware does not support torch.float8_e4m3fn."); return

    if not args.output:
        base, _ = os.path.splitext(args.input)
        distill = "_nodistill" if args.keep_distillation else ""
        args.output = f"{base}_fp8_per_channel{distill}.safetensors"

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("Error: Output file cannot be the same as the input file."); return

    converter_kwargs = {
        'num_iter': args.num_iter, 'lr': args.lr,
        'batch_size': args.batch_size, 'optimizer': args.optimizer
    }

    convert_to_fp8_scaled(
        args.input, args.output, args.t5xxl, args.keep_distillation,
        args.calib_samples, args.min_tensor_size,
        **converter_kwargs
    )

if __name__ == "__main__":
    main()
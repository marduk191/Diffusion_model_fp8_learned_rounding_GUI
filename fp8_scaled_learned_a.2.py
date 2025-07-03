import argparse
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import gc
import math
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Keys containing these strings will not be quantized if --t5xxl is set
AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", "shared"]
DISTILL_AVOID_KEY_NAMES = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]

# Target FP8 format
TARGET_FP8_DTYPE = torch.float8_e4m3fn
# Intermediate dtype for calculations
COMPUTE_DTYPE = torch.float32
# Dtype for storing scale factors is now float32
SCALE_DTYPE = torch.float32
# Minimum tensor size to quantize (parameters below this stay in original dtype)
MIN_QUANTIZE_SIZE = 1024

class OptimizedLearnedRoundingConverter:
    """
    VRAM-optimized implementation of adaptive rounding for converting weights to float8.
    """
    def __init__(self, num_iter=200, lr=3e-2, reg_lambda=0.015, beta_start=20, beta_end=2,
                 early_stop_threshold=5e-6, use_amp=True, batch_size=None, min_size_threshold=1024, optimizer='adagrad'):
        self.num_iter = num_iter
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.early_stop_threshold = early_stop_threshold
        self.use_amp = use_amp and torch.cuda.is_available()
        self.batch_size = batch_size
        self.min_size_threshold = min_size_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer_choice = optimizer
        
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max
        self.f8_min_pos = torch.finfo(TARGET_FP8_DTYPE).tiny
        self.beta_schedule = torch.linspace(self.beta_start, self.beta_end, self.num_iter)
        
        print(f"VRAM-optimized converter initialized on device: {self.device}")
        print(f"  - Optimizer: {self.optimizer_choice}")
        print(f"  - Min tensor size for quantization: {self.min_size_threshold}")
        print(f"  - Scale dtype: {SCALE_DTYPE}")
        if self.use_amp:
            print("  - Using Automatic Mixed Precision")

    def should_quantize_tensor(self, tensor: torch.Tensor) -> bool:
        if tensor.numel() < self.min_size_threshold: return False
        if tensor.ndim != 2: return False
        if min(tensor.shape) < 32: return False
        return True

    def _compute_scale_vectorized(self, W: torch.Tensor) -> torch.Tensor:
        w_max = W.abs().max()
        if w_max < 1e-12:
            return torch.tensor([1.0], device=W.device, dtype=SCALE_DTYPE)
        scale = self.f8_max_val / w_max
        return scale.reshape(1).to(dtype=SCALE_DTYPE)

    def _batched_matmul(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        if self.batch_size and X.shape[0] > self.batch_size:
            results = [X[i:min(i + self.batch_size, X.shape[0])] @ W.T for i in range(0, X.shape[0], self.batch_size)]
            return torch.cat(results, dim=0)
        return X @ W.T

    # _MODIFIED: Loop logic rewritten to fix the NameError
    def convert(self, W_orig: torch.Tensor, X_calib: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.should_quantize_tensor(W_orig):
            return W_orig.to(TARGET_FP8_DTYPE), torch.tensor([1.0], dtype=SCALE_DTYPE)
        
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)
        X_calib = X_calib.to(self.device, dtype=COMPUTE_DTYPE)
        
        scale = self._compute_scale_vectorized(W_float32)
        
        if W_float32.abs().max().item() < 1e-12:
            return torch.zeros_like(W_float32, dtype=TARGET_FP8_DTYPE).cpu(), scale.cpu()

        Y_orig = self._batched_matmul(X_calib, W_float32)
        
        W_scaled = W_float32 * scale.to(COMPUTE_DTYPE)
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
        
        scaler = None
        if self.use_amp:
            try: scaler = torch.amp.GradScaler('cuda')
            except Exception: self.use_amp = False; print("Warning: AMP scaler failed, falling back.")
        
        pbar = tqdm(range(self.num_iter), desc="    Optimizing", leave=False)
        
        for i in pbar:
            beta = self.beta_schedule[i].to(self.device)
            
            # Initialize loss variables for the early stopping check
            reg_loss, total_loss = torch.tensor(float('inf')), torch.tensor(float('inf'))

            if self.use_amp and scaler is not None:
                with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32):
                    W_soft_quant = (floor_scaled + h) / scale.to(COMPUTE_DTYPE)
                    Y_quant = self._batched_matmul(X_calib, W_soft_quant)
                    recon_loss = torch.nn.functional.mse_loss(Y_quant, Y_orig)
                    h_normalized = 2 * h / self.f8_min_pos - 1
                    reg_loss = self.reg_lambda * torch.sum(1 - torch.abs(h_normalized).pow(beta))
                    total_loss = recon_loss + reg_loss
                
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                W_soft_quant = (floor_scaled + h) / scale.to(COMPUTE_DTYPE)
                Y_quant = self._batched_matmul(X_calib, W_soft_quant)
                recon_loss = torch.nn.functional.mse_loss(Y_quant, Y_orig)
                h_normalized = 2 * h / self.f8_min_pos - 1
                reg_loss = self.reg_lambda * torch.sum(1 - torch.abs(h_normalized).pow(beta))
                total_loss = recon_loss + reg_loss

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            with torch.no_grad(): h.clamp_(0, self.f8_min_pos)
            
            current_loss = total_loss.item()
            pbar.set_postfix({"Loss": f"{current_loss:.2e}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})
            
            if i > 10 and (reg_loss.item() < 1e-8 or current_loss < 1e-8):
                pbar.set_description("    Early stop")
                break

        with torch.no_grad():
            W_f8 = (floor_scaled + h.data).to(dtype=TARGET_FP8_DTYPE)

        del X_calib, h, optimizer, Y_orig, floor_scaled, total_loss, reg_loss
        gc.collect()
        if self.device == 'cuda': torch.cuda.empty_cache()

        return W_f8.cpu(), scale.cpu()

def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    finfo = torch.finfo(fp8_dtype)
    return float(finfo.min), float(finfo.max), float(finfo.tiny)

FP8_MIN, FP8_MAX, FP8_MIN_POS = get_fp8_constants(TARGET_FP8_DTYPE)

def should_skip_tensor(key: str, tensor: torch.Tensor, t5xxl: bool, keep_distillation: bool, min_size: int = 1024) -> bool:
    if tensor.numel() < min_size: return True
    if tensor.ndim != 2 or min(tensor.shape) < 32: return True
    avoid_keywords = []
    if t5xxl: avoid_keywords.extend(AVOID_KEY_NAMES)
    if keep_distillation: avoid_keywords.extend(DISTILL_AVOID_KEY_NAMES)
    return any(avoid_name in key for avoid_name in avoid_keywords)

def compute_shared_scale(tensors_dict: Dict[str, torch.Tensor], layer_prefix: str, min_size: int = 1024) -> Optional[torch.Tensor]:
    related_tensors = [t for k, t in tensors_dict.items() if k.startswith(layer_prefix) and t.ndim == 2 and t.numel() >= min_size]
    if len(related_tensors) < 2: return None
    global_max = max(t.abs().max().item() for t in related_tensors)
    if global_max < 1e-12: return None
    shared_scale_val = torch.finfo(TARGET_FP8_DTYPE).max / global_max
    return torch.tensor([shared_scale_val], dtype=SCALE_DTYPE)

def convert_to_fp8_scaled(input_file: str, output_file: str, t5xxl: bool, keep_distillation: bool, 
                         calib_samples: int, use_shared_scales: bool = True, min_tensor_size: int = 1024, **converter_kwargs):
    print(f"Processing: {input_file} -> {output_file}")
    print(f"FP8 format: {TARGET_FP8_DTYPE}, Scale format: {SCALE_DTYPE}")

    tensors: Dict[str, torch.Tensor] = {}
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            tensor_keys = list(f.keys())
            print(f"Loading {len(tensor_keys)} tensors...")
            for key in tqdm(tensor_keys, desc="Loading tensors"):
                tensors[key] = f.get_tensor(key).cpu()
    except Exception as e:
        print(f"Error loading '{input_file}': {e}"); return

    converter = OptimizedLearnedRoundingConverter(**converter_kwargs)

    print("\nScanning model for quantizable dimensions...")
    unique_dimensions = set(t.shape[1] for k, t in tensors.items() if k.endswith('.weight') and not should_skip_tensor(k, t, t5xxl, keep_distillation, min_tensor_size))
    
    calibration_data_cache = {
        in_features: torch.randn(calib_samples, in_features, dtype=torch.bfloat16)
        for in_features in tqdm(sorted(list(unique_dimensions)), desc="Generating calibration data")
    }
    
    new_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted([key for key in tensors.keys() if key.endswith('.weight')])
    skipped_count, processed_count, size_skipped_count = 0, 0, 0

    shared_scales = {}
    if use_shared_scales:
        print("Computing shared scales for related tensors...")
        layer_prefixes = set('.'.join(key.split('.')[:-1]) for key in weight_keys if '.' in key)
        for prefix in layer_prefixes:
            shared_scale = compute_shared_scale(tensors, prefix, min_tensor_size)
            if shared_scale is not None: shared_scales[prefix] = shared_scale

    for key in tqdm(weight_keys, desc="Converting weights"):
        original_tensor = tensors[key]
        base_name = key[:-len('.weight')]
        
        if should_skip_tensor(key, original_tensor, t5xxl, keep_distillation, min_tensor_size):
            if original_tensor.numel() < min_tensor_size: size_skipped_count += 1
            else: skipped_count += 1
            new_tensors[key] = original_tensor
            new_tensors[f"{base_name}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE)
            continue

        processed_count += 1
        if original_tensor.numel() == 0:
            new_tensors[key] = original_tensor.to(TARGET_FP8_DTYPE)
            new_tensors[f"{base_name}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE)
            continue

        in_features = original_tensor.shape[1]
        calibration_data = calibration_data_cache.get(in_features)
        if calibration_data is None:
            print(f"  - WARNING: No calibration data for in_features={in_features}. Skipping {key}.")
            new_tensors[key] = original_tensor; skipped_count += 1; processed_count -= 1
            continue

        quantized_fp8_tensor, scale = converter.convert(original_tensor, calibration_data)

        new_tensors[key] = quantized_fp8_tensor
        scale_weight_key = f"{base_name}.scale_weight"
        
        if use_shared_scales and base_name in shared_scales:
            new_tensors[scale_weight_key] = shared_scales[base_name]
        else:
            new_tensors[scale_weight_key] = scale

        if t5xxl:
            scale_input_key = f"{base_name}.scale_input"
            if not torch.allclose(scale, new_tensors[scale_weight_key], atol=1e-4):
                new_tensors[scale_input_key] = scale

    for key, tensor in tensors.items():
        if key not in new_tensors: new_tensors[key] = tensor

    new_tensors["scaled_fp8"] = torch.empty((0 if t5xxl else 1), dtype=TARGET_FP8_DTYPE)
    new_tensors["scale_format"] = torch.tensor([32], dtype=torch.uint8)

    print("-" * 50)
    print(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_file(new_tensors, output_file)
        print("Conversion complete!")
    except Exception as e:
        print(f"Error saving file '{output_file}': {e}"); return

    print("-" * 50)
    print("Summary:")
    print(f"  - Weights processed       : {processed_count}")
    print(f"  - Weights skipped (rules) : {skipped_count}")
    print(f"  - Weights skipped (size)  : {size_skipped_count}")
    if use_shared_scales: print(f"  - Shared scales used      : {len(shared_scales)}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description=f"Convert safetensors weights to VRAM-Efficient Scaled {TARGET_FP8_DTYPE} format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input", type=str, required=True, help="Input safetensors file path.")
    parser.add_argument("--output", type=str, help="Output safetensors file path. If not provided, generated based on input name.")
    parser.add_argument("--keep_distillation", action='store_true', help="Exclude distillation layers from quantization.")
    parser.add_argument("--t5xxl", action='store_true', help="Exclude certain layers for T5XXL model compatibility.")
    parser.add_argument("--optimizer", type=str, default="adagrad", choices=["adamw", "adagrad"], help="Optimizer for learned rounding. 'adagrad' uses much less VRAM.")
    parser.add_argument("--calib_samples", type=int, default=96, help="Number of random samples for calibration.")
    parser.add_argument("--num_iter", type=int, default=200, help="Number of optimization iterations per tensor.")
    parser.add_argument("--lr", type=float, default=3e-2, help="Learning rate for optimization.")
    parser.add_argument("--reg_lambda", type=float, default=0.015, help="Regularization strength.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for calibration matmul to save VRAM. Try 16 or 32.")
    parser.add_argument("--use_amp", action='store_true', default=True, help="Use Automatic Mixed Precision.")
    parser.add_argument("--early_stop_threshold", type=float, default=5e-6, help="Early stopping threshold.")
    parser.add_argument("--min_tensor_size", type=int, default=1024, help="Minimum tensor size to quantize.")
    parser.add_argument("--use_shared_scales", action='store_true', default=True, help="Use shared scale factors to reduce file size.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}"); return

    try:
        _ = torch.zeros(1, dtype=TARGET_FP8_DTYPE)
    except (RuntimeError, TypeError):
        print("Error: PyTorch or hardware does not support torch.float8_e4m3fn."); return

    if not args.output:
        base_name, _ = os.path.splitext(args.input)
        fp8_type_str = TARGET_FP8_DTYPE.__str__().split('.')[-1]
        distill_str = "_nodistill" if args.keep_distillation else ""
        args.output = f"{base_name}_{fp8_type_str}_scaled{distill_str}_fp32scale.safetensors"

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("Error: Output file cannot be the same as the input file."); return

    converter_kwargs = {k: v for k, v in vars(args).items() if k in [
        'num_iter', 'lr', 'reg_lambda', 'use_amp', 'early_stop_threshold',
        'batch_size', 'min_size_threshold', 'optimizer']}

    convert_to_fp8_scaled(
        args.input, args.output, args.t5xxl, args.keep_distillation,
        args.calib_samples, args.use_shared_scales, args.min_tensor_size,
        **converter_kwargs
    )

if __name__ == "__main__":
    main()
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
# Target FP8 format
TARGET_FP8_DTYPE = torch.float8_e4m3fn
# Intermediate dtype for calculations
COMPUTE_DTYPE = torch.float32
# Dtype for storing scale factors
SCALE_DTYPE = torch.float32

class OptimizedLearnedRoundingConverter:
    """
    Optimized implementation of adaptive rounding for converting weights to float8.
    Key optimizations:
    1. Reduced precision operations where possible
    2. Early stopping with better convergence detection
    3. Vectorized operations
    4. Memory-efficient computation
    5. Adaptive learning rate scheduling
    """
    def __init__(self, num_iter=300, lr=2e-2, reg_lambda=0.01, beta_start=20, beta_end=2, 
                 early_stop_threshold=1e-6, use_amp=True, batch_size=None):
        self.num_iter = num_iter
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.early_stop_threshold = early_stop_threshold
        self.use_amp = use_amp and torch.cuda.is_available()
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Pre-compute FP8 constants
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max
        self.f8_min_pos = torch.finfo(TARGET_FP8_DTYPE).tiny
        
        # Pre-allocate beta schedule
        self.beta_schedule = torch.linspace(self.beta_start, self.beta_end, self.num_iter)
        
        print(f"OptimizedLearnedRoundingConverter initialized on device: {self.device}")
        if self.use_amp:
            print("  - Using Automatic Mixed Precision")

    def _compute_scale_vectorized(self, W: torch.Tensor) -> torch.Tensor:
        """Vectorized scale computation"""
        w_max = W.abs().max()
        if w_max < 1e-12:
            return torch.tensor(1.0, device=W.device)
        return self.f8_max_val / w_max

    def _batched_matmul(self, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Memory-efficient batched matrix multiplication"""
        if self.batch_size and X.shape[0] > self.batch_size:
            results = []
            for i in range(0, X.shape[0], self.batch_size):
                batch_end = min(i + self.batch_size, X.shape[0])
                batch_result = X[i:batch_end] @ W.T
                results.append(batch_result)
            return torch.cat(results, dim=0)
        return X @ W.T

    def convert(self, W_orig: torch.Tensor, X_calib: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized conversion with early stopping and adaptive learning"""
        
        # Move to device with optimal dtype
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)
        X_calib = X_calib.to(self.device, dtype=COMPUTE_DTYPE)
        
        # Quick zero check
        scale = self._compute_scale_vectorized(W_float32)
        if scale.item() == 1.0:  # All zeros case
            quantized_tensor = torch.zeros_like(W_float32, dtype=TARGET_FP8_DTYPE)
            return quantized_tensor.cpu(), scale.reciprocal().cpu().reshape(1)

        W_scaled = W_float32 * scale
        
        # More efficient initialization
        floor_scaled = torch.floor(W_scaled / self.f8_min_pos) * self.f8_min_pos
        h_init = W_scaled - floor_scaled
        h = torch.nn.Parameter(h_init)
        
        # Use Adam optimizer with adaptive learning rate
        optimizer = torch.optim.AdamW([h], lr=self.lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_iter)
        
        # Pre-compute reference output for efficiency
        Y_orig = self._batched_matmul(X_calib, W_float32)
        
        # Optimization tracking
        best_loss = float('inf')
        patience_counter = 0
        patience_limit = 50  # Early stopping patience
        
        # AMP scaler if using mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        pbar = tqdm(range(self.num_iter), desc="    Optimizing rounding", leave=False)
        
        for i in pbar:
            beta = self.beta_schedule[i].to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # Compute soft quantized weights
                    W_soft_quant = (floor_scaled + h) / scale
                    Y_quant = self._batched_matmul(X_calib, W_soft_quant)
                    
                    # Compute losses
                    recon_loss = torch.nn.functional.mse_loss(Y_quant, Y_orig)
                    
                    # Vectorized regularization loss
                    h_normalized = 2 * h / self.f8_min_pos - 1
                    reg_loss = self.reg_lambda * torch.sum(1 - torch.abs(h_normalized).pow(beta))
                    
                    total_loss = recon_loss + reg_loss
                
                # Backward pass with scaling
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision path
                W_soft_quant = (floor_scaled + h) / scale
                Y_quant = self._batched_matmul(X_calib, W_soft_quant)
                
                recon_loss = torch.nn.functional.mse_loss(Y_quant, Y_orig)
                h_normalized = 2 * h / self.f8_min_pos - 1
                reg_loss = self.reg_lambda * torch.sum(1 - torch.abs(h_normalized).pow(beta))
                
                total_loss = recon_loss + reg_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Clamp h values
            with torch.no_grad():
                h.clamp_(0, self.f8_min_pos)
            
            # Early stopping logic
            current_loss = total_loss.item()
            if current_loss < best_loss - self.early_stop_threshold:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            pbar.set_postfix({
                "Recon": f"{recon_loss.item():.2e}", 
                "Reg": f"{reg_loss.item():.2e}",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Early stopping conditions
            if patience_counter >= patience_limit or reg_loss.item() < 1e-8:
                pbar.set_description("    Early stopping")
                break

        # Final quantization
        with torch.no_grad():
            W_quant_final_scaled = floor_scaled + h.data
            W_f8 = W_quant_final_scaled.to(dtype=TARGET_FP8_DTYPE)

        dequant_scale = scale.reciprocal().reshape(1)
        
        # Aggressive memory cleanup
        del W_float32, X_calib, h, optimizer, Y_orig, W_soft_quant, W_scaled, floor_scaled
        if 'Y_quant' in locals():
            del Y_quant
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return W_f8.cpu(), dequant_scale.cpu()

def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    """Gets the min, max, and smallest positive normal value for a given FP8 dtype."""
    finfo = torch.finfo(fp8_dtype)
    return float(finfo.min), float(finfo.max), float(finfo.tiny)

# Global FP8 constants
FP8_MIN, FP8_MAX, FP8_MIN_POS = get_fp8_constants(TARGET_FP8_DTYPE)

def process_tensor_batch(tensor_batch, converter, calibration_data_cache):
    """Process a batch of tensors - for potential parallel processing"""
    results = {}
    
    for key, original_tensor in tensor_batch:
        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            results[key] = {
                'tensor': original_tensor.to(TARGET_FP8_DTYPE),
                'scale': torch.tensor([1.0], dtype=SCALE_DTYPE),
                'skipped': True
            }
            continue
            
        in_features = original_tensor.shape[1]
        if in_features not in calibration_data_cache:
            results[key] = {
                'tensor': original_tensor,
                'scale': None,
                'skipped': True
            }
            continue
            
        calibration_data = calibration_data_cache[in_features]
        quantized_fp8_tensor, dequant_scale = converter.convert(original_tensor, calibration_data)
        
        results[key] = {
            'tensor': quantized_fp8_tensor,
            'scale': dequant_scale.to(SCALE_DTYPE),
            'skipped': False
        }
    
    return results

def convert_to_fp8_scaled(input_file: str, output_file: str, t5xxl: bool, keep_distillation: bool, 
                         calib_samples: int, parallel_processing: bool = False, **converter_kwargs):
    """
    Optimized conversion function with optional parallel processing and better memory management.
    """
    print(f"Processing: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Using FP8 format: {TARGET_FP8_DTYPE}")
    print(f"FP8 Range: [{FP8_MIN}, {FP8_MAX}]")
    print(f"FP8 Min Precision: [{FP8_MIN_POS}]")

    # Load tensors with memory mapping for large files
    tensors: Dict[str, torch.Tensor] = {}
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            tensor_keys = list(f.keys())
            print(f"Loading {len(tensor_keys)} tensors...")
            
            # Load in chunks to manage memory
            for key in tqdm(tensor_keys, desc="Loading tensors"):
                tensors[key] = f.get_tensor(key).cpu()
                
    except Exception as e:
        print(f"Error loading '{input_file}': {e}")
        return

    # Enhanced converter with optimizations
    converter = OptimizedLearnedRoundingConverter(**converter_kwargs)

    # Optimized calibration data generation
    print("\nScanning model for linear layer dimensions...")
    calibration_data_cache = {}
    unique_dimensions = set()
    
    # First pass: collect unique dimensions
    for key, tensor in tensors.items():
        if key.endswith('.weight') and tensor.ndim == 2:
            unique_dimensions.add(tensor.shape[1])
    
    # Generate calibration data for unique dimensions
    print(f"Found {len(unique_dimensions)} unique input dimensions")
    for in_features in sorted(unique_dimensions):
        print(f"  - Generating calibration data for dimension: {in_features}")
        # Use bfloat16 for memory efficiency during calibration
        calibration_data_cache[in_features] = torch.randn(
            calib_samples, in_features, dtype=torch.bfloat16
        )
    
    print("Calibration data generated.\n")

    new_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted([key for key in tensors.keys() if key.endswith('.weight')])
    total_weights = len(weight_keys)
    skipped_count = 0
    processed_count = 0

    print(f"Found {total_weights} weight tensors to potentially process.")

    # Process tensors with progress tracking
    for i, key in enumerate(weight_keys):
        process_this_key = True
        
        # Check exclusion rules
        if t5xxl and any(avoid_name in key for avoid_name in AVOID_KEY_NAMES):
            print(f"({i+1}/{total_weights}) Skipping excluded T5XXL tensor: {key}")
            new_tensors[key] = tensors[key]
            process_this_key = False
            skipped_count += 1

        if keep_distillation and any(avoid_name in key for avoid_name in 
                                   ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]):
            print(f"({i+1}/{total_weights}) Skipping excluded distillation tensor: {key}")
            new_tensors[key] = tensors[key]
            base_name = key[:-len('.weight')]
            scale_weight_key = f"{base_name}.scale_weight"
            new_tensors[scale_weight_key] = torch.tensor([1.0], dtype=SCALE_DTYPE)
            process_this_key = False
            skipped_count += 1

        if not process_this_key:
            continue

        print(f"({i+1}/{total_weights}) Processing tensor: {key}")
        processed_count += 1

        original_tensor = tensors[key]

        # Handle edge cases
        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            print(f"  - Skipping empty or non-2D tensor: {key}")
            new_tensors[key] = original_tensor.to(TARGET_FP8_DTYPE)
            base_name = key[:-len('.weight')]
            scale_weight_key = f"{base_name}.scale_weight"
            new_tensors[scale_weight_key] = torch.tensor([1.0], dtype=SCALE_DTYPE)
            continue

        in_features = original_tensor.shape[1]
        if in_features not in calibration_data_cache:
            print(f"  - WARNING: No calibration data found for in_features={in_features}. Skipping {key}")
            new_tensors[key] = original_tensor
            skipped_count += 1
            processed_count -= 1
            continue

        calibration_data = calibration_data_cache[in_features]

        # Convert with learned rounding
        quantized_fp8_tensor, dequant_scale = converter.convert(original_tensor, calibration_data)

        # Store results
        new_tensors[key] = quantized_fp8_tensor
        base_name = key[:-len('.weight')]
        scale_weight_key = f"{base_name}.scale_weight"
        new_tensors[scale_weight_key] = dequant_scale.to(SCALE_DTYPE)
        
        if t5xxl:
            scale_input_key = f"{base_name}.scale_input"
            new_tensors[scale_input_key] = dequant_scale.detach().clone().to(SCALE_DTYPE)

        print(f"  - Dequant Scale: {dequant_scale.item():.9f}")

    # Add remaining tensors
    for key, tensor in tensors.items():
        if key not in new_tensors:
            new_tensors[key] = tensor

    # Add format marker
    new_tensors["scaled_fp8"] = (torch.empty((2), dtype=TARGET_FP8_DTYPE) 
                                if not t5xxl else torch.empty((0), dtype=TARGET_FP8_DTYPE))

    print("-" * 40)
    print(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_file(new_tensors, output_file)
        print("Conversion complete!")
    except Exception as e:
        print(f"Error saving file '{output_file}': {e}")
        return

    print("-" * 40)
    print("Summary:")
    print(f"  - Original tensor count : {len(tensors)}")
    print(f"  - Weights processed     : {processed_count}")
    print(f"  - Weights skipped       : {skipped_count}")
    print(f"  - Final tensor count    : {len(new_tensors)}")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(
        description=f"Convert safetensors weights to Scaled {TARGET_FP8_DTYPE} format using optimized learned rounding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Original arguments
    parser.add_argument("--input", type=str, required=True, help="Input safetensors file path.")
    parser.add_argument("--output", type=str, help="Output safetensors file path. If not provided, generated based on input name.")
    parser.add_argument("--keep_distillation", action='store_true', help="Exclude distillation layers from quantization.")
    parser.add_argument("--t5xxl", action='store_true', help="Exclude certain layers for T5XXL model compatibility.")

    # Optimization arguments
    parser.add_argument("--calib_samples", type=int, default=128, help="Number of random samples for calibration (reduced from 256).")
    parser.add_argument("--num_iter", type=int, default=300, help="Number of optimization iterations per tensor (reduced from 512).")
    parser.add_argument("--lr", type=float, default=2e-2, help="Learning rate for the rounding optimizer (increased for faster convergence).")
    parser.add_argument("--reg_lambda", type=float, default=0.01, help="Regularization strength for the rounding loss.")
    parser.add_argument("--batch_size", type=int, help="Batch size for calibration matmul (memory optimization).")
    parser.add_argument("--use_amp", action='store_true', default=True, help="Use Automatic Mixed Precision for faster training.")
    parser.add_argument("--early_stop_threshold", type=float, default=1e-6, help="Early stopping threshold for convergence.")
    parser.add_argument("--parallel", action='store_true', help="Enable experimental parallel processing (may use more memory).")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    # Check for FP8 support
    try:
        _ = torch.zeros(1, dtype=TARGET_FP8_DTYPE)
    except (RuntimeError, TypeError):
        print("Error: This version of PyTorch or this hardware does not support torch.float8_e4m3fn.")
        return

    # Generate output filename
    fp8_type_str = TARGET_FP8_DTYPE.__str__().split('.')[-1]
    distill_str = "_nodistill" if args.keep_distillation else ""
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        output_file = f"{base_name}_{fp8_type_str}_scaled_learned_opt{distill_str}.safetensors"
    else:
        output_file = args.output

    if os.path.abspath(args.input) == os.path.abspath(output_file):
        print("Error: Output file cannot be the same as the input file.")
        return

    # Prepare converter arguments
    converter_kwargs = {
        'num_iter': args.num_iter,
        'lr': args.lr,
        'reg_lambda': args.reg_lambda,
        'use_amp': args.use_amp,
        'early_stop_threshold': args.early_stop_threshold,
        'batch_size': args.batch_size,
    }

    convert_to_fp8_scaled(
        args.input,
        output_file,
        args.t5xxl,
        args.keep_distillation,
        args.calib_samples,
        args.parallel,
        **converter_kwargs
    )

if __name__ == "__main__":
    main()
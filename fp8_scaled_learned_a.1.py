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
# Dtype for storing scale factors - ULTRA REDUCED PRECISION FOR MAXIMUM SIZE SAVINGS
SCALE_DTYPE = torch.uint8  # Changed from float16 to uint8 for maximum compression
# Scale quantization parameters - FIXED RANGE
SCALE_MIN_LOG = -12.0  # log10 of minimum scale value (1e-12)
SCALE_MAX_LOG = 6.0    # log10 of maximum scale value (1e6) - much larger range
SCALE_QUANTIZATION_LEVELS = 255  # Use full uint8 range
# Minimum tensor size to quantize (parameters below this stay in original dtype)
MIN_QUANTIZE_SIZE = 1024  # Only quantize tensors with >1024 elements

def quantize_scale_to_uint8(scale: torch.Tensor) -> torch.Tensor:
    """Convert float scale to quantized uint8 representation - FIXED VERSION"""
    # Don't clamp the input scale too aggressively - let the log space handle it
    scale_clamped = torch.clamp(scale, min=1e-12, max=1e6)  # Much wider range
    
    # Convert to log scale for better precision distribution
    log_scale = torch.log10(scale_clamped)
    
    # Normalize to [0, 1] range
    normalized = (log_scale - SCALE_MIN_LOG) / (SCALE_MAX_LOG - SCALE_MIN_LOG)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    # Quantize to uint8
    quantized = torch.round(normalized * SCALE_QUANTIZATION_LEVELS).to(torch.uint8)
    return quantized

def dequantize_scale_from_uint8(quantized_scale: torch.Tensor) -> torch.Tensor:
    """Convert quantized uint8 scale back to float - FIXED VERSION"""
    # Normalize back to [0, 1]
    normalized = quantized_scale.to(torch.float32) / SCALE_QUANTIZATION_LEVELS
    
    # Convert back to log scale
    log_scale = normalized * (SCALE_MAX_LOG - SCALE_MIN_LOG) + SCALE_MIN_LOG
    
    # Convert back to linear scale
    scale = torch.pow(10.0, log_scale)
    return scale

class OptimizedLearnedRoundingConverter:
    """
    Size-optimized implementation of adaptive rounding for converting weights to float8.
    Key optimizations for file size:
    1. Skip quantization of very small tensors
    2. Use shared scale factors where possible
    3. Ultra-reduced precision (8-bit) for scale storage
    4. Early convergence detection to avoid over-optimization
    """
    def __init__(self, num_iter=200, lr=3e-2, reg_lambda=0.015, beta_start=20, beta_end=2, 
                 early_stop_threshold=5e-6, use_amp=True, batch_size=None, min_size_threshold=1024):
        self.num_iter = num_iter  # Reduced from 300
        self.lr = lr  # Increased for faster convergence
        self.reg_lambda = reg_lambda
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.early_stop_threshold = early_stop_threshold
        self.use_amp = use_amp and torch.cuda.is_available()
        self.batch_size = batch_size
        self.min_size_threshold = min_size_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Pre-compute FP8 constants
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max
        self.f8_min_pos = torch.finfo(TARGET_FP8_DTYPE).tiny
        
        # Pre-allocate beta schedule
        self.beta_schedule = torch.linspace(self.beta_start, self.beta_end, self.num_iter)
        
        print(f"Size-optimized converter initialized on device: {self.device}")
        print(f"  - Min tensor size for quantization: {self.min_size_threshold}")
        print(f"  - Scale dtype: {SCALE_DTYPE} (8-bit quantized)")
        print(f"  - Scale range: {10**SCALE_MIN_LOG:.2e} to {10**SCALE_MAX_LOG:.2e}")
        if self.use_amp:
            print("  - Using Automatic Mixed Precision")

    def should_quantize_tensor(self, tensor: torch.Tensor) -> bool:
        """Determine if a tensor should be quantized based on size and characteristics"""
        if tensor.numel() < self.min_size_threshold:
            return False
        if tensor.ndim != 2:
            return False
        # Skip very small matrices where quantization overhead isn't worth it
        if min(tensor.shape) < 32:
            return False
        return True

    def _compute_scale_vectorized(self, W: torch.Tensor) -> torch.Tensor:
        """Vectorized scale computation with uint8 quantized output"""
        w_max = W.abs().max()
        if w_max < 1e-12:
            # Return quantized version of scale=1.0
            return quantize_scale_to_uint8(torch.tensor(1.0, device=W.device))
        
        scale = self.f8_max_val / w_max
        # Quantize the scale to uint8
        return quantize_scale_to_uint8(scale)

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
        """Size-optimized conversion with aggressive early stopping and 8-bit scales"""
        
        # Check if tensor should be quantized
        if not self.should_quantize_tensor(W_orig):
            # Return original tensor and quantized unity scale
            unity_scale_quantized = quantize_scale_to_uint8(torch.tensor([1.0]))
            return W_orig.to(TARGET_FP8_DTYPE), unity_scale_quantized
        
        # Move to device with optimal dtype
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)
        X_calib = X_calib.to(self.device, dtype=COMPUTE_DTYPE)
        
        # Quick zero check
        quantized_scale = self._compute_scale_vectorized(W_float32)
        scale = dequantize_scale_from_uint8(quantized_scale).to(self.device)
        
        if scale.item() == dequantize_scale_from_uint8(quantize_scale_to_uint8(torch.tensor([1.0]))).item():  # All zeros case
            quantized_tensor = torch.zeros_like(W_float32, dtype=TARGET_FP8_DTYPE)
            return quantized_tensor.cpu(), quantized_scale.cpu()

        W_scaled = W_float32 * scale.to(COMPUTE_DTYPE)
        
        # More efficient initialization
        floor_scaled = torch.floor(W_scaled / self.f8_min_pos) * self.f8_min_pos
        h_init = W_scaled - floor_scaled
        h = torch.nn.Parameter(h_init)
        
        # Use Adam optimizer with adaptive learning rate
        optimizer = torch.optim.AdamW([h], lr=self.lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_iter)
        
        # Pre-compute reference output for efficiency
        Y_orig = self._batched_matmul(X_calib, W_float32)
        
        # Optimization tracking with more aggressive early stopping
        best_loss = float('inf')
        patience_counter = 0
        patience_limit = 30  # Reduced patience for faster convergence
        convergence_threshold = 1e-7  # Tighter convergence
        
        # AMP scaler with updated API
        scaler = None
        if self.use_amp:
            try:
                # Try new API first (PyTorch 2.1+)
                scaler = torch.amp.GradScaler('cuda')
            except (AttributeError, TypeError):
                # Fall back to old API for older PyTorch versions
                try:
                    scaler = torch.cuda.amp.GradScaler()
                except:
                    self.use_amp = False
                    print("Warning: Could not initialize AMP scaler, falling back to standard precision")
        
        pbar = tqdm(range(self.num_iter), desc="    Optimizing", leave=False)
        
        for i in pbar:
            beta = self.beta_schedule[i].to(self.device)
            
            if self.use_amp and scaler is not None:
                # Use the new API for autocast
                with torch.amp.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                    # Compute soft quantized weights
                    W_soft_quant = (floor_scaled + h) / scale.to(COMPUTE_DTYPE)
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
                W_soft_quant = (floor_scaled + h) / scale.to(COMPUTE_DTYPE)
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
            
            # Aggressive early stopping logic
            current_loss = total_loss.item()
            if current_loss < best_loss - convergence_threshold:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            pbar.set_postfix({
                "Loss": f"{current_loss:.2e}",
                "LR": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Multiple early stopping conditions
            if (patience_counter >= patience_limit or 
                reg_loss.item() < 1e-8 or 
                current_loss < 1e-8):
                pbar.set_description("    Early stop")
                break

        # Final quantization
        with torch.no_grad():
            W_quant_final_scaled = floor_scaled + h.data
            W_f8 = W_quant_final_scaled.to(dtype=TARGET_FP8_DTYPE)

        # Return quantized scale (already computed)
        
        # Aggressive memory cleanup
        del W_float32, X_calib, h, optimizer, Y_orig, W_soft_quant, W_scaled, floor_scaled
        if 'Y_quant' in locals():
            del Y_quant
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return W_f8.cpu(), quantized_scale.cpu()

def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    """Gets the min, max, and smallest positive normal value for a given FP8 dtype."""
    finfo = torch.finfo(fp8_dtype)
    return float(finfo.min), float(finfo.max), float(finfo.tiny)

# Global FP8 constants
FP8_MIN, FP8_MAX, FP8_MIN_POS = get_fp8_constants(TARGET_FP8_DTYPE)

def should_skip_tensor(key: str, tensor: torch.Tensor, t5xxl: bool, keep_distillation: bool, min_size: int = 1024) -> bool:
    """Centralized logic for determining if a tensor should be skipped"""
    # T5XXL exclusions
    if t5xxl and any(avoid_name in key for avoid_name in AVOID_KEY_NAMES):
        return True
    
    # Distillation exclusions  
    if keep_distillation and any(avoid_name in key for avoid_name in 
                               ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]):
        return True
    
    # Size-based exclusions
    if tensor.numel() < min_size:
        return True
        
    # Shape-based exclusions
    if tensor.ndim != 2 or min(tensor.shape) < 32:
        return True
        
    return False

def compute_shared_scale(tensors_dict: Dict[str, torch.Tensor], layer_prefix: str, min_size: int = 1024) -> Optional[torch.Tensor]:
    """Compute a shared scale factor for related tensors to save space (returns quantized uint8 scale)"""
    related_tensors = []
    for key, tensor in tensors_dict.items():
        if key.startswith(layer_prefix) and tensor.ndim == 2 and tensor.numel() >= min_size:
            related_tensors.append(tensor)
    
    if len(related_tensors) < 2:
        return None
    
    # Compute global max across all related tensors
    global_max = max(tensor.abs().max().item() for tensor in related_tensors)
    if global_max < 1e-12:
        return None
    
    shared_scale = torch.finfo(TARGET_FP8_DTYPE).max / global_max
    # Quantize the shared scale to uint8
    return quantize_scale_to_uint8(torch.tensor([shared_scale]))

def convert_to_fp8_scaled(input_file: str, output_file: str, t5xxl: bool, keep_distillation: bool, 
                         calib_samples: int, use_shared_scales: bool = True, min_tensor_size: int = 1024, **converter_kwargs):
    """
    Size-optimized conversion function with shared scales and selective quantization.
    Now uses 8-bit quantized scales for maximum compression.
    """
    print(f"Processing: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Using FP8 format: {TARGET_FP8_DTYPE}")
    print(f"Scale precision: {SCALE_DTYPE} (8-bit quantized)")
    print(f"Scale range: {10**SCALE_MIN_LOG:.2e} to {10**SCALE_MAX_LOG:.2e}")
    print(f"Min tensor size for quantization: {min_tensor_size}")
    print(f"Shared scales enabled: {use_shared_scales}")

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

    # Enhanced converter with size optimizations
    converter = OptimizedLearnedRoundingConverter(**converter_kwargs)

    # Reduced calibration data generation (smaller samples for speed)
    print("\nScanning model for quantizable dimensions...")
    calibration_data_cache = {}
    unique_dimensions = set()
    
    # First pass: collect unique dimensions only for tensors we'll actually quantize
    for key, tensor in tensors.items():
        if (key.endswith('.weight') and 
            not should_skip_tensor(key, tensor, t5xxl, keep_distillation, min_tensor_size)):
            unique_dimensions.add(tensor.shape[1])
    
    # Generate calibration data for unique dimensions
    print(f"Found {len(unique_dimensions)} unique input dimensions for quantization")
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
    size_skipped_count = 0

    print(f"Found {total_weights} weight tensors to evaluate.")

    # Shared scale computation (optional optimization)
    shared_scales = {}
    if use_shared_scales:
        print("Computing shared scales for related tensors...")
        # Group tensors by layer prefix for shared scaling
        layer_prefixes = set()
        for key in weight_keys:
            if '.' in key:
                layer_prefix = '.'.join(key.split('.')[:-1])  # Remove '.weight'
                layer_prefixes.add(layer_prefix)
        
        for prefix in layer_prefixes:
            shared_scale = compute_shared_scale(tensors, prefix, min_tensor_size)
            if shared_scale is not None:
                shared_scales[prefix] = shared_scale

    # Process tensors with progress tracking
    for i, key in enumerate(weight_keys):
        original_tensor = tensors[key]
        
        # Apply skip logic
        if should_skip_tensor(key, original_tensor, t5xxl, keep_distillation, min_tensor_size):
            if original_tensor.numel() < min_tensor_size:
                print(f"({i+1}/{total_weights}) Skipping small tensor: {key} (size: {original_tensor.numel()})")
                size_skipped_count += 1
            else:
                print(f"({i+1}/{total_weights}) Skipping excluded tensor: {key}")
                skipped_count += 1
            
            new_tensors[key] = original_tensor
            base_name = key[:-len('.weight')]
            scale_weight_key = f"{base_name}.scale_weight"
            # Store quantized unity scale
            new_tensors[scale_weight_key] = quantize_scale_to_uint8(torch.tensor([1.0]))
            continue

        print(f"({i+1}/{total_weights}) Processing tensor: {key}")
        processed_count += 1

        # Handle edge cases for quantizable tensors
        if original_tensor.numel() == 0:
            print(f"  - Skipping empty tensor: {key}")
            new_tensors[key] = original_tensor.to(TARGET_FP8_DTYPE)
            base_name = key[:-len('.weight')]
            scale_weight_key = f"{base_name}.scale_weight"
            new_tensors[scale_weight_key] = quantize_scale_to_uint8(torch.tensor([1.0]))
            continue

        in_features = original_tensor.shape[1]
        if in_features not in calibration_data_cache:
            print(f"  - WARNING: No calibration data for in_features={in_features}. Skipping {key}")
            new_tensors[key] = original_tensor
            skipped_count += 1
            processed_count -= 1
            continue

        calibration_data = calibration_data_cache[in_features]

        # Convert with learned rounding (now returns quantized scale)
        quantized_fp8_tensor, quantized_scale = converter.convert(original_tensor, calibration_data)

        # Store results with ultra-optimized scale storage
        new_tensors[key] = quantized_fp8_tensor
        base_name = key[:-len('.weight')]
        scale_weight_key = f"{base_name}.scale_weight"
        
        # Use shared scale if available, otherwise individual scale
        if use_shared_scales and base_name in shared_scales:
            new_tensors[scale_weight_key] = shared_scales[base_name]
        else:
            new_tensors[scale_weight_key] = quantized_scale

        # For T5XXL, only add scale_input if it's different from scale_weight
        if t5xxl:
            scale_input_key = f"{base_name}.scale_input"
            # Only store if different (save space) - compare dequantized values
            dequant_scale = dequantize_scale_from_uint8(quantized_scale)
            dequant_shared = dequantize_scale_from_uint8(new_tensors[scale_weight_key])
            if not torch.allclose(dequant_scale, dequant_shared, atol=1e-4):  # Looser tolerance for 8-bit
                new_tensors[scale_input_key] = quantized_scale

        # Print dequantized scale for reference
        dequant_scale_val = dequantize_scale_from_uint8(quantized_scale).item()
        print(f"  - Scale: {dequant_scale_val:.6f} (quantized: {quantized_scale.item()})")

    # Add remaining tensors (non-weights)
    for key, tensor in tensors.items():
        if key not in new_tensors:
            new_tensors[key] = tensor

    # Add minimal format marker with scale info
    if t5xxl:
        new_tensors["scaled_fp8"] = torch.empty((0), dtype=TARGET_FP8_DTYPE)
        new_tensors["scale_format"] = torch.tensor([8], dtype=torch.uint8)  # Indicate 8-bit scales
    else:
        new_tensors["scaled_fp8"] = torch.empty((1), dtype=TARGET_FP8_DTYPE)  # Reduced size
        new_tensors["scale_format"] = torch.tensor([8], dtype=torch.uint8)  # Indicate 8-bit scales

    print("-" * 50)
    print(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_file(new_tensors, output_file)
        print("Conversion complete!")
    except Exception as e:
        print(f"Error saving file '{output_file}': {e}")
        return

    print("-" * 50)
    print("Summary:")
    print(f"  - Original tensor count   : {len(tensors)}")
    print(f"  - Weights processed       : {processed_count}")
    print(f"  - Weights skipped (rules) : {skipped_count}")
    print(f"  - Weights skipped (size)  : {size_skipped_count}")
    print(f"  - Final tensor count      : {len(new_tensors)}")
    print(f"  - Scale format            : 8-bit quantized (uint8)")
    print(f"  - Scale range             : {10**SCALE_MIN_LOG:.2e} to {10**SCALE_MAX_LOG:.2e}")
    if use_shared_scales:
        print(f"  - Shared scales used      : {len(shared_scales)}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description=f"Convert safetensors weights to Size-Optimized Scaled {TARGET_FP8_DTYPE} format with 8-bit scales.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Original arguments
    parser.add_argument("--input", type=str, required=True, help="Input safetensors file path.")
    parser.add_argument("--output", type=str, help="Output safetensors file path. If not provided, generated based on input name.")
    parser.add_argument("--keep_distillation", action='store_true', help="Exclude distillation layers from quantization.")
    parser.add_argument("--t5xxl", action='store_true', help="Exclude certain layers for T5XXL model compatibility.")

    # Size optimization arguments
    parser.add_argument("--calib_samples", type=int, default=96, help="Number of random samples for calibration (reduced for speed).")
    parser.add_argument("--num_iter", type=int, default=200, help="Number of optimization iterations per tensor (reduced for speed).")
    parser.add_argument("--lr", type=float, default=3e-2, help="Learning rate (increased for faster convergence).")
    parser.add_argument("--reg_lambda", type=float, default=0.015, help="Regularization strength.")
    parser.add_argument("--batch_size", type=int, help="Batch size for calibration matmul (memory optimization).")
    parser.add_argument("--use_amp", action='store_true', default=True, help="Use Automatic Mixed Precision.")
    parser.add_argument("--early_stop_threshold", type=float, default=5e-6, help="Early stopping threshold.")
    parser.add_argument("--min_tensor_size", type=int, default=1024, help="Minimum tensor size to quantize.")
    parser.add_argument("--use_shared_scales", action='store_true', default=True, help="Use shared scale factors to reduce file size.")
    parser.add_argument("--parallel", action='store_true', help="Enable experimental parallel processing.")

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

    # Update global minimum size if specified
    min_tensor_size = args.min_tensor_size

    # Generate output filename
    fp8_type_str = TARGET_FP8_DTYPE.__str__().split('.')[-1]
    distill_str = "_nodistill" if args.keep_distillation else ""
    size_str = "_8bit" if not args.use_shared_scales else "_8bit_shared"
    
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        output_file = f"{base_name}_{fp8_type_str}_scaled_ultra{distill_str}{size_str}_fixed.safetensors"
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
        'min_size_threshold': args.min_tensor_size,
    }

    convert_to_fp8_scaled(
        args.input,
        output_file,
        args.t5xxl,
        args.keep_distillation,
        args.calib_samples,
        args.use_shared_scales,
        min_tensor_size,
        **converter_kwargs
    )

if __name__ == "__main__":
    main()
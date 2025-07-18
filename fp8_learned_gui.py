import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import queue
import subprocess
from pathlib import Path

class FP8QuantizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FP8 Quantization Tool with Learned Rounding")
        self.root.geometry("800x900")
        
        # Queue for thread communication
        self.output_queue = queue.Queue()
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.keep_distillation = tk.BooleanVar()
        self.t5xxl = tk.BooleanVar()
        self.calib_samples = tk.IntVar(value=256)
        self.num_iter = tk.IntVar(value=512)
        self.lr = tk.DoubleVar(value=0.01)
        self.reg_lambda = tk.DoubleVar(value=0.01)
        self.beta_start = tk.DoubleVar(value=20.0)
        self.beta_end = tk.DoubleVar(value=2.0)
        self.optimizer_choice = tk.StringVar(value="RMSprop")
        
        # Progress tracking
        self.total_weights = 0
        self.processed_weights = 0
        self.current_weight = ""
        self.current_iteration = 0
        self.max_iterations = 0
        
        self.setup_ui()
        
        # Start checking for output updates
        self.check_queue()
        
    def setup_ui(self):
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Main tab
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Main Settings")
        
        # Advanced tab
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced Settings")
        
        # Output tab
        output_frame = ttk.Frame(notebook)
        notebook.add(output_frame, text="Output")
        
        self.setup_main_tab(main_frame)
        self.setup_advanced_tab(advanced_frame)
        self.setup_output_tab(output_frame)
        
    def setup_main_tab(self, parent):
        # File selection section
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding="10")
        file_frame.pack(fill="x", pady=(0, 10))
        
        # Input file
        ttk.Label(file_frame, text="Input File:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        input_frame = ttk.Frame(file_frame)
        input_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(input_frame, textvariable=self.input_file).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(input_frame, text="Browse", command=self.browse_input).grid(row=0, column=1)
        
        # Output file
        ttk.Label(file_frame, text="Output File (leave empty for auto-generation):").grid(row=2, column=0, sticky="w", pady=(0, 5))
        output_frame = ttk.Frame(file_frame)
        output_frame.grid(row=3, column=0, sticky="ew")
        output_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(output_frame, textvariable=self.output_file).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(output_frame, text="Browse", command=self.browse_output).grid(row=0, column=1)
        
        file_frame.columnconfigure(0, weight=1)
        
        # Model options
        options_frame = ttk.LabelFrame(parent, text="Model Options", padding="10")
        options_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Checkbutton(options_frame, text="T5XXL Model (exclude norm, bias, embed_tokens, shared layers)", 
                       variable=self.t5xxl).pack(anchor="w")
        ttk.Checkbutton(options_frame, text="Keep Distillation Layers (exclude distillation layers from quantization)", 
                       variable=self.keep_distillation).pack(anchor="w")
        
        # Basic parameters
        basic_frame = ttk.LabelFrame(parent, text="Basic Parameters", padding="10")
        basic_frame.pack(fill="x", pady=(0, 10))
        
        # Calibration samples
        ttk.Label(basic_frame, text="Calibration Samples:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        calib_spin = ttk.Spinbox(basic_frame, from_=32, to=2048, textvariable=self.calib_samples, width=10)
        calib_spin.grid(row=0, column=1, sticky="w")
        ttk.Label(basic_frame, text="(Number of random samples for calibration)").grid(row=0, column=2, sticky="w", padx=(10, 0))
        
        # Iterations
        ttk.Label(basic_frame, text="Optimization Iterations:").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(5, 0))
        iter_spin = ttk.Spinbox(basic_frame, from_=100, to=2000, textvariable=self.num_iter, width=10)
        iter_spin.grid(row=1, column=1, sticky="w", pady=(5, 0))
        ttk.Label(basic_frame, text="(Number of optimization steps per tensor)").grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        
        # Learning rate
        ttk.Label(basic_frame, text="Learning Rate:").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=(5, 0))
        lr_entry = ttk.Entry(basic_frame, textvariable=self.lr, width=10)
        lr_entry.grid(row=2, column=1, sticky="w", pady=(5, 0))
        ttk.Label(basic_frame, text="(Optimizer learning rate)").grid(row=2, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        
        # Optimizer selection
        optimizer_frame = ttk.LabelFrame(parent, text="Optimizer Selection", padding="10")
        optimizer_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(optimizer_frame, text="Optimizer:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        optimizer_combo = ttk.Combobox(optimizer_frame, textvariable=self.optimizer_choice, 
                                     values=["RMSprop", "Adam", "AdamW", "SGD", "GOODDOG"], 
                                     state="readonly", width=15)
        optimizer_combo.grid(row=0, column=1, sticky="w")
        
        # Optimizer info
        optimizer_info = ttk.Label(optimizer_frame, text="", foreground="blue")
        optimizer_info.grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))
        
        def update_optimizer_info(*args):
            info_text = {
                "RMSprop": "Good default choice, stable convergence",
                "Adam": "Adaptive learning rate, good for most cases",
                "AdamW": "Adam with weight decay, often better than Adam",
                "SGD": "Simple but may need learning rate tuning",
                "GOODDOG": "Custom optimizer (requires GOODDOG package)"
            }
            optimizer_info.config(text=info_text.get(self.optimizer_choice.get(), ""))
        
        self.optimizer_choice.trace("w", update_optimizer_info)
        update_optimizer_info()
        
        # Control buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Button(button_frame, text="Start Conversion", command=self.start_conversion, 
                  style="Accent.TButton").pack(side="left", padx=(0, 10))
        
        # Progress section
        progress_frame = ttk.Frame(button_frame)
        progress_frame.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode="determinate")
        self.progress_bar.pack(fill="x", pady=(0, 2))
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack(anchor="w")
        
        ttk.Button(button_frame, text="Clear Output", command=self.clear_output).pack(side="right")
        
    def setup_advanced_tab(self, parent):
        # Advanced parameters
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Learned Rounding Parameters", padding="10")
        advanced_frame.pack(fill="x", pady=(0, 10))
        
        # Regularization lambda
        ttk.Label(advanced_frame, text="Regularization Lambda:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        reg_entry = ttk.Entry(advanced_frame, textvariable=self.reg_lambda, width=10)
        reg_entry.grid(row=0, column=1, sticky="w")
        ttk.Label(advanced_frame, text="(Regularization strength for rounding loss)").grid(row=0, column=2, sticky="w", padx=(10, 0))
        
        # Beta start
        ttk.Label(advanced_frame, text="Beta Start:").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(5, 0))
        beta_start_entry = ttk.Entry(advanced_frame, textvariable=self.beta_start, width=10)
        beta_start_entry.grid(row=1, column=1, sticky="w", pady=(5, 0))
        ttk.Label(advanced_frame, text="(Starting value for beta annealing)").grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        
        # Beta end
        ttk.Label(advanced_frame, text="Beta End:").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=(5, 0))
        beta_end_entry = ttk.Entry(advanced_frame, textvariable=self.beta_end, width=10)
        beta_end_entry.grid(row=2, column=1, sticky="w", pady=(5, 0))
        ttk.Label(advanced_frame, text="(Ending value for beta annealing)").grid(row=2, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        
        # Info section
        info_frame = ttk.LabelFrame(parent, text="Parameter Information", padding="10")
        info_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        info_text = """
Advanced Parameters Explained:

• Regularization Lambda: Controls how much the optimizer prefers values close to 0 or 1 in the rounding mask. Higher values lead to more binary (0/1) rounding decisions.

• Beta Start/End: Controls the annealing schedule for the regularization term. The beta value decreases from start to end over the optimization iterations, allowing for more exploration early and more exploitation later.

• Calibration Samples: Number of random input samples used to evaluate the reconstruction loss. More samples = more accurate but slower.

• Optimization Iterations: Number of gradient descent steps per tensor. More iterations = potentially better quantization but slower conversion.

Optimizer Notes:
• RMSprop: Good default, handles noisy gradients well
• Adam/AdamW: Adaptive learning rates, often converge faster
• SGD: Simple but may need careful learning rate tuning
• GOODDOG: Custom optimizer (requires separate installation)

The script uses the AdaRound technique adapted for FP8 quantization, which learns optimal rounding decisions to minimize reconstruction error.
        """
        
        info_label = tk.Text(info_frame, wrap="word", height=15, state="disabled")
        info_label.pack(fill="both", expand=True)
        info_label.config(state="normal")
        info_label.insert("1.0", info_text.strip())
        info_label.config(state="disabled")
        
    def setup_output_tab(self, parent):
        # Output text area
        self.output_text = scrolledtext.ScrolledText(parent, wrap="word", height=30)
        self.output_text.pack(fill="both", expand=True, padx=10, pady=10)
        
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input Safetensors File",
            filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Select Output File",
            defaultextension=".safetensors",
            filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")]
        )
        if filename:
            self.output_file.set(filename)
            
    def clear_output(self):
        self.output_text.delete("1.0", tk.END)
        
    def log_output(self, message):
        self.output_queue.put(message)
        
    def check_queue(self):
        try:
            while True:
                message = self.output_queue.get_nowait()
                self.output_text.insert(tk.END, message + "\n")
                self.output_text.see(tk.END)
                self.parse_progress_message(message)
                self.root.update_idletasks()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_queue)
    
    def parse_progress_message(self, message):
        """Parse console output to extract progress information"""
        try:
            # Look for weight processing messages
            if "weight tensors to potentially process" in message:
                # Extract total number of weights
                import re
                match = re.search(r'Found (\d+) weight tensors', message)
                if match:
                    self.total_weights = int(match.group(1))
                    self.processed_weights = 0
                    self.update_progress()
            
            elif "Processing tensor:" in message:
                # Extract current weight being processed
                import re
                match = re.search(r'\((\d+)/(\d+)\) Processing tensor: (.+)', message)
                if match:
                    current_num = int(match.group(1))
                    total_num = int(match.group(2))
                    weight_name = match.group(3)
                    self.processed_weights = current_num - 1  # -1 because we're starting to process
                    self.current_weight = weight_name
                    self.total_weights = total_num
                    self.update_progress()
            
            elif "Skipping excluded" in message or "Skipping empty" in message:
                # Count skipped weights as processed
                import re
                match = re.search(r'\((\d+)/(\d+)\)', message)
                if match:
                    current_num = int(match.group(1))
                    self.processed_weights = current_num
                    self.update_progress()
            
            elif "Optimizing rounding" in message:
                # Extract optimization progress
                import re
                match = re.search(r'(\d+)%', message)
                if match:
                    opt_progress = int(match.group(1))
                    # Update sub-progress for current weight
                    self.update_progress(opt_progress)
            
            elif "Conversion complete!" in message:
                self.progress_bar.config(value=100)
                self.progress_label.config(text="Conversion Complete!")
                
        except Exception as e:
            # Don't let progress parsing errors crash the GUI
            pass
    
    def update_progress(self, optimization_progress=0):
        """Update the progress bar and label"""
        if self.total_weights > 0:
            # Calculate overall progress
            base_progress = (self.processed_weights / self.total_weights) * 100
            
            # Add sub-progress for current weight optimization
            if optimization_progress > 0 and self.processed_weights < self.total_weights:
                weight_contribution = (1 / self.total_weights) * 100
                sub_progress = (optimization_progress / 100) * weight_contribution
                total_progress = base_progress + sub_progress
            else:
                total_progress = base_progress
            
            self.progress_bar.config(value=min(total_progress, 100))
            
            # Update label
            if self.processed_weights < self.total_weights:
                if optimization_progress > 0:
                    label_text = f"Processing {self.processed_weights + 1}/{self.total_weights}: {self.current_weight} ({optimization_progress}%)"
                else:
                    label_text = f"Processing {self.processed_weights + 1}/{self.total_weights}: {self.current_weight}"
            else:
                label_text = f"Completed {self.processed_weights}/{self.total_weights} weights"
            
            self.progress_label.config(text=label_text)
        else:
            self.progress_label.config(text="Initializing...")
            
    def generate_modified_script(self):
        """Generate a modified version of the script with the selected optimizer"""
        script_content = f'''import argparse
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple
from tqdm import tqdm
import gc

# Keys containing these strings will not be quantized if --t5xxl is set
AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", "shared"]
# Target FP8 format
TARGET_FP8_DTYPE = torch.float8_e4m3fn
# Intermediate dtype for calculations
COMPUTE_DTYPE = torch.float32
# Dtype for storing scale factors
SCALE_DTYPE = torch.float32

class LearnedRoundingConverter:
    def __init__(self, num_iter=500, lr=1e-3, reg_lambda=0.01, beta_start=20, beta_end=2, optimizer_name="RMSprop"):
        self.num_iter = num_iter
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.optimizer_name = optimizer_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max
        print(f"LearnedRoundingConverter initialized on device: {{self.device}}")
        print(f"Using optimizer: {{self.optimizer_name}}")

    def get_optimizer(self, parameters):
        if self.optimizer_name == "RMSprop":
            return torch.optim.RMSprop(parameters, lr=self.lr)
        elif self.optimizer_name == "Adam":
            return torch.optim.Adam(parameters, lr=self.lr)
        elif self.optimizer_name == "AdamW":
            return torch.optim.AdamW(parameters, lr=self.lr)
        elif self.optimizer_name == "SGD":
            return torch.optim.SGD(parameters, lr=self.lr, momentum=0.9)
        elif self.optimizer_name == "GOODDOG":
            try:
                from GOODDOG import GOODDOG
                return GOODDOG(parameters, lr=self.lr, adaptive_muon=False, invariant=True)
            except ImportError:
                print("GOODDOG optimizer not found, falling back to RMSprop")
                return torch.optim.RMSprop(parameters, lr=self.lr)
        else:
            return torch.optim.RMSprop(parameters, lr=self.lr)

    def convert(self, W_orig: torch.Tensor, X_calib: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)
        X_calib = X_calib.to(self.device, dtype=COMPUTE_DTYPE)

        w_max = W_float32.abs().max()
        if w_max < 1e-12:
            print("  - Tensor is all zeros, skipping optimization.")
            scale = torch.tensor(1.0, device=self.device)
            quantized_tensor = torch.zeros_like(W_float32, dtype=TARGET_FP8_DTYPE)
            return quantized_tensor.cpu(), scale.reciprocal().cpu().reshape(1)

        scale = self.f8_max_val / w_max
        W_scaled = W_float32 * scale

        h_init = W_scaled - (torch.floor(W_scaled / FP8_MIN_POS) * FP8_MIN_POS)
        h = torch.nn.Parameter(h_init)

        optimizer = self.get_optimizer([h])
        beta_schedule = torch.linspace(self.beta_start, self.beta_end, self.num_iter).to(self.device)

        pbar = tqdm(range(self.num_iter), desc="    Optimizing rounding", leave=False)
        for i in pbar:
            beta = beta_schedule[i]
            W_soft_quant = ((torch.floor(W_scaled / FP8_MIN_POS) * FP8_MIN_POS) + h) / scale
            Y_orig = X_calib @ W_float32.T
            Y_quant = X_calib @ W_soft_quant.T
            recon_loss = (Y_orig - Y_quant).pow(2).mean()
            reg_loss = self.reg_lambda * torch.sum(1 - torch.abs(2 * h/FP8_MIN_POS - 1).pow(beta))
            total_loss = recon_loss + reg_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                h.clamp_(0, FP8_MIN_POS)
            
            # Print progress every 10% for GUI parsing
            if i % max(1, self.num_iter // 10) == 0:
                progress_pct = int((i / self.num_iter) * 100)
                print(f"    Optimizing rounding: {{progress_pct}}% - Recon: {{recon_loss.item():.4e}}, Reg: {{reg_loss.item():.4e}}")
            
            pbar.set_postfix({{"Recon Loss": f"{{recon_loss.item():.4e}}", "Reg Loss": f"{{reg_loss.item():.4e}}"}})
            if reg_loss.item() < 1e-8:
                print(f"    Optimizing rounding: 100% - Early stopping (reg_loss < 1e-8)")
                break

        with torch.no_grad():
            W_quant_final_scaled = (torch.floor(W_scaled / FP8_MIN_POS) * FP8_MIN_POS) + h.data
            W_f8 = W_quant_final_scaled.to(dtype=TARGET_FP8_DTYPE)

        dequant_scale = scale.reciprocal().reshape(1)
        del W_float32, X_calib, h, optimizer, Y_orig, Y_quant, W_soft_quant, W_scaled
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return W_f8.cpu(), dequant_scale.cpu()

def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    finfo = torch.finfo(fp8_dtype)
    return float(finfo.min), float(finfo.max), float(finfo.tiny)

FP8_MIN, FP8_MAX, FP8_MIN_POS = get_fp8_constants(TARGET_FP8_DTYPE)

def convert_to_fp8_scaled(input_file: str, output_file: str, t5xxl: bool, keep_distillation: bool, 
                         calib_samples: int, **converter_kwargs):
    print(f"Processing: {{input_file}}")
    print(f"Output will be saved to: {{output_file}}")
    print(f"Using FP8 format: {{TARGET_FP8_DTYPE}}")
    print(f"FP8 Range: [{{FP8_MIN}}, {{FP8_MAX}}]")
    print(f"FP8 Min Precision: [{{FP8_MIN_POS}}]")

    tensors: Dict[str, torch.Tensor] = {{}}
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).cpu()
    except Exception as e:
        print(f"Error loading '{{input_file}}': {{e}}")
        return

    converter = LearnedRoundingConverter(**converter_kwargs)

    print("\\nScanning model for linear layer dimensions...")
    calibration_data_cache = {{}}
    for key, tensor in tensors.items():
        if key.endswith('.weight') and tensor.ndim == 2:
            in_features = tensor.shape[1]
            if in_features not in calibration_data_cache:
                print(f"  - Found new in_features dimension: {{in_features}}. Generating calibration data.")
                calibration_data_cache[in_features] = torch.randn(
                    calib_samples, in_features, dtype=torch.bfloat16
                )
    print("Calibration data generated.\\n")

    new_tensors: Dict[str, torch.Tensor] = {{}}
    weight_keys = sorted([key for key in tensors.keys() if key.endswith('.weight')])
    total_weights = len(weight_keys)
    skipped_count = 0
    processed_count = 0

    print(f"Found {{total_weights}} weight tensors to potentially process.")

    for i, key in enumerate(weight_keys):
        process_this_key = True
        if t5xxl and any(avoid_name in key for avoid_name in AVOID_KEY_NAMES):
            print(f"({{i+1}}/{{total_weights}}) Skipping excluded T5XXL tensor: {{key}}")
            new_tensors[key] = tensors[key]
            process_this_key = False
            skipped_count += 1

        if keep_distillation and any(avoid_name in key for avoid_name in ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]):
            print(f"({{i+1}}/{{total_weights}}) Skipping excluded distillation tensor: {{key}}")
            new_tensors[key] = tensors[key]
            base_name = key[:-len('.weight')]
            scale_weight_key = f"{{base_name}}.scale_weight"
            new_tensors[scale_weight_key] = torch.tensor([1.0], dtype=SCALE_DTYPE)
            process_this_key = False
            skipped_count += 1

        if not process_this_key:
            continue

        print(f"({{i+1}}/{{total_weights}}) Processing tensor: {{key}}")
        processed_count += 1

        original_tensor = tensors[key]

        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            print(f"  - Skipping empty or non-2D tensor: {{key}}")
            new_tensors[key] = tensors[key].to(TARGET_FP8_DTYPE)
            base_name = key[:-len('.weight')]
            scale_weight_key = f"{{base_name}}.scale_weight"
            new_tensors[scale_weight_key] = torch.tensor([1.0], dtype=SCALE_DTYPE)
            continue

        in_features = original_tensor.shape[1]
        if in_features not in calibration_data_cache:
             print(f"  - WARNING: No calibration data found for in_features={{in_features}}. Skipping {{key}}")
             new_tensors[key] = original_tensor
             skipped_count += 1
             processed_count -= 1
             continue

        calibration_data = calibration_data_cache[in_features]
        quantized_fp8_tensor, dequant_scale = converter.convert(original_tensor, calibration_data)

        new_tensors[key] = quantized_fp8_tensor
        base_name = key[:-len('.weight')]
        scale_weight_key = f"{{base_name}}.scale_weight"
        new_tensors[scale_weight_key] = dequant_scale.to(SCALE_DTYPE)
        if t5xxl:
            scale_input_key = f"{{base_name}}.scale_input"
            new_tensors[scale_input_key] = dequant_scale.detach().clone().to(SCALE_DTYPE)

        print(f"  - Dequant Scale  : {{dequant_scale.item():.9}}")
        print(f"  - Weight  : {{quantized_fp8_tensor}}")

    for key, tensor in tensors.items():
        if key not in new_tensors:
            new_tensors[key] = tensor
            print(f"(+) Adding original non-quantized tensor: {{key}}")

    new_tensors["scaled_fp8"] = torch.empty((2), dtype=TARGET_FP8_DTYPE) if not t5xxl else torch.empty((0), dtype=TARGET_FP8_DTYPE)

    print("-" * 40)
    print(f"Saving {{len(new_tensors)}} tensors to {{output_file}}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_file(new_tensors, output_file)
        print("Conversion complete!")
    except Exception as e:
        print(f"Error saving file '{{output_file}}': {{e}}")
        return

    print("-" * 40)
    print("Summary:")
    print(f"  - Original tensor count : {{len(tensors)}}")
    print(f"  - Weights processed     : {{processed_count}}")
    print(f"  - Weights skipped       : {{skipped_count}}")
    print(f"  - Final tensor count    : {{len(new_tensors)}}")
    print("-" * 40)

if __name__ == "__main__":
    import sys
    # Parse command line arguments passed from GUI
    args = sys.argv[1:]
    
    input_file = None
    output_file = None
    t5xxl = False
    keep_distillation = False
    calib_samples = 256
    num_iter = 512
    lr = 1e-2
    reg_lambda = 0.01
    beta_start = 20.0
    beta_end = 2.0
    optimizer_name = "RMSprop"
    
    i = 0
    while i < len(args):
        if args[i] == '--input' and i + 1 < len(args):
            input_file = args[i + 1]
            i += 2
        elif args[i] == '--output' and i + 1 < len(args):
            output_file = args[i + 1]
            i += 2
        elif args[i] == '--t5xxl':
            t5xxl = True
            i += 1
        elif args[i] == '--keep_distillation':
            keep_distillation = True
            i += 1
        elif args[i] == '--calib_samples' and i + 1 < len(args):
            calib_samples = int(args[i + 1])
            i += 2
        elif args[i] == '--num_iter' and i + 1 < len(args):
            num_iter = int(args[i + 1])
            i += 2
        elif args[i] == '--lr' and i + 1 < len(args):
            lr = float(args[i + 1])
            i += 2
        elif args[i] == '--reg_lambda' and i + 1 < len(args):
            reg_lambda = float(args[i + 1])
            i += 2
        elif args[i] == '--beta_start' and i + 1 < len(args):
            beta_start = float(args[i + 1])
            i += 2
        elif args[i] == '--beta_end' and i + 1 < len(args):
            beta_end = float(args[i + 1])
            i += 2
        elif args[i] == '--optimizer' and i + 1 < len(args):
            optimizer_name = args[i + 1]
            i += 2
        else:
            i += 1
    
    if not input_file:
        print("Error: Input file is required")
        sys.exit(1)
    
    if not output_file:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{{base_name}}_float8_e4m3fn_scaled_learned.safetensors"
    
    converter_kwargs = {{
        'num_iter': num_iter,
        'lr': lr,
        'reg_lambda': reg_lambda,
        'beta_start': beta_start,
        'beta_end': beta_end,
        'optimizer_name': optimizer_name
    }}
    
    convert_to_fp8_scaled(input_file, output_file, t5xxl, keep_distillation, calib_samples, **converter_kwargs)
'''
        
        return script_content
        
    def start_conversion(self):
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input file")
            return
            
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Input file does not exist")
            return
        
        # Reset progress tracking
        self.total_weights = 0
        self.processed_weights = 0
        self.current_weight = ""
        self.progress_bar.config(value=0)
        self.progress_label.config(text="Starting conversion...")
        
        # Clear output
        self.clear_output()
        
        # Start conversion in a separate thread
        thread = threading.Thread(target=self.run_conversion)
        thread.daemon = True
        thread.start()
        
    def run_conversion(self):
        try:
            # Generate the modified script
            script_content = self.generate_modified_script()
            
            # Write to temporary file
            temp_script = "temp_fp8_script.py"
            with open(temp_script, 'w') as f:
                f.write(script_content)
            
            # Build command line arguments
            cmd = [sys.executable, temp_script]
            cmd.extend(['--input', self.input_file.get()])
            
            if self.output_file.get():
                cmd.extend(['--output', self.output_file.get()])
            
            if self.t5xxl.get():
                cmd.append('--t5xxl')
            
            if self.keep_distillation.get():
                cmd.append('--keep_distillation')
            
            cmd.extend(['--calib_samples', str(self.calib_samples.get())])
            cmd.extend(['--num_iter', str(self.num_iter.get())])
            cmd.extend(['--lr', str(self.lr.get())])
            cmd.extend(['--reg_lambda', str(self.reg_lambda.get())])
            cmd.extend(['--beta_start', str(self.beta_start.get())])
            cmd.extend(['--beta_end', str(self.beta_end.get())])
            cmd.extend(['--optimizer', self.optimizer_choice.get()])
            
            self.log_output(f"Starting conversion with command: {' '.join(cmd)}")
            self.log_output("=" * 60)
            
            # Run the conversion
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output in real-time
            for line in process.stdout:
                self.log_output(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                self.log_output("=" * 60)
                self.log_output("Conversion completed successfully!")
            else:
                self.log_output("=" * 60)
                self.log_output(f"Conversion failed with return code: {process.returncode}")
                
        except Exception as e:
            self.log_output(f"Error during conversion: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_script):
                    os.remove(temp_script)
            except:
                pass
            
            # Reset progress on completion
            self.root.after(0, lambda: self.progress_label.config(text="Ready"))

def main():
    root = tk.Tk()
    app = FP8QuantizationGUI(root)
    photo = tk.PhotoImage(file = '48x48_trans.png')
    root.wm_iconphoto(False, photo)
    root.mainloop()

if __name__ == "__main__":
    main()
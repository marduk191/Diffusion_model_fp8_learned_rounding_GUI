# FP8 Quantization Tool with Learned Rounding

A user-friendly GUI application for converting PyTorch models to FP8 (8-bit floating-point) format using learned rounding techniques. This tool implements the AdaRound algorithm adapted for FP8 quantization to minimize reconstruction error and maintain model quality.

## Features

- **Easy-to-use GUI**: Intuitive interface with tabbed layout for basic and advanced settings
- **Learned Rounding**: Implements AdaRound technique for optimal quantization with minimal accuracy loss
- **Multiple Optimizers**: Support for RMSprop, Adam, AdamW, SGD, and GOODDOG optimizers
- **Real-time Progress**: Live progress tracking with detailed status updates
- **Model-specific Options**: Special handling for T5XXL and distillation models
- **Calibration-based**: Uses random calibration samples for accurate quantization
- **FP8 Format**: Targets `torch.float8_e4m3fn` format for efficient inference

## Requirements
(If you run this in your comfyui venv, you can skip this)
```bash
pip install torch torchvision torchaudio
pip install safetensors
pip install tqdm
pip install tkinter  # Usually included with Python
```

### Optional Dependencies

- **GOODDOG Optimizer**: Install separately if using the GOODDOG optimizer option
- **CUDA**: Recommended for GPU acceleration during quantization

## Installation

1. Clone or download the repository
2. Install the required dependencies
3. Run the application:

```
activate your venv before starting
python fp8_learned_gui.py
```

## Usage

### Basic Usage

1. **Select Input File**: Choose your `.safetensors` model file
2. **Set Output File**: (Optional) Specify output location, or leave blank for auto-generation
3. **Configure Basic Parameters**:
   - **Calibration Samples**: Number of random samples for calibration (default: 256)
   - **Optimization Iterations**: Number of optimization steps per tensor (default: 512)
   - **Learning Rate**: Optimizer learning rate (default: 0.01)
   - **Optimizer**: Choose optimization algorithm (default: RMSprop)
4. **Start Conversion**: Click "Start Conversion" to begin the process

### Advanced Settings

Access the "Advanced Settings" tab for fine-tuning:

- **Regularization Lambda**: Controls rounding preference (default: 0.01)
- **Beta Start/End**: Annealing schedule parameters (default: 20.0 → 2.0)
- **Model-specific Options**:
  - **T5XXL Model**: Excludes norm, bias, embed_tokens, and shared layers
  - **Keep Distillation**: Excludes distillation layers from quantization

### Model-Specific Options

#### T5XXL Models
Enable the "T5XXL Model" option to:
- Skip quantization of normalization layers, bias terms, embedding tokens, and shared layers
- Add additional scale tensors for proper inference

#### Distillation Models
Enable "Keep Distillation Layers" to:
- Preserve distillation-specific layers in original precision
- Maintain model compatibility with distillation frameworks

## Output Format

The tool generates quantized models with:
- **FP8 Weights**: Quantized to `torch.float8_e4m3fn` format
- **Scale Tensors**: Per-tensor dequantization scales stored as `{layer_name}.scale_weight`
- **Metadata**: `scaled_fp8` tensor indicating the quantization format

## Algorithm Details

### Learned Rounding (AdaRound)

The tool implements a learned rounding approach that:

1. **Calibration**: Uses random input samples to evaluate reconstruction error
2. **Optimization**: Learns optimal rounding decisions through gradient descent
3. **Regularization**: Encourages binary (0/1) rounding decisions
4. **Annealing**: Gradually reduces exploration for better convergence

### Mathematical Formulation

The optimization minimizes:
```
Loss = Reconstruction_Loss + λ * Regularization_Loss
```

Where:
- **Reconstruction Loss**: `||Y_original - Y_quantized||²`
- **Regularization Loss**: Encourages rounding values near 0 or 1
- **λ**: Regularization strength (configurable)

## Performance Considerations

- **GPU Acceleration**: Automatically uses CUDA if available
- **Memory Management**: Includes garbage collection and cache clearing
- **Batch Processing**: Processes tensors individually to manage memory
- **Progress Tracking**: Real-time updates without blocking the GUI

## Optimizer Comparison

| Optimizer | Characteristics | Best For |
|-----------|----------------|----------|
| **RMSprop** | Stable, handles noisy gradients well | General use (recommended) |
| **Adam** | Adaptive learning rates, fast convergence | Most quantization tasks |
| **AdamW** | Adam with weight decay, often superior | Complex models |
| **SGD** | Simple, may need learning rate tuning | Fine-tuning scenarios |
| **GOODDOG** | Custom optimizer (requires separate install) | Research applications |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce calibration samples or process on CPU
2. **Slow Conversion**: Decrease optimization iterations or use fewer calibration samples
3. **Poor Quality**: Increase iterations, adjust learning rate, or try different optimizers

### Error Messages

- **"Input file is required"**: Select a valid `.safetensors` file
- **"GOODDOG optimizer not found"**: Install GOODDOG package or use alternative optimizer
- **"No calibration data found"**: Model structure may be incompatible

## File Structure

```
fp8_learned_gui.py          # Main GUI application
├── GUI Components
│   ├── File selection
│   ├── Parameter configuration
│   ├── Progress tracking
│   └── Output logging
├── Conversion Engine
│   ├── LearnedRoundingConverter
│   ├── Optimizer selection
│   └── FP8 quantization logic
└── Utility Functions
    ├── Progress parsing
    ├── Script generation
    └── Error handling
```

## Contributing

Contributions are welcome! Please consider:

- **Bug Reports**: Include system info, error messages, and reproduction steps
- **Feature Requests**: Describe use cases and expected behavior
- **Code Contributions**: Follow existing code style and include tests

## License

This project is open-source. Please check the repository for specific license terms.

## Acknowledgments

- Based on the AdaRound algorithm for learned quantization
- Inspired by research in neural network quantization
- Built with PyTorch and Tkinter for cross-platform compatibility
- Based on the work here https://github.com/Clybius/Learned-Rounding

<img width="790" height="922" alt="Screenshot 2025-07-17 115851" src="https://github.com/user-attachments/assets/7f9679e8-6189-4f4b-8479-eb6255ecd1e8" />
<img width="797" height="926" alt="Screenshot 2025-07-17 115757" src="https://github.com/user-attachments/assets/89f7a47a-5cd8-4b84-8876-608e180a4970" />
<img width="1030" height="859" alt="image" src="https://github.com/user-attachments/assets/bb41f492-9ba1-4a64-b252-fa071762721a" />


Based on the work here https://github.com/Clybius/Learned-Rounding

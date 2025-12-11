# Model Architecture Details

This document provides detailed descriptions of all model architectures, input/output formats, working principles, and their relationships to weather prediction tasks.

**[中文](MODEL.md) | [English](MODEL_EN.md)**

## 📋 Table of Contents

1. [Task Overview](#task-overview)
2. [Traditional Deep Learning Models](#traditional-deep-learning-models)
   - [Linear Regression](#linear-regression)
   - [LSTM](#lstm)
   - [Transformer (Sequence Modeling)](#transformer-sequence-modeling)
   - [CNN](#cnn)
   - [ConvLSTM](#convlstm)
   - [Weather Transformer](#weather-transformer)
3. [WeatherDiff Module](#weatherdiff-module)
   - [VAE (Stable Diffusion VAE)](#stable-diffusion-vae)
   - [RAE (Representation Autoencoder)](#rae-representation-autoencoder)
   - [Pixel U-Net](#pixel-u-net)
   - [Latent U-Net](#latent-u-net)
   - [Diffusion Model](#diffusion-model)
4. [Model Comparison and Selection](#model-comparison-and-selection)

---

## Task Overview

### Prediction Task Definition

**Goal**: Predict future 4 time steps (1 day) of weather conditions based on past 12 time steps (3 days, 6-hour intervals) of weather data.

**Data Format**:
- **Input**: `(batch, input_length=12, channels, H=32, W=64)`
  - `input_length=12`: Past 3 days of data (12 six-hour intervals)
  - `channels`: Number of meteorological variables (e.g., temperature, pressure, etc.)
  - `H=32, W=64`: Spatial resolution (latitude × longitude, equiangular grid)
  
- **Output**: `(batch, output_length=4, channels, H=32, W=64)`
  - `output_length=4`: Future 1 day prediction (4 six-hour intervals)

**Data Normalization**:
- **Traditional Models**: Z-score normalization (mean 0, std 1)
- **WeatherDiff Module**: MinMax normalization to [-1, 1] (compatible with Stable Diffusion VAE)

### Core Challenges

1. **Spatiotemporal Dependencies**: Weather systems have complex spatiotemporal correlations
2. **Multi-scale Features**: From local convection to global circulation patterns
3. **Uncertainty**: Weather systems have inherent chaotic properties
4. **Computational Efficiency**: Global 64×32 grid requires efficient processing

---

## Traditional Deep Learning Models

### Linear Regression

#### Model Structure

```python
# Single-variable version (lr)
X: (batch, 12, features) → flatten → (batch, 12*features)
y: (batch, 4, features) → flatten → (batch, 4*features)
Ridge(alpha=1.0).fit(X_flat, y_flat)

# Multi-variable version (lr_multi)
# Each variable trains an independent Ridge model
for var in variables:
    model[var] = Ridge(alpha=10.0)
```

#### Input/Output

- **Input**: `(batch, 12, features)` - flattened to `(batch, 12*features)`
- **Output**: `(batch, 4, features)` - flattened to `(batch, 4*features)`
- **Parameters**: L2 regularization coefficient `alpha`

#### Working Principle

- **Linear Mapping**: Directly learns linear transformation from input to output
- **No Temporal Modeling**: Flattens time series, losing temporal order information
- **No Spatial Modeling**: Flattens spatial grid, losing spatial structure

#### Relationship to Task

- **Advantages**: 
  - Extremely fast training, suitable for quick baseline testing
  - Few parameters, less prone to overfitting
- **Limitations**: 
  - Cannot model nonlinear spatiotemporal dependencies
  - Low prediction accuracy, only serves as baseline

#### Use Cases

- ✅ Quick data flow validation
- ✅ Simple single-variable prediction
- ❌ Not suitable for practical applications

#### Extension: Multi-Output Linear Regression

`src/models/linear_regression.py` also implements `MultiOutputLinearRegression`:

**Model Structure**:
```python
# Each variable trains an independent Ridge model
for var in variables:
    X_flat = (n_samples, input_length * total_features)  # Features of all variables
    y_var = (n_samples, output_length * grid_points_per_var)  # Only grid points of this variable
    model[var] = Ridge(alpha=10.0).fit(X_scaled, y_scaled)
```

**Features**:
- **One model per variable**: e.g., one model for 2m_temperature, one for geopotential
- **Shared input features**: All variables use the same input features (containing historical data of all variables)
- **Independent prediction**: Each model only predicts all grid points of the corresponding variable
- **Regularization**: Uses larger alpha (10.0 vs 1.0) to prevent overfitting

**Difference from Standard LR**:
- Standard LR: All variables share one model, predicting all grid points of all variables
- Multi-output LR: Independent model per variable, but input contains information from all variables

**Use Cases**:
- ✅ Multi-variable prediction (each variable has independent physical properties)
- ✅ Scenarios requiring variable-specific regularization

---

### LSTM

#### Model Structure

```
Input: (batch, 12, features)
  ↓
LSTM layers (hidden_size=128, num_layers=2)
  ↓
Take last timestep hidden state: (batch, hidden_size)
  ↓
Fully connected layer: (batch, hidden_size) → (batch, 4*features)
  ↓
Reshape: (batch, 4, features)
```

#### Input/Output

- **Input**: `(batch, input_length=12, input_size)`
  - `input_size`: Flattened feature count (`channels * H * W`)
- **Output**: `(batch, output_length=4, input_size)`
- **Key Parameters**:
  - `hidden_size`: LSTM hidden dimension (default 128)
  - `num_layers`: Number of LSTM layers (default 2)
  - `dropout`: Dropout rate (default 0.2)

#### Working Principle

1. **Temporal Modeling**: LSTM models temporal dependencies through gating mechanisms (forget gate, input gate, output gate)
2. **Memory Mechanism**: Cell state stores long-term memory
3. **Sequence Processing**: Processes timestep by timestep, preserving temporal order information
4. **Spatial Information Loss**: Spatial dimensions are flattened at input, cannot utilize spatial structure

#### Relationship to Task

- **Advantages**:
  - Effectively models temporal dependencies
  - Suitable for single-point time series prediction
- **Limitations**:
  - Loses spatial structure (flattening operation)
  - Cannot capture spatial correlations (e.g., similarity of adjacent grids)
  - Large parameter count (fully connected layers)

#### Use Cases

- ✅ Single-point prediction (e.g., single weather station)
- ✅ Cases where features are already extracted/dimensionally reduced
- ❌ Not suitable for grid prediction requiring spatial information

#### Extensions: Other LSTM Variants

`src/models/lstm.py` also implements other LSTM variants:

1. **BidirectionalLSTM**:
   - Bidirectional LSTM, can utilize both past and future information
   - Output dimension is `2 * hidden_size`
   - Suitable for encoder scenarios, but cannot use future information in prediction tasks

2. **LSTMSeq2Seq**:
   - Encoder-decoder architecture
   - Encoder processes input sequence, decoder autoregressively generates output
   - More suitable for sequence-to-sequence prediction tasks

---

### Transformer (Sequence Modeling)

#### Model Structure

Standard Transformer model (`TransformerModel`):

```
Input: (batch, 12, features)
  ↓
Input projection: Linear(features, d_model=128)
  ↓
Positional encoding: PositionalEncoding (sinusoidal encoding)
  ↓
Transformer encoder (3 layers):
  MultiheadAttention (4 heads) → LayerNorm → FFN → LayerNorm
  ↓
Take last timestep: (batch, d_model)
  ↓
Output projection:
  LayerNorm → Dropout → Linear(d_model, d_model//2) → ReLU → 
  Dropout → Linear(d_model//2, features*4)
  ↓
Reshape: (batch, 4, features)
```

#### Input/Output

- **Input**: `(batch, input_length=12, input_size)`
  - `input_size`: Flattened feature count (`channels * H * W`)
- **Output**: `(batch, output_length=4, input_size)`
- **Key Parameters**:
  - `d_model`: Transformer embedding dimension (default 128, optimized to be smaller)
  - `nhead`: Number of attention heads (default 4, optimized to be smaller)
  - `num_layers`: Number of Transformer layers (default 3, optimized to be smaller)
  - `dropout`: Dropout rate (default 0.2, optimized to be larger)

#### Working Principle

1. **Attention Mechanism**: Uses multi-head self-attention to capture dependencies within sequence
2. **Positional Encoding**: Sinusoidal positional encoding provides temporal order information
3. **Sequence Modeling**: Transformer encoder processes entire input sequence
4. **Spatial Information Loss**: Spatial dimensions are flattened at input, cannot utilize spatial structure
5. **Optimized Design**: Reduced parameters for single-variable prediction scenario to prevent overfitting

#### Relationship to Task

- **Advantages**:
  - Effectively models long-range temporal dependencies (attention mechanism)
  - Parallel computation, high training efficiency
  - Optimized parameters, suitable for single-variable prediction
- **Limitations**:
  - ❌ **Loses spatial structure** (flattening operation)
  - ❌ Cannot capture spatial correlations
  - ❌ Computational complexity O(n²), sequence length limited

#### Use Cases

- ✅ Single-point time series prediction
- ✅ Cases where features are already extracted/dimensionally reduced
- ✅ Tasks requiring long-range temporal dependencies
- ❌ Not suitable for grid prediction requiring spatial information

#### Extensions: Other Transformer Variants

`src/models/transformer.py` also implements other Transformer variants:

1. **TransformerSeq2Seq**:
   - Encoder-decoder architecture
   - Encoder processes input sequence, decoder uses learnable query tokens to generate output
   - More suitable for sequence-to-sequence prediction tasks
   - Parameters: `d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4`

2. **SpatialTransformer**:
   - Treats each spatial position as a token
   - Uses Transformer to model relationships between spatial positions
   - Note: For large spatial grids (e.g., 32×64=2048 tokens), computation is heavy
   - May not be used in current implementation

---

### CNN

#### Model Structure

```
Input: (batch, 12, channels, H, W)
  ↓
Flatten temporal dimension: (batch, 12*channels, H, W)
  ↓
Encoder:
  Conv2d(12*channels, 64, k=3, p=1) → BN → ReLU
  Conv2d(64, 128, k=3, p=1) → BN → ReLU
  Conv2d(128, 128, k=3, s=2, p=1) → BN → ReLU  # Downsample: 64×32 → 32×16
  Conv2d(128, 256, k=3, p=1) → BN → ReLU
  ↓
Decoder:
  ConvTranspose2d(256, 128, k=4, s=2, p=1) → BN → ReLU  # Upsample: 32×16 → 64×32
  Conv2d(128, 64, k=3, p=1) → BN → ReLU
  Conv2d(64, 4*channels, k=3, p=1)  # Output
  ↓
Reshape: (batch, 4, channels, H, W)
```

#### Input/Output

- **Input**: `(batch, input_length=12, channels, H=32, W=64)`
- **Output**: `(batch, output_length=4, channels, H=32, W=64)`
- **Key Parameters**:
  - `hidden_channels`: Hidden layer channels (default 64)
  - `input_channels`: Number of input variables

#### Working Principle

1. **Spatial Feature Extraction**: Convolution operations capture local spatial patterns (e.g., temperature gradients, pressure systems)
2. **Multi-scale Representation**: Learns features at different scales through downsampling and upsampling
3. **Temporal Information Processing**: Treats multiple timesteps as multi-channel input, but **does not explicitly model temporal dependencies**
4. **Spatial Structure Preservation**: Maintains spatial dimensions, utilizes correlations of adjacent grids

#### Relationship to Task

- **Advantages**:
  - Effectively extracts spatial features (e.g., fronts, cyclones)
  - High computational efficiency (convolution operations)
  - Relatively fewer parameters
- **Limitations**:
  - **No temporal modeling**: Treats timesteps as channels, cannot learn temporal evolution patterns
  - Cannot capture long-term temporal dependencies

#### Use Cases

- ✅ Spatial pattern prediction (e.g., static weather maps)
- ✅ Limited computational resources
- ❌ Not suitable for prediction tasks requiring temporal dependencies

#### Extension: DeepCNN

`src/models/cnn.py` also implements `DeepCNN`:

- **Residual Connections**: Uses ResidualBlock to build deeper networks
- **Structure**: Input projection → Multiple residual blocks → Output projection
- **Advantage**: Deeper networks can learn more complex spatial patterns
- **Parameters**: `n_residual_blocks` controls number of residual blocks (default 3)

---

### ConvLSTM

#### Model Structure

```
Input: (batch, 12, channels, H, W)
  ↓
ConvLSTM encoder (multi-layer):
  ConvLSTMCell: Convolutional LSTM cell
    - Input gate (i): Controls new information flow
    - Forget gate (f): Controls old information forgetting
    - Cell gate (g): Candidate values
    - Output gate (o): Controls output
    - All operations are convolutions, not fully connected
  ↓
Take last layer's last timestep: (batch, hidden_channels, H, W)
  ↓
Output projection:
  Conv2d(hidden_channels, hidden_channels//2)
  Conv2d(hidden_channels//2, 4*channels)
  ↓
Reshape: (batch, 4, channels, H, W)
```

#### ConvLSTM Cell Details

```python
# ConvLSTMCell core operations
combined = concat([input, hidden_state])  # (B, C_in+C_hid, H, W)
gates = Conv2d(combined) → (B, 4*C_hid, H, W)  # i, f, g, o
i, f, g, o = split(gates)

cell_next = f * cell + i * g  # Update cell state
hidden_next = o * tanh(cell_next)  # Update hidden state
```

#### Input/Output

- **Input**: `(batch, input_length=12, input_channels, H=32, W=64)`
- **Output**: `(batch, output_length=4, input_channels, H=32, W=64)`
- **Key Parameters**:
  - `hidden_channels`: ConvLSTM hidden channels (default 64)
  - `num_layers`: Number of ConvLSTM layers (default 2)
  - `kernel_size`: Convolution kernel size (default 3)

#### Working Principle

1. **Spatiotemporal Joint Modeling**: 
   - **Temporal Dimension**: LSTM gating mechanism models temporal dependencies
   - **Spatial Dimension**: Convolution operations preserve spatial structure
   
2. **Memory Mechanism**: 
   - Cell state stores long-term spatiotemporal memory
   - Hidden state encodes current spatiotemporal features

3. **Multi-scale Features**: 
   - Multi-layer ConvLSTM learns features at different abstraction levels
   - Lower layers capture local patterns, higher layers capture global patterns

4. **Spatial Correlations**: 
   - Convolution operations automatically capture correlations of adjacent grids
   - No need to manually design spatial dependencies

#### Relationship to Task

- **Advantages**:
  - ✅ **Simultaneously models spatiotemporal dependencies**, most suitable for weather prediction tasks
  - ✅ Preserves spatial structure, fully utilizes grid data
  - ✅ Memory mechanism suitable for capturing weather system evolution patterns
  - ✅ Best performance among deterministic models

- **Limitations**:
  - Large parameter count
  - Longer training time
  - Cannot quantify uncertainty

#### Use Cases

- ✅ **General weather prediction** (recommended)
- ✅ Tasks requiring spatiotemporal modeling
- ✅ Deterministic prediction scenarios

#### Extension: ConvLSTM Seq2Seq

`src/models/convlstm.py` also implements `ConvLSTMSeq2Seq`:

**Model Structure**:
```
Input: (batch, 12, channels, H, W)
  ↓
Encoder (ConvLSTM multi-layer):
  Process input sequence, extract historical spatiotemporal features
  ↓
Take last layer's last state: (h, c)
  ↓
Decoder (autoregressive):
  for t in range(output_length):
    decoder_input = last frame (or previous step prediction)
    h, c = ConvLSTMCell(decoder_input, (h, c))
    out = Conv2d(h) → (batch, channels, H, W)
    Use out as next step's decoder_input
  ↓
Output: (batch, 4, channels, H, W)
```

**Features**:
- Encoder-decoder architecture, more suitable for sequence-to-sequence tasks
- Autoregressive generation: Uses prediction as next step input (free-running mode)
- Optional teacher forcing: Can use ground truth as next step input during training
- Suitable for scenarios requiring longer output sequences or wanting to inject external conditions at decoding stage

---

### Weather Transformer

#### Model Structure

```
Input: (batch, 12, channels, H, W)
  ↓
Patch Embedding (each timestep):
  Conv2d(C, embed_dim, kernel=patch_size, stride=patch_size)
  (32, 64) → (8, 8) patches, each patch 4×8
  ↓
Positional encoding:
  - Temporal positional encoding: Sinusoidal encoding (strong extrapolation ability)
  - Spatial positional encoding: Learnable parameters
  ↓
Spatiotemporal attention encoder (Encoder):
  Spatial Attention: Attention between patches within same timestep
  Temporal Attention: Attention across timesteps for same patch
  Factorized design: O(T*N² + N*T²) vs O((T*N)²)
  ↓
Learnable output queries:
  (batch, 4, 8*8, embed_dim)  # Queries for future 4 timesteps
  ↓
Decoder:
  Shallow Transformer decoder
  ↓
Output projection:
  Linear(embed_dim, patch_size² * channels)
  ↓
Reshape: (batch, 4, channels, H, W)
```

#### Input/Output

- **Input**: `(batch, input_length=12, input_channels, H=32, W=64)`
- **Output**: `(batch, output_length=4, input_channels, H=32, W=64)`
- **Key Parameters**:
  - `d_model`: Model dimension (default 128)
  - `n_heads`: Number of attention heads (default 4)
  - `n_layers`: Number of encoder layers (default 4)
  - `patch_size`: Patch size (default (4, 8))

#### Working Principle

1. **Patch-based Processing**:
   - Divides spatial grid into patches (similar to ViT)
   - Each patch serves as a token
   - Reduces computational complexity

2. **Factorized Spatiotemporal Attention**:
   - **Spatial Attention**: Attention between patches within same timestep
     - Captures spatial correlations (e.g., similarity of adjacent regions)
   - **Temporal Attention**: Attention across timesteps for same patch
     - Captures temporal evolution patterns
   - **Advantage**: Computational complexity reduced from O((T*N)²) to O(T*N² + N*T²)

3. **Positional Encoding**:
   - Temporal positional encoding: Sinusoidal encoding (better extrapolation ability)
   - Spatial positional encoding: Learnable parameters (adapts to irregular Earth grid)

4. **Lightweight Design**:
   - Parameter count ~1.6M (comparable to ConvLSTM)
   - Uses post-LN (residual after LayerNorm) with better initialization

#### Relationship to Task

- **Advantages**:
  - ✅ Captures long-range spatiotemporal dependencies (attention mechanism)
  - ✅ Lightweight design, fewer parameters
  - ✅ Suitable for capturing large-scale weather systems (e.g., global circulation)

- **Limitations**:
  - Patch division may lose details
  - Requires more data for training
  - Computational complexity still relatively high

#### Use Cases

- ✅ Tasks requiring long-range dependencies
- ✅ Large-scale weather system prediction
- ✅ Sufficient computational resources

---

## WeatherDiff Module

WeatherDiff is a weather prediction module based on Stable Diffusion architecture, treating meteorological grid data as images and using pre-trained VAE and U-Net architectures for spatiotemporal prediction.

### Core Concepts

1. **Image-based Processing**: Treats weather fields as images (each timestep is a frame)
2. **VAE Compression**: Uses pre-trained VAE to compress high-dimensional images to low-dimensional latent space
3. **Latent Space Prediction**: Predicts in latent space, reducing computational complexity
4. **Probabilistic Modeling**: Diffusion models support uncertainty quantification

---

### VAE (Stable Diffusion VAE)

#### Model Structure

```
Encoder:
  Input image: (B, C, H, W)  # Range [-1, 1]
    ↓
  Convolutional downsampling: H, W → H//8, W//8
    ↓
  Latent vector: (B, 4, H//8, W//8)  # Compression ratio 8×8 = 64×
  
Decoder:
  Latent vector: (B, 4, H//8, W//8)
    ↓
  Convolutional upsampling: H//8, W//8 → H, W
    ↓
  Reconstructed image: (B, C, H, W)  # Range [-1, 1]
```

#### Input/Output

- **Encoding**:
  - **Input**: `(batch, channels, H, W)` - Range [-1, 1]
  - **Output**: `(batch, 4, H//8, W//8)` - Latent vector
  
- **Decoding**:
  - **Input**: `(batch, 4, H//8, W//8)` - Latent vector
  - **Output**: `(batch, channels, H, W)` - Range [-1, 1]

#### Training Strategy

Current implementation only supports loading Stable Diffusion pre-trained VAE weights (default from HuggingFace, can specify custom weights via `--vae-pretrained-path`). Can separately control whether encoder/decoder participate in training:

- `--freeze-encoder`: Freeze encoder, only train decoder+U-Net
- `--freeze-decoder`: Freeze decoder, only train encoder+U-Net
- Both: Completely freeze VAE, only train U-Net
- Neither: Fine-tune encoder/decoder together with U-Net

#### Working Principle

1. **Pre-trained Model**: Uses Stable Diffusion's pre-trained VAE (trained on natural images)
2. **Compressed Representation**: Compresses 512×512 images to 64×64 latent space (64× compression ratio)
3. **Semantic Preservation**: Latent space preserves image semantic information, suitable for generation tasks
4. **Reconstruction Error**: For weather data, reconstruction RMSE ~5-10K (temperature units)
5. **Trainability**: Supports freezing VAE (inference only) or training VAE (fine-tuning)

#### Relationship to Task

- **Advantages**:
  - ✅ Significantly reduces computational complexity (64× compression)
  - ✅ Pre-trained model, no need to train from scratch
  - ✅ Latent space more suitable for generation tasks
  - ✅ Flexible control of VAE fine-tuning scope through freeze switches

- **Limitations**:
  - ❌ Large reconstruction error (5-10K), may lose details
  - ❌ Pre-trained on natural images, may not be suitable for weather data (can be improved through fine-tuning)
  - ❌ Requires data normalization to [-1, 1]

#### Use Cases

- ✅ Large image prediction (512×512 and above)
- ✅ Scenarios requiring reduced memory and computation
- ✅ Scenarios requiring fine-tuning VAE encoder/decoder for weather data
- ❌ Not suitable for tasks requiring high-precision reconstruction

---

### RAE (Representation Autoencoder)

#### Model Structure

```
Encoder:
  Input image: (B, C, H, W)  # Range [0, 1], automatically converted from [-1, 1]
    ↓
  Resize to encoder_input_size (e.g., 256×256)
    ↓
  Vision Transformer (DINOv2/SigLIP2/MAE)
    ↓
  Latent vector: (B, latent_dim, H_latent, W_latent)
    # latent_dim depends on encoder (e.g., 768 for DINOv2-base)
    # H_latent, W_latent = encoder_input_size // patch_size

Decoder:
  Latent vector: (B, latent_dim, H_latent, W_latent)
    ↓
  Vision Transformer Decoder (MAE-based)
    ↓
  Reconstructed image: (B, C, H, W)  # Range [0, 1], automatically converted to [-1, 1]
```

#### Input/Output

- **Encoding**:
  - **Input**: `(batch, channels, H, W)` - Range [-1, 1] (automatically converted to [0, 1])
  - **Output**: `(batch, latent_dim, H_latent, W_latent)` - Latent vector
    - `latent_dim`: Depends on encoder (DINOv2-base: 768, SigLIP2-base: 768)
    - `H_latent, W_latent`: Depends on encoder input size and patch size
  
- **Decoding**:
  - **Input**: `(batch, latent_dim, H_latent, W_latent)` - Latent vector
  - **Output**: `(batch, channels, H, W)` - Range [-1, 1] (automatically converted from [0, 1])

#### Supported Encoder Types

1. **DINOv2** (`Dinov2withNorm`):
   - Config path: `facebook/dinov2-base`
   - Input size: 224×224 (default)
   - Latent dimension: 768
   - Latent space: 14×14 (224/16=14)

2. **SigLIP2** (`SigLIP2wNorm`) ⭐ **Recommended**:
   - Config path: `google/siglip2-base-patch16-256`
   - Input size: 256×256 (default)
   - Latent dimension: 768
   - Latent space: 16×16 (256/16=16)
   - **Advantage**: Larger input size, better spatial resolution

3. **MAE** (`MAEwNorm`):
   - Config path: `facebook/vit-mae-base`
   - Input size: 224×224
   - Latent dimension: 768
   - Latent space: 14×14

#### Working Principle

1. **Encoder Fixed**: Encoder parameters fixed, do not participate in training (`freeze_encoder=True`)
2. **Decoder Fine-tunable**: Decoder parameters trainable (`freeze_decoder=False`), supports fine-tuning
3. **Automatic Range Conversion**: 
   - Input: [-1, 1] → [0, 1] (used internally by RAE)
   - Output: [0, 1] → [-1, 1] (matches Weather project)
4. **Flexible Configuration**: Supports multiple encoder types, can choose based on needs

#### Relationship to Task

- **Advantages**:
  - ✅ **Decoder Fine-tunable**: Compared to SD VAE, RAE's decoder can be fine-tuned for weather data
  - ✅ **Multiple Encoder Choices**: Supports DINOv2, SigLIP2, MAE and other pre-trained encoders
  - ✅ **Larger Latent Dimension**: 768 dimensions (vs SD VAE's 4 dimensions), may preserve more information
  - ✅ **Automatic Range Conversion**: No need to manually handle data ranges

- **Limitations**:
  - ❌ Encoder requires resizing input images (may lose details)
  - ❌ Larger latent dimension may increase computation
  - ❌ Requires additional decoder training

#### Use Cases

- ✅ Scenarios requiring fine-tunable decoder
- ✅ Scenarios wanting to leverage different pre-trained encoder features
- ✅ Large image prediction (512×512 and above)
- ✅ Sufficient memory (larger latent dimension)

#### Differences from SD VAE

| Feature | SD VAE | RAE |
|---------|--------|-----|
| Latent channels | 4 | Depends on encoder (e.g., 768) |
| Latent shape | (4, H//8, W//8) | (latent_dim, H_latent, W_latent) |
| Encoder | Fixed | Fixed (configurable) |
| Decoder | Fixed | **Fine-tunable** ⭐ |
| Input range | [-1, 1] | [-1, 1] (internal conversion) |
| Output range | [-1, 1] | [-1, 1] (internal conversion) |
| Input resize | No | Yes (to encoder_input_size) |

#### Usage Example

```python
from weatherdiff.vae import RAEWrapper

vae_wrapper = RAEWrapper(
    encoder_cls='SigLIP2wNorm',
    encoder_config_path='google/siglip2-base-patch16-256',
    encoder_input_size=256,
    decoder_config_path='vit_mae-base',
    decoder_patch_size=16,
    device='cuda',
    freeze_encoder=True,
    freeze_decoder=False  # decoder fine-tunable
)

# Encode
latent = vae_wrapper.encode(images)  # (B, 768, 16, 16)

# Decode
reconstructed = vae_wrapper.decode(latent)  # (B, C, H, W)
```

**Currently, the project mainly uses two modes:**

1. **Stage-A (RAE Reconstruction)**: Only Encoder + Decoder, reconstructing future weather fields.
2. **Stage-1 Direct (Direct Prediction)**: Using historical sequences through Encoder → Decoder to directly predict future weather fields (without DiT).

#### Encoder+Decoder Direct Prediction Model Structure
```
History Encoder (ExternalRepEncoder for history):
  Input history sequence: (B, T_in, C_in, H, W)
    ↓
  Time + channel concatenation: (B, T_in * C_in, H, W)
    ↓
  1×1 convolution in_adapter: (T_in*C_in) → 3 channels
    ↓
  Resize to encoder_input_size (e.g., 518×518)
    ↓
  Pre-trained Vision Transformer backbone (DINOv2 / MAE / SigLIP2)
    ↓
  History tokens:
    (B, latent_dim, H_latent, W_latent)

RAE Decoder (RAEDecoder):
  History tokens: (B, latent_dim, H_latent, W_latent)
    ↓
  RAEDecoder directly predicts future:
    ↓
  Predicted future weather field: (B, T_out, C_out, H, W)
    # Example: from 12-step history → 4-step future 2m_temperature
```

#### Input/Output

- **Encoding**:
  - **Input**: `(B, T_in, C_in, H, W)` - x_hist
  - **Output**: `(B, latent_dim, H_latent, W_latent)` - tokens_hist
    - `latent_dim`: Depends on encoder (DINOv2-base: 768, SigLIP2-base: 768, DINOv2-G/14: 1536)
    - `H_latent, W_latent`: Depends on encoder input size and patch size
  
- **Decoding**:
  - **Input**: `(B, latent_dim, H_latent, W_latent)` - tokens_hist
  - **Output**: `(B, T_out, C_out, H, W)` - y_pred

#### Supported Encoder Types (Examples)

1. **DINOv2-G/14** (currently used in experiments)
   - `timm` name: `vit_giant_patch14_dinov2`
   - Input size: 518×518
   - Patch size: 14
   - Latent dimension: 1536
   - Latent space: 37×37

2. **DINOv2-Base/16**
   - `facebook/dinov2-base`
   - Input size: 224×224
   - Patch size: 16
   - Latent dimension: 768
   - Latent space: 14×14

3. **SigLIP2 / MAE / Other ViTs**
   - Can switch by modifying `timm.create_model` name
   - External structure (in_adapter → ViT → RAEDecoder) remains unchanged

#### Working Principle (for current project)

1. **Encoder backbone frozen, only train in_adapter (optional)**
   - DINOv2 backbone parameters fixed (`freeze_backbone=True`)
   - 1×1 in_adapter can be trained or frozen, used to map weather fields to encoder's expected distribution space

2. **RAEDecoder fine-tunable**
   - Decoder weights loaded from RAE pre-trained weights (or randomly initialized), continue training on weather tasks
   - Stage-A: Learn "future → latent space → reconstruct future" mapping
   - Stage-1: Learn "history → latent space → predict future" mapping

3. **Automatic resize / interpolation**
   - Input ERA5 grid is small (64×32)
   - ExternalRepEncoder internally resizes to ViT pre-trained size (e.g., 518×518), encodes to 37×37 tokens
   - Decoder output is then interpolated/reshaped back to original grid size `(H, W)`

4. **Flexible combination of Stage-A / Stage-1 / Stage-B**
   - Stage-A: Only evaluate RAE reconstruction quality (MSE/MAE) to assess latent space representation
   - Stage-1: Directly use RAE for deterministic prediction
   - Stage-B (optional): Add DH-DiT / SimpleDiT on latent space for diffusion prediction (existing RAE+LDM version)

#### Relationship to Task (Weather Prediction)

- **Advantages**:
  - ✅ **Decoder can be fine-tuned for weather fields**: Compared to SD VAE which is fixed and image-oriented, RAE decoder can be retrained on ERA5, making reconstruction/prediction more aligned with meteorological structure
  - ✅ **Leverage large-scale visual pre-training (DINOv2)**: Has advantages in spatial structure understanding, beneficial for capturing fronts, troughs, vortices, etc.
  - ✅ **High-dimensional latent representation**: 1536×37×37 latent capacity is much larger than 4×(H/8×W/8), more suitable for preserving multi-variable, multi-temporal information
  - ✅ **Strong generality**: Same ExternalRepEncoder + RAEDecoder can be shared by Stage-A reconstruction, Stage-1 Direct, Stage-B diffusion

- **Limitations**:
  - ❌ Must resize ERA5 grid (64×32 → 518×518), introducing interpolation error
  - ❌ Latent is very large, requires high memory and computing power
  - ❌ Decoder / in_adapter require additional training, engineering complexity higher than "directly using SD VAE"

#### Use Cases

- ✅ Want to leverage DINOv2 and other **strong pre-trained vision models** in weather tasks
- ✅ Need **fine-tunable decoder** to adapt to ERA5/WeatherBench datasets
- ✅ High accuracy requirements for spatial structure (pattern recognition, extreme events), not satisfied with simple CNN/VAE
- ✅ Training environment with memory ≥ 24GB (especially when using DINOv2-G/14)

#### Differences from SD VAE (comparison in this project)

| Feature | SD VAE | RAE-Weather |
|---------|--------|-------------|
| Latent channels | 4 | latent_dim (e.g., 1536 for DINOv2-G/14) |
| Latent space | (4, H/8, W/8) | (latent_dim, H_latent, W_latent) |
| Encoder backbone | CNN-based VAE, fixed | ViT-based (DINOv2/MAE/SigLIP2), fixed |
| Decoder | Fixed | **Fine-tunable** (RAEDecoder) |
| Input range / Normalization | [-1,1] / simple normalization | Normalized weather fields (dataset statistics/climatology) |
| Input resize | Usually not needed | Need resize to encoder_input_size |
| Fitting to task | VAE reconstructs images | RAE reconstructs & directly predicts weather fields |

---

### Pixel U-Net

#### Model Structure

```
Input: (batch, 12, channels, H, W)
  ↓
Flatten temporal dimension: (batch, 12*channels, H, W)
  ↓
U-Net:
  Downsampling path (Encoder):
    ConvBlock → MaxPool → ConvBlock → MaxPool → ...
    Preserve skip connections
  ↓
  Bottleneck layer:
    ConvBlock
  ↓
  Upsampling path (Decoder):
    UpSample → Concat(skip) → ConvBlock → ...
  ↓
Output: (batch, 4*channels, H, W)
  ↓
Reshape: (batch, 4, channels, H, W)
```

#### U-Net Detailed Structure

```python
# Convolution block (uses GroupNorm and SiLU activation)
ConvBlock:
  Conv2d → GroupNorm(8) → SiLU → Conv2d → GroupNorm(8) → SiLU

# Downsampling block
DownBlock:
  ConvBlock(in_ch, out_ch) → MaxPool2d(2)
  Save skip connection (for concatenation during upsampling)

# Upsampling block
UpBlock:
  ConvTranspose2d(in_ch, in_ch//2, stride=2)
  Concat([up, skip])  # Concatenate skip connection
  ConvBlock(in_ch//2+skip_ch, out_ch)

# Overall structure
Input: (B, T_in*C, H, W)
  → Input convolution: (B, base_channels, H, W)
  → Downsampling path (depth layers): Gradually downsample, save skip
  → Bottleneck layer: ConvBlock
  → Upsampling path (depth layers): Gradually upsample, use skip
  → Output convolution: (B, T_out*C, H, W)
```

#### Input/Output

- **Input**: `(batch, input_length=12, channels, H, W)`
- **Output**: `(batch, output_length=4, channels, H, W)`
- **Key Parameters**:
  - `base_channels`: Base channel count (default 64)
  - `depth`: U-Net depth (default 4)

#### Working Principle

1. **Image-to-Image Prediction**: Directly predicts in pixel space
2. **Multi-scale Features**: U-Net's downsampling-upsampling structure captures multi-scale features
3. **Skip Connections**: Preserves detail information, avoids information loss
4. **Temporal Information**: Treats multiple timesteps as multi-channel input

#### Relationship to Task

- **Advantages**:
  - ✅ Fast training (directly in pixel space)
  - ✅ Deterministic results (no randomness)
  - ✅ Suitable for image prediction tasks

- **Limitations**:
  - ❌ High memory requirements (large images)
  - ❌ No temporal modeling (timesteps as channels)
  - ❌ Cannot quantify uncertainty

#### Use Cases

- ✅ Small images (64×32)
- ✅ Quick prototype validation
- ❌ Not suitable for large images (512×512)

---

### Latent U-Net

#### Model Structure

**Using SD VAE**:
```
Input images: (batch, 12, channels, H, W)
  ↓
SD VAE encoding (batch processing, control memory):
  (batch*12, channels, H, W) → (batch*12, 4, H//8, W//8)
  ↓
Reshape: (batch, 12, 4, H//8, W//8)
  ↓
Flatten temporal dimension: (batch, 12*4, H//8, W//8)
  ↓
Latent U-Net (in latent space):
  Input convolution → Downsampling path → Bottleneck → Upsampling path → Output convolution
  (Uses same U-Net structure as Pixel U-Net, but different input/output channels)
  ↓
Output: (batch, 4*4, H//8, W//8)
  ↓
Reshape: (batch, 4, 4, H//8, W//8)
  ↓
SD VAE decoding (batch processing):
  (batch*4, 4, H//8, W//8) → (batch*4, channels, H, W)
  ↓
Reshape: (batch, 4, channels, H, W)
```

**Using RAE**:
```
Input images: (batch, 12, channels, H, W)
  ↓
RAE encoding (batch processing):
  (batch*12, channels, H, W) 
    → Resize to encoder_input_size (e.g., 256×256)
    → Vision Transformer Encoder (DINOv2/SigLIP2/MAE)
    → (batch*12, latent_dim, H_latent, W_latent)
    # latent_dim depends on encoder (e.g., 768 for DINOv2-base)
    # H_latent, W_latent = encoder_input_size // patch_size
  ↓
Reshape: (batch, 12, latent_dim, H_latent, W_latent)
  ↓
Flatten temporal dimension: (batch, 12*latent_dim, H_latent, W_latent)
  ↓
Latent U-Net (in latent space):
  Input convolution → Downsampling path → Bottleneck → Upsampling path → Output convolution
  (Same structure as SD VAE version, but latent_channels=latent_dim)
  ↓
Output: (batch, 4*latent_dim, H_latent, W_latent)
  ↓
Reshape: (batch, 4, latent_dim, H_latent, W_latent)
  ↓
RAE decoding (batch processing, decoder fine-tunable):
  (batch*4, latent_dim, H_latent, W_latent)
    → Vision Transformer Decoder (MAE-based, trainable)
    → (batch*4, channels, H, W)
  ↓
Reshape: (batch, 4, channels, H, W)
```

#### Input/Output

- **Input**: `(batch, input_length=12, channels, H, W)` - Pixel space
- **Output**: `(batch, output_length=4, channels, H, W)` - Pixel space
- **Intermediate Representation (SD VAE)**: Latent space `(batch, T, 4, H//8, W//8)`
- **Intermediate Representation (RAE)**: Latent space `(batch, T, latent_dim, H_latent, W_latent)`
- **Key Parameters**:
  - `vae_type`: VAE type, 'sd' or 'rae'
  - `base_channels`: U-Net base channel count (default 128)
  - `depth`: U-Net depth (default 3)
  - `vae_batch_size`: VAE encoding/decoding batch size (control memory)
  - **SD VAE Specific Parameters**:
    - `vae_model_id`: SD VAE model ID (default 'stable-diffusion-v1-5')
    - `vae_pretrained_path`: Optional, pre-trained weight path (overrides default SD weights)
    - `freeze_encoder`: Whether to freeze encoder
    - `freeze_decoder`: Whether to freeze decoder
  - **RAE Specific Parameters**:
    - `rae_encoder_cls`: Encoder type ('Dinov2withNorm', 'SigLIP2wNorm', 'MAEwNorm')
    - `rae_encoder_config_path`: Encoder config path
    - `rae_encoder_input_size`: Encoder input size (default 256)
    - `rae_decoder_config_path`: Decoder config path
    - `rae_decoder_patch_size`: Decoder patch size (default 16)
    - `freeze_encoder`: Freeze encoder (default True)
    - `freeze_decoder`: Freeze decoder (default False)

#### Working Principle

1. **VAE Encoding**: 
   - **SD VAE**: Encodes input images to latent space (64× compression, 4 channels)
     - Only supports loading pre-trained weights, can specify custom path
     - Controls which parts participate in training via `freeze_encoder` / `freeze_decoder`
   - **RAE**: Encodes input images to latent space after resizing (latent_dim channels, e.g., 768)
2. **Latent Space Prediction**: U-Net predicts future frames in latent space
3. **VAE Decoding**: 
   - **SD VAE**: Decodes predicted latent vectors back to pixel space
     - Decoder trainable by default, can disable fine-tuning via `freeze_decoder`
   - **RAE**: Decodes predicted latent vectors back to pixel space (fine-tunable decoder)

4. **Advantages**:
   - **SD VAE**: Low memory requirements (latent space 64× smaller than pixel space), stable pre-trained model, supports VAE fine-tuning
   - **RAE**: Fine-tunable decoder, may achieve better reconstruction quality, supports multiple encoder choices

#### Relationship to Task

- **Advantages**:
  - ✅ **Low memory requirements** (512×512 → 64×64 for SD VAE)
  - ✅ More stable training
  - ✅ Smoother generation results
  - ✅ Suitable for large images
  - ✅ **SD VAE**: Both encoder/decoder can be fine-tuned separately
  - ✅ **RAE**: Fine-tunable decoder, may achieve better reconstruction quality

- **Limitations**:
  - ❌ VAE reconstruction error (5-10K for SD VAE, can be improved through fine-tuning)
  - ❌ Requires VAE encoding/decoding steps (increases computation time)
  - ❌ Cannot quantify uncertainty
  - ❌ **RAE**: Encoder requires resizing input, larger latent dimension

#### Use Cases

- ✅ **Large image prediction** (recommended)
- ✅ Limited memory (SD VAE)
- ✅ Tasks requiring smooth prediction results
- ✅ **Requiring fine-tunable decoder** (RAE)

---

### Diffusion Model

#### Model Structure

```
Training phase:
  Condition (past frames): (batch, 12, channels, H, W)
    ↓ VAE encoding
  Condition latent vector: (batch, 12, 4, H//8, W//8)
  
  Target (future frames): (batch, 4, channels, H, W)
    ↓ VAE encoding
  Target latent vector: (batch, 4, 4, H//8, W//8)
    ↓
  Add noise: latent_target + noise * sqrt(beta_t)
    ↓
  U-Net predicts noise: predicted_noise
    ↓
  Loss: MSE(predicted_noise, true_noise)

Inference phase:
  Condition latent vector: (batch, 12, 4, H//8, W//8)
  Random noise: (batch, 4, 4, H//8, W//8)
    ↓
  Gradual denoising (T steps):
    for t in [T-1, ..., 0]:
      predicted_noise = U-Net(noisy_latent, t, condition)
      latent = scheduler.step(predicted_noise, t, latent)
    ↓
  Denoised latent vector: (batch, 4, 4, H//8, W//8)
    ↓ VAE decoding
  Predicted image: (batch, 4, channels, H, W)
```

#### Diffusion U-Net Structure

```python
# U-Net with time embedding
Input: 
  - noisy_latent: (B, 4*4, H//8, W//8)  # Noisy future frames
  - condition: (B, 12*4, H//8, W//8)   # Past frames
  - timestep: (B,)
  
Concatenate: (B, (12+4)*4, H//8, W//8)
  ↓
Time embedding: TimestepEmbedding(timestep) → (B, time_emb_dim)
  ↓
U-Net (with time embedding):
  Downsampling blocks: ConvBlockWithTime(x, time_emb)
  Bottleneck layer: ConvBlockWithTime(x, time_emb)
  Upsampling blocks: ConvBlockWithTime(x, time_emb)
  ↓
Output: (B, 4*4, H//8, W//8)  # Predicted noise
```

#### Input/Output

- **Training**:
  - **Input**: 
    - Condition: `(batch, input_length=12, channels, H, W)`
    - Target: `(batch, output_length=4, channels, H, W)`
  - **Output**: Predicted noise `(batch, output_length=4, 4, H//8, W//8)`

- **Inference**:
  - **Input**: Condition `(batch, input_length=12, channels, H, W)`
  - **Output**: Prediction `(batch, output_length=4, channels, H, W)`
  - **Optional**: Generate multiple samples (ensemble prediction)

#### Working Principle

1. **Forward Diffusion Process (Training)**:
   - Gradually adds noise to target: `x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise`
   - Learns to predict noise added at each step

2. **Reverse Denoising Process (Inference)**:
   - Starts from random noise
   - Gradually denoises, predicts and removes noise at each step
   - Finally obtains clear prediction result

3. **Conditional Generation**:
   - Uses past frames as condition input to U-Net
   - U-Net predicts noise based on condition and timestep

4. **Uncertainty Quantification**:
   - Can generate multiple samples (different random seeds)
   - Ensemble prediction provides uncertainty estimates

#### Noise Scheduler

```python
# DDPM scheduler
beta_t = linear_schedule(0.0001, 0.02, T=1000)
alpha_t = 1 - beta_t
alpha_bar_t = cumprod(alpha_t)

# Add noise
noisy = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

# Denoising step
predicted_x_0 = (x_t - sqrt(1-alpha_bar_t) * predicted_noise) / sqrt(alpha_bar_t)
x_{t-1} = sqrt(alpha_bar_{t-1}) * predicted_x_0 + sqrt(1-alpha_bar_{t-1}) * predicted_noise
```

#### Relationship to Task

- **Advantages**:
  - ✅ **Uncertainty Quantification**: Can generate multiple future scenarios
  - ✅ **Ensemble Prediction**: Average of multiple samples more accurate
  - ✅ **Probabilistic Modeling**: Suitable for chaotic properties of weather systems
  - ✅ **High-quality Generation**: Diffusion models generate high quality

- **Limitations**:
  - ❌ Long training time (needs to learn denoising process)
  - ❌ Slow inference (requires multi-step denoising, e.g., 50-1000 steps)
  - ❌ High memory requirements (U-Net + VAE)

#### Use Cases

- ✅ **Tasks requiring uncertainty estimation** (recommended)
- ✅ Ensemble prediction
- ✅ Probabilistic weather forecasting
- ❌ Not suitable for fast inference scenarios

---

## Model Comparison and Selection

### Model Feature Comparison Table

| Model | Spatiotemporal Modeling | Training Speed | Inference Speed | Uncertainty | Memory Requirements | Parameters | Recommended Scenarios |
|-------|-------------------------|----------------|-----------------|-------------|---------------------|------------|----------------------|
| **Linear Regression** | ✗ | ⚡⚡⚡ | ⚡⚡⚡ | ✗ | Low | Very Few | Quick Baseline |
| **LSTM** | Temporal | ⚡⚡ | ⚡⚡ | ✗ | Medium | Medium | Single-point Prediction |
| **Transformer** | Temporal | ⚡⚡ | ⚡⚡ | ✗ | Medium | Medium | Single-point Prediction (Long-range Dependencies) |
| **CNN** | Spatial | ⚡⚡ | ⚡⚡⚡ | ✗ | Medium | Medium | Spatial Patterns |
| **ConvLSTM** | Spatiotemporal | ⚡ | ⚡⚡ | ✗ | Medium | Medium | **General Prediction** ⭐ |
| **Weather Transformer** | Spatiotemporal | ⚡ | ⚡ | ✗ | Medium | Few | Long-range Dependencies |
| **Pixel U-Net** | Spatial (Channel Stacking) | ⚡⚡ | ⚡⚡ | ✗ | High | Medium | Small Images |
| **Latent U-Net (SD VAE)** | Spatial (Latent Stacking) | ⚡⚡ | ⚡⚡ | ✗ | Low | Medium | **Large Images** ⭐ |
| **Latent U-Net (RAE)** | Spatial (Latent Stacking) | ⚡⚡ | ⚡⚡ | ✗ | Medium | Medium | **Large Images (Fine-tunable)** ⭐ |
| **Diffusion** | Spatiotemporal | 🐢 | 🐢 | ✓ | High | Medium | **Probabilistic Prediction** ⭐ |

### Selection Guide

#### 1. Quick Baseline Testing
- **Recommendation**: Linear Regression
- **Reason**: Extremely fast training, validates data flow

#### 2. Deterministic Prediction (General)
- **Recommendation**: ConvLSTM or Latent U-Net
- **ConvLSTM**: Suitable for 64×32 small size, fast training
- **Latent U-Net (SD VAE)**: Suitable for 512×512 large size, memory-friendly
- **Latent U-Net (RAE)**: Suitable for 512×512 large size, decoder fine-tunable

#### 3. Uncertainty Quantification
- **Recommendation**: Diffusion Model
- **Reason**: Only model supporting probabilistic prediction

#### 4. Limited Computational Resources
- **CPU**: Linear Regression, LSTM
- **Single GPU (8GB)**: CNN, ConvLSTM
- **Single GPU (12GB+)**: Latent U-Net, Diffusion

#### 5. Large Images (512×512)
- **Recommendation**: Latent U-Net (SD VAE or RAE)
- **SD VAE**: Low memory requirements, stable training
- **RAE**: Fine-tunable decoder, may achieve better reconstruction quality

### Model-Task Relationship Summary

1. **Spatiotemporal Modeling Requirements**:
   - ✅ ConvLSTM, Weather Transformer, Latent U-Net, Diffusion
   - ⚠️ Transformer, LSTM (temporal only, no spatial)
   - ❌ Linear Regression, CNN (no temporal)

2. **Uncertainty Requirements**:
   - ✅ Diffusion Model
   - ❌ All other models (deterministic)

3. **Computational Efficiency Requirements**:
   - ✅ Linear Regression, CNN, Pixel U-Net
   - ❌ Diffusion Model

4. **Large-scale Data**:
   - ✅ Latent U-Net (SD VAE or RAE compression)
   - ❌ Pixel U-Net (insufficient memory)

5. **Requiring Fine-tunable Decoder**:
   - ✅ Latent U-Net (RAE)
   - ❌ Latent U-Net (SD VAE, decoder fixed)

---

## Summary

This project provides a complete model system from simple baselines to complex probabilistic models:

1. **Traditional Models**: Suitable for quick validation and simple tasks
2. **ConvLSTM**: Best choice for deterministic prediction
3. **WeatherDiff Module**: 
   - **Latent U-Net (SD VAE)**: Deterministic prediction for large images, memory-friendly
   - **Latent U-Net (RAE)**: Deterministic prediction for large images, decoder fine-tunable
   - **Diffusion**: Probabilistic prediction and uncertainty quantification

Selecting the appropriate model requires balancing:
- Prediction accuracy vs computational efficiency
- Deterministic vs uncertainty
- Model complexity vs data volume
- Fixed decoder vs fine-tunable decoder

Recommend starting with ConvLSTM, then selecting based on needs:
- **Large images (512×512)**: Latent U-Net (SD VAE or RAE)
- **Requiring uncertainty estimation**: Diffusion
- **Requiring fine-tunable decoder**: Latent U-Net (RAE)


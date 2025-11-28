# Weather Prediction

A deep learning-based weather prediction system supporting multiple model architectures and prediction methods.

**[中文](README.md) | [English](README_EN.md)**

## 📖 Project Overview

This project is a deep learning-based global weather prediction research project with the following features:

- **Prediction Task**: Predict future 1 day (4 time steps) of weather conditions based on past 3 days (12 time steps, 6-hour intervals) of weather data
- **Multi-variable Support**: Supports multiple meteorological variables including temperature, geopotential height, wind speed, specific humidity, etc.
- **Multiple Models**: From traditional deep learning models (CNN, ConvLSTM, Transformer) to WeatherDiff module based on Stable Diffusion
- **Uncertainty Quantification**: Supports probabilistic prediction and uncertainty estimation through Diffusion models
- **Global Coverage**: Based on ERA5 global reanalysis data, supports various resolutions from 64×32 to 512×512

## 📊 Data Description

![Data Example](assests/data_example.png)

### Data Source

- **Source**: WeatherBench2 - ERA5 reanalysis data
- **Path**: `gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr`
- **Resolution**: 64×32 equiangular grid (longitude × latitude)
- **Longitude Range**: [0.00, 354.38]
- **Latitude Range**: [-87.19, 87.19]
- **Time Interval**: 6 hours
- **Time Points**: 92044
- **Time Range**: 1959-01-01 to 2021-12-31

### Main Variables

| Variable Name | Description | Dimensions |
|--------------|-------------|------------|
| `2m_temperature` | 2-meter temperature | (time, lat, lon) |
| `geopotential` | Geopotential height | (time, level, lat, lon) |
| `10m_u_component_of_wind` | 10-meter U wind | (time, lat, lon) |
| `10m_v_component_of_wind` | 10-meter V wind | (time, lat, lon) |
| `specific_humidity` | Specific humidity | (time, level, lat, lon) |

### Data Format

```python
# Input sequence
X: (n_samples, input_length, features)
   input_length = 12  # Past 12 time steps (3 days)

# Output sequence  
Y: (n_samples, output_length, features)
   output_length = 4  # Future 4 time steps (1 day)
```

## 🏗️ Model Architecture

The project supports multiple models:

- **Linear Regression**: Fast baseline model
- **CNN**: Convolutional Neural Network, best performance (RMSE=1.20 K), fast training
- **ConvLSTM**: Spatiotemporal joint modeling, excellent performance (RMSE=1.24 K)
- **LSTM**: Time series modeling, suitable for single-point prediction
- **Transformer**: Sequence modeling, no spatiotemporal information, suitable for single-point prediction
- **Weather Transformer**: Factorized spatiotemporal attention, lightweight design

Based on Stable Diffusion architecture, treating meteorological data as images for prediction:

- **Pixel U-Net**: Direct prediction in pixel space, excellent performance (RMSE=1.25 K)
- **Latent U-Net**: 
  - **SD VAE**: Stable Diffusion pre-trained VAE (512×512→64×64)
  - **RAE**: Representation Autoencoder, supports multiple encoders (DINOv2/SigLIP2/MAE)
- **Diffusion Model**: Probabilistic prediction, supports uncertainty quantification

**For detailed model architecture, please refer to [MODEL_EN.md](MODEL_EN.md)**

## 🔬 Latent Codec Reconstruction Testing

Before using Latent U-Net for weather prediction, it is recommended to test the reconstruction capabilities of different Latent Codecs (encoder-decoder) to select the codec most suitable for weather data.

We provide complete reconstruction testing tools supporting:

- **VAE (Stable Diffusion VAE)**: Pre-trained SD VAE
- **RAE (Representation Autoencoder)**: Supports multiple encoders (DINOv2-B, MAE, SigLIP2, etc.)
- **S-VAE (Hyperspherical VAE)**: Customizable trainable VAE

### 📊 Test Results Summary

Reconstruction test results based on 1460 test samples (2m_temperature, 256×256):

| Codec | RMSE (K) | Correlation | Recommendation |
|-------|----------|-------------|----------------|
| **RAE-MAE** ⭐ | **0.94** | **0.999** | ⭐⭐⭐⭐⭐ Best |
| **VAE** | 2.35 | 0.998 | ⭐⭐⭐⭐ Excellent |
| **RAE-MAE_decXL** | 4.44 | 0.994 | ⭐⭐⭐ Good |
| **RAE-DINOv2-B** | 4.92 | 0.972 | ⭐⭐⭐ Good |

**For detailed test results and complete metrics, please refer to [reconstruction/README.md](reconstruction/README.md)**

## 📈 Evaluation Metrics

### Deterministic Metrics
- **RMSE** (Root Mean Square Error): Root mean square error, main metric
- **MAE** (Mean Absolute Error): Mean absolute error

### Spatiotemporal Resolution Metrics

- **RMSE vs Lead Time**: Error variation with prediction step
- **Spatial Error Map**: Prediction accuracy in different regions
- **Time Series Plot**: Time series comparison between predictions and ground truth

## 🔧 Environment Setup

### Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install basic dependencies (including WeatherDiff dependencies)
pip install -r requirements.txt
```

## 🚀 Script Execution

### Traditional Models

```bash
# CNN (Recommended, best performance)
bash scripts/run_cnn.sh 2m_temperature

# ConvLSTM (Spatiotemporal modeling)
bash scripts/run_convlstm.sh 2m_temperature

# Weather Transformer
bash scripts/run_weather_transformer.sh 2m_temperature

# Pixel U-Net
bash scripts/run_pixel_unet.sh 2m_temperature

# VAE (SD) Latent U-Net
bash scripts/run_vae_latent_unet.sh 2m_temperature

# RAE Latent U-Net
bash scripts/run_rae_latent_unet.sh 2m_temperature

# Diffusion model (Probabilistic prediction)
bash scripts/run_diffusion.sh
```

**Note**:
- VAE/RAE Latent U-Net requires data preprocessing (scripts will handle automatically)
- For large images (512×512), it is recommended to use Latent U-Net to save memory
- Supports mixed precision training (`--use-amp --amp-dtype bfloat16`) and gradient accumulation (`--gradient-accumulation-steps 2`)

### Latent Codec Reconstruction Testing

Before using Latent U-Net, it is recommended to test the reconstruction capabilities of different codecs:

```bash
# Test VAE reconstruction
python reconstruction/test_vae_reconstruction.py \
    --data-path reconstruction/weather_images \
    --n-test-samples 100

# Test RAE reconstruction (batch test multiple encoders)
cd reconstruction
bash test_rae_reconstruction.sh

# Compare reconstruction effects of multiple codecs
python reconstruction/compare_reconstructions.py \
    --original-dir reconstruction/weather_images \
    --reconstructed-dirs \
        outputs/vae_reconstruction/reconstructed \
        outputs/rae_reconstruction/recon_samples_DINOv2-B/RAE-pretrained-bs4-fp32 \
    --labels VAE RAE-DINOv2-B \
    --output comparison.png
```

For detailed instructions, please refer to [reconstruction/README.md](reconstruction/README.md)

## 🔬 Experimental Results

All results are evaluation metrics in **physical space**. Three main meteorological variables are evaluated: 2-meter temperature, geopotential height, and specific humidity.

### 1. 2-meter Temperature (2m_temperature)

| Model | MSE | MAE (K) | RMSE (K) | Step 1 | Step 2 | Step 3 | Step 4 |
|-------|-----|---------|----------|--------|---------|--------|--------|
| **CNN** ⭐ | 1.446 | 0.753 | **1.203** | 0.768 | 1.061 | 1.309 | 1.535 |
| **ConvLSTM** | 1.542 | 0.758 | 1.242 | 0.736 | 1.091 | 1.365 | 1.604 |
| **Pixel U-Net** ⭐ | - | 0.783 | 1.252 | 0.775 | 1.128 | 1.382 | 1.578 |
| **Weather Transformer** | 1.821 | 0.863 | 1.349 | 0.952 | 1.225 | 1.455 | 1.662 |
| LSTM | 6.557 | 1.729 | 2.561 | 2.523 | 2.543 | 2.571 | 2.604 |
| Multi-Output LR | 7.128 | 1.756 | 2.670 | 2.227 | 2.549 | 2.802 | 3.034 |
| Transformer | 11.335 | 2.300 | 3.367 | 3.363 | 3.371 | 3.367 | 3.366 |
| Latent U-Net (SD-VAE) | - | 7.228 | 8.115 | 7.653 | 8.520 | 8.039 | 8.221 |
| Latent U-Net (RAE, MAE) | - | 8.755 | 17.257 | 17.177 | 17.306 | 17.227 | 17.317 |

### 2. Geopotential Height (geopotential)

| Model | MSE | MAE (m²/s²) | RMSE (m²/s²) | Step 1 | Step 2 | Step 3 | Step 4 |
|-------|-----|-------------|--------------|--------|---------|--------|--------|
| **CNN** ⭐ | 37938 | 123.2 | **194.8** | 89.5 | 140.8 | 211.0 | 281.7 |
| **Pixel U-Net** | - | 138.8 | 201.7 | 107.8 | 156.0 | 216.2 | 283.0 |
| **Weather Transformer** | 43920 | 140.5 | 209.6 | 117.6 | 165.9 | 225.9 | 288.6 |
| **ConvLSTM** | 48306 | 133.0 | 219.8 | 77.5 | 148.9 | 236.8 | 330.1 |
| Multi-Output LR | 400591 | 416.7 | 632.9 | 364.5 | 540.2 | 699.7 | 829.5 |
| LSTM | 485145 | 501.2 | 696.5 | 673.1 | 685.9 | 703.0 | 723.1 |
| Transformer | 602376 | 547.9 | 776.1 | 762.9 | 770.0 | 779.9 | 791.4 |
| Latent U-Net (SD-VAE) | - | 868.4 | 1011.7 | 758.4 | 888.6 | 1100.9 | 1231.7 |
| Latent U-Net (RAE, MAE) | - | 566.1 | 1268.2 | 1255.5 | 1258.9 | 1267.9 | 1290.3 |

### 3. Specific Humidity (specific_humidity)

| Model | MSE | MAE | RMSE | Step 1 | Step 2 | Step 3 | Step 4 |
|-------|-----|-----|------|--------|---------|--------|--------|
| **Pixel U-Net** ⭐ | - | 0.000223 | **0.000451** | 0.000357 | 0.000417 | 0.000480 | 0.000529 |
| **ConvLSTM** | 0.0 | 0.000341 | 0.000530 | 0.000294 | 0.000459 | 0.000588 | 0.000693 |
| **CNN** | 0.0 | 0.000362 | 0.000550 | 0.000332 | 0.000487 | 0.000608 | 0.000701 |
| Pixel U-Net (2) | - | 0.000365 | 0.000560 | 0.000327 | 0.000501 | 0.000622 | 0.000716 |
| **Weather Transformer** | 0.0 | 0.000381 | 0.000583 | 0.000357 | 0.000523 | 0.000645 | 0.000738 |
| Latent U-Net (SD-VAE) | - | 0.000575 | 0.000800 | 0.000564 | 0.000742 | 0.000853 | 0.000980 |
| LSTM | 0.000001 | 0.000780 | 0.001109 | 0.001103 | 0.001107 | 0.001111 | 0.001115 |
| Transformer | 0.000001 | 0.000786 | 0.001120 | 0.001119 | 0.001120 | 0.001120 | 0.001122 |
| Multi-Output LR | 0.000001 | 0.000840 | 0.001196 | 0.001046 | 0.001168 | 0.001250 | 0.001305 |

### Results Analysis

#### Best Model Summary

**2-meter Temperature**:
1. **CNN** (RMSE=1.20 K) - Best traditional model ⭐
2. **ConvLSTM** (RMSE=1.24 K) - Second best traditional model
3. **Pixel U-Net** (RMSE=1.25 K) - Best WeatherDiff model ⭐
4. **Weather Transformer** (RMSE=1.35 K) - ViT-based Transformer

**Geopotential Height**:
1. **CNN** (RMSE=194.8 m²/s²) - Best model ⭐
2. **Pixel U-Net** (RMSE=201.7 m²/s²) - Second best model
3. **Weather Transformer** (RMSE=209.6 m²/s²)
4. **ConvLSTM** (RMSE=219.8 m²/s²)

**Specific Humidity**:
1. **Pixel U-Net** (RMSE=0.000451) - Best model ⭐
2. **ConvLSTM** (RMSE=0.000530)
3. **CNN** (RMSE=0.000550)
4. **Weather Transformer** (RMSE=0.000583)

#### Key Findings

1. **Model Performance Comparison**:
   - **CNN** performs best on temperature and geopotential height prediction, indicating that simple CNN architecture is very effective at 64×32 resolution
   - **Pixel U-Net** performs best on specific humidity prediction and is close to optimal on temperature and geopotential height
   - **ConvLSTM** performs stably on all variables and is a reliable baseline
   - **Weather Transformer** performs moderately but outperforms traditional LSTM and Transformer on all variables

2. **WeatherDiff Module Performance**:
   - **Pixel U-Net** performs excellently, ranking in the top 3 on all three variables
   - **Latent U-Net (SD-VAE)** performs poorly on temperature prediction (RMSE=8.12 K) but acceptable on geopotential height and specific humidity
   - **Latent U-Net (RAE)** performs poorly on all variables and may require further tuning

3. **Lead Time Analysis**:
   - All models show increasing error with prediction step (Step 1 → Step 4)
   - Best models perform excellently on short-term prediction (6 hours):
     - Temperature: RMSE < 0.8 K
     - Geopotential height: RMSE < 90 m²/s²
     - Specific humidity: RMSE < 0.0004
   - Long-term prediction (24 hours) shows significant error growth but still acceptable

4. **Model Selection Recommendations**:
   - **Quick Deployment**: Choose CNN, fast training, optimal performance on temperature and geopotential height
   - **Balance Performance and Scalability**: Choose ConvLSTM or Pixel U-Net
   - **Specific Humidity Prediction**: Prioritize Pixel U-Net
   - **Large Images**: Choose Latent U-Net (SD-VAE), memory-friendly
   - **Long-range Dependencies**: Choose Weather Transformer

**Note**:
- Step 1-4 correspond to predictions for 6, 12, 18, and 24 hours in the future
- All metrics are calculated in physical space
- Temperature unit: Kelvin (K)
- Geopotential height unit: m²/s²
- Specific humidity is dimensionless

### Output Results

After training and prediction, results are saved in the `outputs/<model_name>/` directory:

```
outputs/<model_name>/
├── best_model.pt              # Best model weights
├── config.json               # Training configuration
├── training_history.json     # Training history
├── prediction_metrics.json  # Evaluation metrics
├── predictions_data/         # Prediction data (numpy format)
├── timeseries_*.png          # Time series comparison plots
├── spatial_comparison_*.png  # Spatial comparison plots
└── rmse_vs_leadtime_*.png    # RMSE vs prediction step plots
```

## 🔮 Future Work

Based on current experimental results and project progress, the following are potential future research directions and improvements:

### 1. S-VAE (Hyperspherical VAE) Exploration

- **Goal**: Explore the performance of S-VAE using hyperspherical distribution (von Mises-Fisher distribution) as latent space distribution on weather prediction tasks
- **Motivation**: Hyperspherical distribution may be more suitable for capturing periodic features in weather data (such as seasonal cycles, diurnal cycles, etc.)
- **Research Directions**:
  - Compare S-VAE with standard VAE and RAE on reconstruction and prediction tasks
  - Analyze the impact of hyperspherical latent space on weather data representation
  - Evaluate the impact of different latent space dimensions on performance
- **Reference**: For detailed implementation instructions, please refer to [reconstruction/SVAE_README.md](reconstruction/SVAE_README.md)

### 2. Deep Exploration of Diffusion Architecture

- **Goal**: Systematically evaluate the performance of Diffusion models on weather prediction tasks, especially probabilistic prediction and uncertainty quantification capabilities
- **Research Directions**:
  - Compare different Diffusion architectures (DDPM, DDIM, Latent Diffusion, etc.)
  - Evaluate probabilistic prediction quality (CRPS, Spread-Skill Ratio, etc.)
  - Explore conditional Diffusion models, using historical observations as conditions
  - Study the impact of different noise scheduling strategies on prediction performance
  - Evaluate Diffusion model performance on extreme weather event prediction

### 3. Codec Upper Limit Capability Verification

- **Goal**: Verify the maximum reconstruction capability limits of different codecs (VAE, RAE, S-VAE) through overfitting experiments
- **Research Methods**:
  - Perform overfitting training on small-scale datasets to observe reconstruction error lower bounds
  - Analyze theoretical capacity and actual performance of different codecs
  - Evaluate the impact of latent space dimensions on reconstruction capability
  - Compare optimal performance of different codecs under the same conditions
- **Significance**: Determine performance upper limits of each codec, providing theoretical basis for model selection

### 4. Multi-variable Relationship Exploration

- **Current Status**: Currently focusing on single-variable prediction (temperature, geopotential height, specific humidity, etc., predicted independently)
- **Future Directions**:
  - **Multi-variable Joint Prediction**: Simultaneously predict multiple related variables, utilizing physical relationships between variables
  - **Inter-variable Dependency Modeling**: Explore how to explicitly model physical constraints between variables (such as temperature-geopotential height relationships, wind field-pressure field relationships, etc.)
  - **Multi-variable Latent Codec**: Design codecs capable of encoding multiple variables simultaneously, maintaining variable relationships in latent space
  - **Physical Constraint Integration**: Introduce physical laws as constraints (such as mass conservation, energy conservation, etc.)
  - **Variable Importance Analysis**: Study the contribution of different variables to prediction tasks

### 5. More Model Architecture Exploration

- **Graph Neural Networks (GNN)**:
  - Model global grid as graph structure, use GNN to capture spatial relationships
  - Particularly suitable for handling irregular grids and local features
- **Neural ODE/Neural SDE**:
  - Model weather evolution as continuous dynamical systems
  - May be more suitable for long-term prediction and physical consistency
- **Hybrid Architectures**:
  - Combine advantages of different architectures (CNN, Transformer, Diffusion, etc.)
  - Explore multi-scale feature fusion strategies
- **Attention Mechanism Improvements**:
  - Explore efficient attention mechanisms (Factorized Attention, Linear Attention, etc.)
  - Study optimal design of spatiotemporal attention in weather prediction
- **Memory-Augmented Networks**:
  - Introduce external memory modules to store long-term weather patterns
  - May help capture seasonality and long-term trends

### 6. Other Research Directions

#### 6.1 Longer Prediction Time Range
- Extend from current 1 day (4 time steps) to 3-7 days or even longer time ranges
- Study error accumulation and pattern decay in long-term prediction

#### 6.2 Higher Resolution Support
- Extend from current 64×32 resolution to 128×64, 256×128, or even 512×256
- Evaluate computational efficiency and performance improvements at high resolution

#### 6.3 Ensemble Learning and Model Fusion
- Explore ensemble prediction of multiple models
- Study fusion strategies for predictions from different models
- Evaluate improvements in prediction stability and accuracy from ensemble methods

#### 6.4 Physical Constraints and Interpretability
- Introduce physical constraint loss functions to ensure predictions comply with physical laws
- Develop interpretability tools to analyze weather patterns learned by models
- Visualize key regions and features that models focus on

#### 6.5 Specialized Modeling for Extreme Weather Events
- Design specialized models for extreme weather events (typhoons, heavy rain, cold waves, etc.)
- Study training strategies under imbalanced data
- Evaluate model performance on extreme event prediction

#### 6.6 Real-time Prediction and Online Learning
- Explore online learning strategies to adapt models to new observations
- Study incremental learning and continual learning applications in weather prediction
- Optimize inference speed to support real-time prediction needs

#### 6.7 Cross-dataset Generalization
- Evaluate model generalization on different datasets (different time ranges, different regions, etc.)
- Study domain adaptation and transfer learning strategies
- Explore few-shot learning applications in weather prediction

## 📚 References

### Data and Benchmarks
- [WeatherBench2](https://weatherbench2.readthedocs.io/) - Weather prediction benchmark
- [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) - ECMWF reanalysis data

### Model Papers
- [ConvLSTM](https://arxiv.org/abs/1506.04214) - Shi et al., 2015
- [Transformer](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [U-Net](https://arxiv.org/abs/1505.04597) - Ronneberger et al., 2015
- [DDPM](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
- [Stable Diffusion](https://arxiv.org/abs/2112.10752) - Rombach et al., 2022
- [RAE](https://arxiv.org/abs/2510.11690) - Boyang Zheng et al., 2025

## 📧 Contact

If you encounter any issues or have suggestions, please feel free to open an Issue or PR.

---

For more detailed model architecture information, please refer to [MODEL_EN.md](MODEL_EN.md)


# Chronos Architecture Documentation

## System Overview

Chronos is designed as a modular, scalable ML platform for productivity forecasting. The architecture follows best practices for ML engineering with clear separation of concerns.

## Component Architecture

### 1. Data Layer (`chronos/data/`)

**Responsibilities:**
- Data loading and parsing
- Preprocessing and normalization
- Sliding window generation

**Key Classes:**
- `DataLoader`: Loads time-series data from CSV
- `Preprocessor`: Handles resampling, filling, normalization

**Data Flow:**
```
CSV → DataLoader → Preprocessor → Normalized Series → Sliding Windows
```

### 2. Feature Engineering (`chronos/features/`)

**Responsibilities:**
- Behavioral feature extraction
- Vector embeddings generation
- Feature normalization

**Key Classes:**
- `FeatureExtractor`: Extracts statistical, trend, seasonality features
- `EmbeddingGenerator`: Generates semantic embeddings using sentence transformers

**Features Extracted:**
- Statistical: mean, std, min, max, median, quartiles
- Trend: slope, intercept, R², strength
- Seasonality: dominant frequency, spectral entropy
- Volatility: coefficient of variation, mean absolute change
- Autocorrelation: lag-1 correlation

### 3. Model Layer (`chronos/models/`)

**Responsibilities:**
- Model definitions
- Ensemble creation
- Model inference

**Models Implemented:**

#### LSTM (`lstm_model.py`)
- Multi-layer LSTM with dropout
- Configurable hidden size and layers
- Single-step or multi-step forecasting

#### TCN (`tcn_model.py`)
- Temporal Convolutional Network
- Dilated causal convolutions
- Residual connections

#### Transformer (`transformer_model.py`)
- Multi-head self-attention
- Positional encoding
- Configurable depth and width

#### Ensemble (`ensemble.py`)
- Weighted average of multiple models
- Median ensemble option
- Stacking support (with meta-model)

### 4. Training Layer (`chronos/training/`)

**Responsibilities:**
- Model training orchestration
- MLflow integration
- Early stopping and checkpointing

**Key Classes:**
- `Trainer`: Handles training loop, validation, logging
- `MLflowTracker`: Experiment tracking and model versioning

**Training Flow:**
```
Data → Dataset → DataLoader → Trainer → Model Checkpoint
                                      ↓
                                 MLflow Tracking
```

### 5. Evaluation Layer (`chronos/evaluation/`)

**Responsibilities:**
- Metric calculation
- Baseline benchmarking
- Model explainability

**Key Classes:**
- `BenchmarkRunner`: Runs ARIMA, Prophet, RNN baselines
- `ModelExplainer`: SHAP-based explainability

**Metrics:**
- Accuracy: RMSE, MAE, MAPE, R², directional accuracy
- Stability: Coefficient of variation
- Latency: Prediction time
- Composite: Weighted combination

### 6. Recommendation Engine (`chronos/recommendations/`)

**Responsibilities:**
- Trend detection
- Bottleneck identification
- Personalized recommendations
- User clustering

**Key Classes:**
- `RecommendationEngine`: Generates coaching recommendations
- `UserClustering`: Clusters users by productivity patterns

**Recommendation Flow:**
```
Time Series → Trend Detection → Bottleneck Analysis → Recommendation Generation
                    ↓
            User Clustering (optional)
```

### 7. API Layer (`chronos/api/`)

**Responsibilities:**
- RESTful API endpoints
- Request/response validation
- Response caching
- Async processing

**Endpoints:**
- `/forecast`: Multi-horizon forecasting
- `/patterns`: Trend and anomaly detection
- `/recommend`: Personalized recommendations
- `/explain`: Model explainability

**Architecture:**
```
FastAPI App → Route Handlers → Business Logic → Models → Response
                    ↓
                Cache Layer
```

### 8. Frontend (`web/`)

**Responsibilities:**
- Interactive visualizations
- User interface
- API integration

**Features:**
- Forecast visualization with confidence intervals
- Radar charts for behavioral fingerprints
- Daily productivity heatmaps
- Trend analysis plots
- Recommendation display
- Explainability panels

## Data Flow

### Training Pipeline

```
1. Load Data (CSV)
   ↓
2. Preprocess (resample, normalize)
   ↓
3. Create Sliding Windows
   ↓
4. Split Train/Val
   ↓
5. Train Model (with MLflow tracking)
   ↓
6. Evaluate & Save Checkpoint
```

### Inference Pipeline

```
1. API Request
   ↓
2. Check Cache
   ↓
3. Preprocess Input Window
   ↓
4. Model Inference
   ↓
5. Post-process Predictions
   ↓
6. Generate Response (cache result)
```

### Recommendation Pipeline

```
1. Load User Time Series
   ↓
2. Extract Features
   ↓
3. Detect Trend
   ↓
4. Identify Bottlenecks
   ↓
5. Match User Cluster (optional)
   ↓
6. Generate Recommendations
   ↓
7. Create Multi-Day Plan (optional)
```

## Model Architecture Details

### LSTM Architecture

```
Input (batch, seq_len, 1)
    ↓
LSTM Layer 1 (hidden_size=64)
    ↓
LSTM Layer 2 (hidden_size=64)
    ↓
Dropout (0.1)
    ↓
Linear (hidden_size → output_size)
    ↓
Output (batch, output_size)
```

### TCN Architecture

```
Input (batch, seq_len, 1)
    ↓
Temporal Block 1 (dilation=1)
    ↓
Temporal Block 2 (dilation=2)
    ↓
Temporal Block 3 (dilation=4)
    ↓
Linear (channels → output_size)
    ↓
Output (batch, output_size)
```

### Transformer Architecture

```
Input (batch, seq_len, 1)
    ↓
Input Projection (1 → d_model)
    ↓
Positional Encoding
    ↓
Transformer Encoder (n_layers=4, nhead=8)
    ↓
Output Projection (d_model → output_size)
    ↓
Output (batch, output_size)
```

## Caching Strategy

- **Cache TTL**: 300 seconds (5 minutes)
- **Cache Key**: MD5 hash of request parameters
- **Cache Scope**: Per endpoint and parameters
- **Invalidation**: Time-based (TTL)

## Scalability Considerations

### Horizontal Scaling
- Stateless API design
- Model loading per worker
- Shared cache (Redis in production)

### Performance Optimization
- Async endpoints for I/O-bound operations
- Batch inference for multiple requests
- Model quantization for faster inference
- Response caching to reduce computation

### Resource Management
- Model lazy loading
- Memory-efficient data loaders
- Gradient accumulation for large batches

## Security Considerations

- Input validation via Pydantic
- Rate limiting (to be added)
- Authentication/authorization (to be added)
- Secure model storage

## Monitoring and Observability

- MLflow for experiment tracking
- API metrics (latency, throughput)
- Model performance metrics
- Error logging and tracking

## Future Enhancements

1. **Real-time Streaming**: Kafka integration for live data
2. **Distributed Training**: Multi-GPU support
3. **Model Serving**: TorchServe or Triton
4. **A/B Testing**: Framework for model comparison
5. **AutoML**: Automated hyperparameter tuning
6. **Multi-tenancy**: User isolation and quotas


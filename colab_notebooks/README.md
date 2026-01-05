# LLMRouter Jupyter Notebooks

This directory contains Jupyter notebooks for training and inference of all LLMRouter methods on Colab.

## Directory Structure

```
colab_notebooks/
├── custom_router/             # Custom Router Tutorial
│   └── 01_creating_custom_routers.ipynb
├── data_preparation/          # Shared data preparation
│   └── 01_data_preparation.ipynb
├── knnrouter/                 # K-Nearest Neighbors Router
│   └── 01_knnrouter_training_inference.ipynb
├── svmrouter/                 # Support Vector Machine Router
│   └── 01_svmrouter_training_inference.ipynb
├── mlprouter/                 # Multi-Layer Perceptron Router
│   └── 01_mlprouter_training_inference.ipynb
├── mfrouter/                  # Matrix Factorization Router
│   └── 01_mfrouter_training_inference.ipynb
├── dcrouter/                  # Dual Contrastive Router
│   └── 01_dcrouter_training_inference.ipynb
├── graphrouter/               # Graph Neural Network Router
│   └── 01_graphrouter_training_inference.ipynb
├── causallm_router/           # Causal Language Model Router
│   └── 01_causallm_router_training_inference.ipynb
├── automix_router/            # Automatic LLM Mixing Router
│   └── 01_automix_router_training_inference.ipynb
├── hybrid_llm_router/         # Hybrid LLM Router
│   └── 01_hybrid_llm_router_training_inference.ipynb
├── gmtrouter/                 # Graph-based Multi-Turn Router
│   └── 01_gmtrouter_training_inference.ipynb
├── elorouter/                 # Elo Rating Router (inference only)
│   └── 01_elorouter_inference.ipynb
├── smallest_llm/              # SmallestLLM Baseline (inference only)
│   └── 01_smallest_llm_inference.ipynb
├── largest_llm/               # LargestLLM Baseline (inference only)
│   └── 01_largest_llm_inference.ipynb
├── router_r1/                 # RouterR1 Agentic Router (inference only)
│   └── 01_router_r1_inference.ipynb
├── knnmultiroundrouter/       # KNN Multi-Round Router
│   └── 01_knnmultiroundrouter_training_inference.ipynb
└── llmmultiroundrouter/       # LLM Multi-Round Router (inference only)
    └── 01_llmmultiroundrouter_inference.ipynb
```

## Router Comparison

| Router | Type | Training Required | GPU Required | Best For |
|--------|------|-------------------|--------------|----------|
| KNNRouter | Classification | Yes | No | Simple, interpretable routing |
| SVMRouter | Classification | Yes | No | High-dimensional data |
| MLPRouter | Neural Network | Yes | No | Complex decision boundaries |
| MFRouter | Matrix Factorization | Yes | No | Collaborative filtering |
| DCRouter | Transformer | Yes | Recommended | High accuracy routing |
| GraphRouter | GNN | Yes | Recommended | Relational patterns |
| CausalLMRouter | LLM Finetuning | Yes | Required | Complex queries |
| AutomixRouter | POMDP | Yes | No | Cost-efficient routing |
| HybridLLMRouter | MLP | Yes | No | Binary routing |
| GMTRouter | HeteroGNN | Yes | Recommended | Personalized multi-turn |
| EloRouter | Rating-based | No | No | Baseline/simple cases |
| SmallestLLM | Baseline | No | No | Cost-efficiency baseline |
| LargestLLM | Baseline | No | No | Performance upper bound |
| RouterR1 | Agentic | No | Required | Complex reasoning |
| KNNMultiRoundRouter | Multi-Round KNN | Yes | No | Multi-step queries |
| LLMMultiRoundRouter | Multi-Round LLM | No | Optional | Zero-shot multi-round |

## Using with Google Colab

All notebooks are compatible with Google Colab. To use:

1. Upload the notebook to Colab
2. Install dependencies:
   ```python
   !pip install llmrouter-lib transformers torch
   ```
3. Clone the repository (for data and configs):
   ```python
   !git clone https://github.com/ulab-uiuc/LLMRouter.git
   %cd LLMRouter
   ```
4. Run the notebook cells

## Configuration Files

Training configurations are located in:
- `configs/model_config_train/` - Training configurations
- `configs/model_config_test/` - Inference configurations

## API Keys

For full inference (calling LLM APIs), set environment variables:

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-key'
os.environ['ANTHROPIC_API_KEY'] = 'your-key'
# Or for multiple keys:
os.environ['API_KEYS'] = '["key1", "key2"]'
```

## Common Issues

### Out of Memory (OOM)
- Reduce `batch_size` in configuration
- Use CPU instead of GPU for smaller models
- Enable gradient checkpointing for large models

### Missing Data
- Run `01_data_preparation.ipynb` first
- Check data paths in configuration files
- Verify all required files exist

### Import Errors
- Install all dependencies: `pip install llmrouter-lib`
- Ensure you're in the correct directory
- Check Python path includes project root

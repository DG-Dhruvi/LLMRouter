---
title: LLMRouter
---

# LLMRouter: An Open-Source Library for LLM Routing

![LLMRouter](assets/llmrouter.png)

LLMRouter is an open-source routing system that selects the best large language model for each query based on capability, cost, and latency. It provides a unified CLI for training, inference, and interactive chat, plus a plugin system for custom routers.

## Key capabilities
- Smart routing across 16+ router families, from lightweight baselines to multi-round strategies
- Unified CLI for training, batch inference, and chat UI
- Plugin system for custom routers without modifying core code
- Example configs and datasets for reproducible experiments

## Quickstart
```bash
pip install llmrouter

llmrouter infer --router knnrouter --config configs/model_config_test/knnrouter.yaml --query "What is machine learning?"
```

## Next steps
- Getting started: installation, quickstart, config basics
- Learn LLMRouter: architecture and router families
- API Reference: CLI and Python APIs

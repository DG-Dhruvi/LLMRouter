# Installation

LLMRouter can be installed from PyPI, or from source if you want the example configs and data in the repository.

## Requirements
- Python 3.10+
- Optional: CUDA-capable GPU for faster training

!!! note
    The documentation site is built from the `website` branch, but the runnable code and example assets live on `main`.

## Option A: Install from PyPI

```bash
python -m pip install -U pip
python -m pip install llmrouter
```

## Option B: Install from source (recommended for examples)

```bash
git clone https://github.com/ulab-uiuc/LLMRouter.git
cd LLMRouter
```

### Create an environment

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install (editable)

```bash
python -m pip install -U pip
python -m pip install -e .
```

## Verify

```bash
llmrouter --version
llmrouter list-routers
python -c "import llmrouter; print(llmrouter.__version__)"
```

## API keys (only for real API calls)

`llmrouter infer` (without `--route-only`) calls an OpenAI-compatible endpoint via LiteLLM and requires `API_KEYS`.
It supports either a single key or a JSON list of keys (used round-robin).

**Single key**

```bash
export API_KEYS="YOUR_KEY"
```

```powershell
$env:API_KEYS="YOUR_KEY"
```

**Multiple keys (JSON list)**

```bash
export API_KEYS='["key1","key2","key3"]'
```

```powershell
$env:API_KEYS='["key1","key2","key3"]'
```

!!! warning
    Do not commit API keys to git. Use environment variables or a secret manager.

## Where does `api_endpoint` come from?

For inference, the API base URL is resolved in this order:

1. `llm_data[model_name].api_endpoint` in your `llm_data` JSON
2. `api_endpoint` in your router YAML config

If neither is set, inference fails with an "API endpoint not found" error.

## Next

- Continue with [Quickstart](quickstart.md).

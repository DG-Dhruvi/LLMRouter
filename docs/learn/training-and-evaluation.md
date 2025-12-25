# Training and evaluation

## Training workflow
1. Choose a router and config file.
2. Run `llmrouter train --router <name> --config <path>`.
3. The router loads config and data via `DataLoader`.
4. The trainer runs `train()` and writes to `model_path.save_model_path`.

`--device auto` will select GPU when available; use `--quiet` to reduce output.

## Inference workflow
1. Load a router with `llmrouter infer`.
2. Run routing only with `--route-only`, or perform full inference with API calls.
3. Use `--output` and `--output-format` for batch runs.

## Router availability
Not all routers are trainable. If a router is not in the training registry, `llmrouter train` will report it as unsupported. Use `llmrouter list-routers` to confirm availability.

## Evaluation options
- Use `--route-only` to evaluate routing decisions without calling APIs.
- Use full inference to evaluate end-to-end output quality.
- The `llmrouter/evaluation` package includes scripts and examples for evaluation.
- Optional evaluation helpers are exposed in `llmrouter.utils` when extra dependencies are installed.

## Artifacts and overrides
- Training checkpoints are written to `model_path.save_model_path`.
- For inference, `--load_model_path` overrides `model_path.load_model_path` in the config.

## Common pitfalls
- Missing data files in `data_path` will cause warnings or errors.
- Optional router dependencies may not be installed; check `llmrouter list-routers`.
- Full inference requires `API_KEYS` to be set for external API calls.
- `--route-only` returns routing decisions and does not call any LLM APIs.

## Reproducibility tips
- Version your config files along with data snapshots.
- Keep the router name, config, and model artifact together for each experiment.

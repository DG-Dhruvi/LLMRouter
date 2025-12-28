# Core classes

## MetaRouter
Base class for all routers.

Source: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/meta_router.py

### Signature
```python
class MetaRouter(torch.nn.Module, ABC):
    def __init__(self, model: nn.Module, yaml_path: str | None = None, resources=None):
        ...
```

### Summary
`MetaRouter` standardizes routing behavior and optionally loads config and data on initialization.

!!! note "Key responsibilities"
    - Hold the underlying model and shared resources
    - Provide `route_single` and `route_batch`
    - Load config and attach datasets via `DataLoader` when `yaml_path` is provided
    - Provide utility helpers such as `save_router` and `load_router`

### Common attributes
| Attribute | Type | Description |
| --- | --- | --- |
| `model` | `nn.Module` | Underlying routing model |
| `resources` | `Any` | Optional shared resources (tokenizer, env, etc.) |
| `cfg` | `dict` | YAML config (empty if not loaded) |
| `metric_weights` | `list[float]` | Metric weights from `cfg.metric.weights` |

### Parameters
| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `nn.Module` | Underlying routing model | required |
| `yaml_path` | `str or None` | YAML config path, loads data if provided | `None` |
| `resources` | `Any` | Optional shared resources | `None` |

### Methods
- `route_single(batch)`: route a single input (subclasses implement)
- `route_batch(batch)`: route a batch input (subclasses implement)
- `forward(batch)`: delegates to `route_batch` (PyTorch `nn.Module` API)
- `compute_metrics(outputs, batch) -> dict`: optional, defaults to `{}`
- `save_router(path)`: `torch.save(self.state_dict(), path)`
- `load_router(path)`: `torch.load(...); self.load_state_dict(...)`

!!! note "YAML + data loading side effects"
    If you pass `yaml_path`, `MetaRouter` loads the YAML into `self.cfg` and uses `DataLoader` to attach data fields directly onto the router instance (for example, `query_data_train`, `llm_data`).
    See [Data utilities](data.md) and [Config reference](config.md).

## BaseTrainer
Base class for router trainers.

Source: https://github.com/ulab-uiuc/LLMRouter/blob/main/llmrouter/models/base_trainer.py

### Signature
```python
class BaseTrainer(ABC):
    def __init__(self, router, optimizer=None, device="cuda", **kwargs):
        ...
```

### Summary
Trainers encapsulate the training loop for each router type.

### Parameters
| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `router` | `torch.nn.Module` | Router instance to train | required |
| `optimizer` | `torch.optim.Optimizer or None` | Optimizer for training | `None` |
| `device` | `str` | Device for training | `cuda` |
| `**kwargs` | `Any` | Extra trainer args | none |

### Methods
- `loss_func(outputs, batch) -> torch.Tensor`: define task-specific loss (subclasses should override)
- `train(dataloader: Any = None)`: define the full training loop (required)

### Example
```python
from llmrouter.models.meta_router import MetaRouter
from llmrouter.models.base_trainer import BaseTrainer

class MyRouter(MetaRouter):
    def route_single(self, query_input: dict) -> dict:
        return {"model_name": "model_a"}

    def route_batch(self, batch: list) -> list:
        return [self.route_single(q) for q in batch]

class MyTrainer(BaseTrainer):
    def loss_func(self, outputs, batch):
        ...

    def train(self, dataloader):
        ...
```

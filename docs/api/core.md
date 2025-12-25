# Core classes

## MetaRouter
Base class for all routers.

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

### Parameters
| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `nn.Module` | Underlying routing model | required |
| `yaml_path` | `str or None` | YAML config path, loads data if provided | `None` |
| `resources` | `Any` | Optional shared resources | `None` |

### Methods
- `route_single(batch)`
- `route_batch(batch)`
- `compute_metrics(outputs, batch)`
- `save_router(path)`
- `load_router(path)`

## BaseTrainer
Base class for router trainers.

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
- `loss_func(outputs, batch)`
- `train(dataloader)`

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

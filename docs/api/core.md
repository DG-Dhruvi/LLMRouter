# Core classes

## MetaRouter
Path: `llmrouter.models.meta_router.MetaRouter`

Constructor:
- `MetaRouter(model, yaml_path=None, resources=None)`

Required methods:
- `route_single(batch)`
- `route_batch(batch)`

Optional helpers:
- `compute_metrics(outputs, batch)`
- `save_router(path)`
- `load_router(path)`

If `yaml_path` is provided, the router loads config and uses `DataLoader` to attach datasets to the router instance.

## BaseTrainer
Path: `llmrouter.models.base_trainer.BaseTrainer`

Constructor:
- `BaseTrainer(router, optimizer=None, device="cuda", **kwargs)`

Required methods:
- `loss_func(outputs, batch)`
- `train(dataloader)`

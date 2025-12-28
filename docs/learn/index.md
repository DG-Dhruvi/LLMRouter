# Learn LLMRouter

This section explains how LLMRouter is put together and how to reason about routers, data, training, and deployment.

!!! note
    This docs site is built from the `website` branch. The runnable code, example configs, and example data live on the `main` branch.

## What you will learn
- How the CLI maps to the core Python modules
- How routers load configs and data
- How router families differ and when to use them
- How training and inference are wired
- How plugins extend the system
- How to deploy and operate routing workflows

## Reading map
- [Architecture](architecture.md): execution paths, core classes, and data flow
- [Router families](router-families.md): categories and trade-offs
- [Data and metrics](data-and-metrics.md): expected inputs and objective weights
- [Training and evaluation](training-and-evaluation.md): workflows, artifacts, and pitfalls
- [Plugin system](plugin-system.md): custom routers without forking
- [Deployment](deployment.md): inference, batch, and integration patterns

## Where to go next
- If you want hands-on steps: [Quickstart](../getting-started/quickstart.md) and [Tutorials](../tutorials/index.md)
- If you want a complete list of routers and links: [API Reference - Routers](../api/routers.md)
- If you want command flags: [API Reference - CLI](../api/cli.md)

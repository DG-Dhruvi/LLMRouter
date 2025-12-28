# API Reference

This section documents the public CLI and Python APIs exposed by the `llmrouter` package.

!!! note
    The docs site is built from the `website` branch, but the runnable implementation lives on `main`.
    Source links in this section point to `main` to reduce drift.

## Quick links
- [CLI](cli.md)
- [Core classes](core.md)
- [Routers](routers.md#router-table)
- [Config reference](config.md)
- [Plugin system](plugins.md)
- [Data utilities](data.md)
- [Utils](utils.md)

## Conventions
- CLI router names are lowercase strings (for example, `knnrouter`). See [Routers](routers.md) for aliases.
- Relative YAML paths resolve against the directory containing `llmrouter/`. See [Config reference](config.md).
- `llmrouter infer` reads/writes JSON. See [Data formats](../getting-started/data-formats.md) for schemas.

## Layout
Each page follows a consistent structure so you can scan quickly:
- Summary of the object or command
- Signature
- Key responsibilities
- Parameters and return values
- Examples and notes

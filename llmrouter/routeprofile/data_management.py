"""
Profile data management utilities.

Provides functions to initialise a user-owned profile data directory from
the bundled defaults, and to add/modify models, tasks, domains, and queries
in that directory without touching the bundled (read-only) data.
"""

from __future__ import annotations

import json
import os
import shutil
import warnings

from llmrouter.routeprofile.data import get_profile_data_dir

# ── Files that make up a complete profile data directory ─────────────────────

_PROFILE_FILES = [
    "model_feature_standard.json",
    "model_feature_newllm.json",
    "model_family_feature.json",
    "task_feature.json",
    "domain_feature.json",
    "domain_task_map.json",
    "task_queries_standard.json",
    "task_queries_newllm.json",
]

_MODE_TO_MODEL_FILES = {
    "standard": ["model_feature_standard.json"],
    "newllm":   ["model_feature_newllm.json"],
    "both":     ["model_feature_standard.json", "model_feature_newllm.json"],
}

_MODE_TO_QUERY_FILES = {
    "standard": ["task_queries_standard.json"],
    "newllm":   ["task_queries_newllm.json"],
    "both":     ["task_queries_standard.json", "task_queries_newllm.json"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _path(output_dir: str, filename: str) -> str:
    return os.path.join(output_dir, filename)


# ── Public API ────────────────────────────────────────────────────────────────

def init_profile_data_dir(output_dir: str) -> None:
    """Copy bundled profile data files into *output_dir*.

    Files that already exist in *output_dir* are left untouched so that
    user modifications are preserved across re-runs.
    """
    os.makedirs(output_dir, exist_ok=True)
    src_dir = get_profile_data_dir()
    copied, skipped = [], []
    for fname in _PROFILE_FILES:
        src = os.path.join(src_dir, fname)
        dst = _path(output_dir, fname)
        if not os.path.exists(src):
            warnings.warn(
                f"Bundled file not found, skipping: {src}",
                UserWarning, stacklevel=2,
            )
            continue
        if os.path.exists(dst):
            skipped.append(fname)
        else:
            shutil.copy2(src, dst)
            copied.append(fname)
    if copied:
        print(f"[profile] Initialised {output_dir}: copied {len(copied)} file(s).")
    if skipped:
        print(f"[profile] Kept existing file(s) in {output_dir}: {', '.join(skipped)}")


def add_domain(name: str, feature: str, output_dir: str) -> None:
    """Add or modify a domain entry.

    Writes *name → feature* into ``domain_feature.json`` and ensures
    *name* exists as a key (initialised to ``[]``) in ``domain_task_map.json``.
    If the domain already exists, only the feature text is updated.

    Args:
        name:       Domain key (e.g. ``"multimodal"``).
        feature:    Text description used as the domain node's Longformer feature.
        output_dir: Directory containing the profile JSON files.
    """
    init_profile_data_dir(output_dir)

    # domain_feature.json
    feat_path = _path(output_dir, "domain_feature.json")
    feat = _load(feat_path)
    action = "Modified" if name in feat else "Added"
    feat[name] = feature
    _save(feat, feat_path)

    # domain_task_map.json — initialise empty list only on Add
    map_path = _path(output_dir, "domain_task_map.json")
    dmap = _load(map_path)
    if name not in dmap:
        dmap[name] = []
        _save(dmap, map_path)

    print(f"[profile] {action} domain '{name}' in {output_dir}.")


def add_task(
    name: str,
    feature: str,
    output_dir: str,
    *,
    domains: list[str] | None = None,
    queries: list[str] | None = None,
    queries_mode: str | None = None,
) -> None:
    """Add or modify a benchmark/task entry.

    Writes *name → feature* into ``task_feature.json``. If *domains* is
    provided, appends *name* to each domain's list in ``domain_task_map.json``.
    If *queries* is provided, appends them to the appropriate
    ``task_queries_{mode}.json`` via :func:`add_query`.

    Every domain in *domains* **must** already exist in ``domain_feature.json``;
    if any are missing, a ``ValueError`` is raised and no files are modified.

    Args:
        name:         Benchmark key matching keys used in ``model detailed_scores``.
        feature:      Text description for the dataset node's Longformer feature.
        output_dir:   Directory containing the profile JSON files.
        domains:      Domain(s) this benchmark belongs to (optional).
        queries:      Representative query strings to associate with this task.
        queries_mode: Which query file(s) to update — ``"standard"``,
                      ``"newllm"``, or ``"both"``. Required when *queries* is
                      provided.
    """
    if queries and queries_mode is None:
        raise ValueError(
            "'queries_mode' is required when 'queries' are provided. "
            "Pass mode='standard', 'newllm', or 'both'."
        )

    init_profile_data_dir(output_dir)

    # Validate domains up-front — fail before writing anything
    if domains:
        feat_path = _path(output_dir, "domain_feature.json")
        known_domains = set(_load(feat_path).keys())
        missing = [d for d in domains if d not in known_domains]
        if missing:
            cmds = "\n  ".join(
                f"llmrouter profile add-domain --name {d!r} "
                f"--feature '...' --output-dir {output_dir}"
                for d in missing
            )
            raise ValueError(
                f"Domain(s) not found: {missing}.\n"
                f"Run first:\n  {cmds}"
            )

    # task_feature.json
    task_path = _path(output_dir, "task_feature.json")
    tasks = _load(task_path)
    action = "Modified" if name in tasks else "Added"
    tasks[name] = feature
    _save(tasks, task_path)

    # domain_task_map.json
    if domains:
        map_path = _path(output_dir, "domain_task_map.json")
        dmap = _load(map_path)
        for d in domains:
            if name not in dmap[d]:
                dmap[d].append(name)
        _save(dmap, map_path)

    domain_info = f" (domains: {domains})" if domains else ""
    print(f"[profile] {action} task '{name}'{domain_info} in {output_dir}.")

    # Append queries if provided (task is already written, so it's valid)
    if queries:
        add_query(name, queries, output_dir, mode=queries_mode)


def add_query(
    task_name: str,
    queries: list[str],
    output_dir: str,
    *,
    mode: str,
) -> None:
    """Append query strings to ``task_queries_{mode}.json`` for a given task.

    Duplicate query strings (exact match) are silently skipped so the
    operation is idempotent. If *task_name* is not present in the queries
    file it is initialised to an empty list before appending.

    Args:
        task_name:  Benchmark key — must already exist in ``task_feature.json``.
        queries:    List of query strings to append.
        output_dir: Directory containing the profile JSON files.
        mode:       Which file(s) to update: ``"standard"``, ``"newllm"``,
                    or ``"both"``.
    """
    if mode not in _MODE_TO_QUERY_FILES:
        raise ValueError(
            f"mode must be 'standard', 'newllm', or 'both'; got '{mode!r}'."
        )

    init_profile_data_dir(output_dir)

    # Validate task exists
    task_path = _path(output_dir, "task_feature.json")
    known_tasks = set(_load(task_path).keys())
    if task_name not in known_tasks:
        raise ValueError(
            f"Task '{task_name}' not found in task_feature.json. "
            f"Run 'llmrouter profile add-task --name {task_name!r} ...' first."
        )

    target_files = _MODE_TO_QUERY_FILES[mode]
    for fname in target_files:
        fpath = _path(output_dir, fname)
        qdata = _load(fpath)
        existing = qdata.get(task_name, [])
        existing_set = set(existing)
        new_ones = [q for q in queries if q not in existing_set]
        if new_ones:
            existing.extend(new_ones)
            qdata[task_name] = existing
            _save(qdata, fpath)

    files_str = " + ".join(target_files)
    print(f"[profile] Appended queries to '{task_name}' in {output_dir} ({files_str}).")


def add_model(
    name: str,
    output_dir: str,
    *,
    feature: str | None = None,
    architecture: str | None = None,
    arch_feature: str | None = None,
    detailed_scores: dict[str, float | None] | None = None,
    new_tasks: list[dict] | None = None,
    size: str | None = None,
    parameters: float | None = None,
    model_id: str | None = None,
    service: str | None = None,
    api_endpoint: str | None = None,
    replace: bool = False,
    mode: str = "both",
) -> dict:
    """Add or modify a model entry in model_feature JSON file(s).

    **Add mode** (model not present): ``feature`` and ``architecture`` are
    required; all other fields are optional.

    **Modify mode** (model already present): only the explicitly provided
    arguments are merged into the existing entry; unspecified fields are
    left unchanged. Pass ``replace=True`` to overwrite the entire record.

    ``new_tasks`` (list of ``{name, feature, domain?}`` dicts) are written
    to ``task_feature.json`` *before* the model entry, so their benchmark
    keys are valid when score validation runs.

    Args:
        name:           Model key — must match the key in ``llm_data.json``.
        output_dir:     Directory containing the profile JSON files.
        feature:        Model text description (Longformer node feature).
        architecture:   Architecture class name (e.g. ``"LlamaForCausalLM"``).
        arch_feature:   Description for a *new* architecture node. Required
                        when *architecture* is not already in
                        ``model_family_feature.json``.
        detailed_scores: Benchmark scores dict.  Keys unknown to
                        ``task_feature.json`` trigger a UserWarning.
        new_tasks:      List of task dicts to register before the model.
                        Each dict: ``{"name": str, "feature": str,
                        "domain": str | None}``.
        size:           Human-readable size string (e.g. ``"13B"``).
        parameters:     Parameter count in billions.
        model_id:       Provider model identifier.
        service:        API service provider name.
        api_endpoint:   API endpoint URL.
        replace:        If ``True`` and the model already exists, replace the
                        entire record rather than merging.
        mode:           Which model feature file(s) to update: ``"standard"``,
                        ``"newllm"``, or ``"both"`` (default: ``"both"``).

    Returns:
        The final model entry dict that was written to disk.
    """
    if mode not in _MODE_TO_MODEL_FILES:
        raise ValueError(
            f"mode must be 'standard', 'newllm', or 'both'; got '{mode!r}'."
        )

    init_profile_data_dir(output_dir)

    # 1. Register new_tasks first (so their benchmark keys become valid)
    if new_tasks:
        for t in new_tasks:
            t_name = t.get("name")
            t_feature = t.get("feature")
            t_domain = t.get("domain")
            if not t_name or not t_feature:
                raise ValueError(
                    f"Each entry in new_tasks must have 'name' and 'feature'. Got: {t}"
                )
            add_task(
                t_name, t_feature, output_dir,
                domains=[t_domain] if t_domain else None,
            )

    # 2. Architecture validation / registration
    arch_path = _path(output_dir, "model_family_feature.json")
    arch_data = _load(arch_path)
    if architecture is not None and architecture not in arch_data:
        if arch_feature is None:
            raise ValueError(
                f"Architecture '{architecture}' not found in model_family_feature.json. "
                f"Provide arch_feature (--arch-feature) with a text description for this new architecture."
            )
        arch_data[architecture] = arch_feature
        _save(arch_data, arch_path)
        print(f"[profile] Added architecture '{architecture}' to model_family_feature.json.")

    # 3. Score validation (warn on unknown benchmarks)
    if detailed_scores:
        task_path = _path(output_dir, "task_feature.json")
        known_tasks = set(_load(task_path).keys())
        unknown = [k for k in detailed_scores if k not in known_tasks]
        if unknown:
            warnings.warn(
                f"Benchmark(s) not in task_feature.json: {unknown}. "
                f"These scores will be stored but will NOT create graph edges. "
                f"Run `llmrouter profile add-task` first if you want graph edges.",
                UserWarning, stacklevel=2,
            )

    # 4. Warn if no scores provided (model node will be isolated from dataset nodes)
    if not detailed_scores:
        warnings.warn(
            f"No benchmark scores provided for '{name}'. "
            f"The model node will have no model↔dataset edges in the graph. "
            f"emb_gnn / flat profiles will still work via the text feature.",
            UserWarning, stacklevel=2,
        )

    # 5. Load file(s) and determine existing entry
    std_path = _path(output_dir, "model_feature_standard.json")
    newllm_path = _path(output_dir, "model_feature_newllm.json")

    std_data = _load(std_path) if mode in ("standard", "both") else None
    newllm_data = _load(newllm_path) if mode in ("newllm", "both") else None

    # is_new is determined from the primary file for the chosen mode
    if mode == "newllm":
        existing = newllm_data.get(name)
    else:  # standard or both — use standard as canonical
        existing = std_data.get(name)
    is_new = existing is None

    if is_new:
        # Add mode — feature and architecture are required
        if feature is None:
            raise ValueError(f"'feature' is required when adding a new model ('{name}').")
        if architecture is None:
            raise ValueError(f"'architecture' is required when adding a new model ('{name}').")
        entry: dict = {}
    elif replace:
        # Full replace — same requirements as Add
        if feature is None:
            raise ValueError(f"'feature' is required when using replace=True ('{name}').")
        if architecture is None:
            raise ValueError(f"'architecture' is required when using replace=True ('{name}').")
        entry = {}
    else:
        # Modify — start from existing, merge provided fields
        entry = dict(existing)

    # Merge / set top-level fields
    if feature is not None:
        entry["feature"] = feature
    if architecture is not None:
        entry["architecture"] = architecture
    if size is not None:
        entry["size"] = size
    if parameters is not None:
        entry["parameters"] = parameters
    if model_id is not None:
        entry["model"] = model_id
    if service is not None:
        entry["service"] = service
    if api_endpoint is not None:
        entry["api_endpoint"] = api_endpoint

    # Merge detailed_scores (patch individual benchmark entries)
    if detailed_scores is not None:
        existing_scores = entry.get("detailed_scores", {})
        existing_scores.update(detailed_scores)
        entry["detailed_scores"] = existing_scores

    # 6. Write to file(s) based on mode
    if std_data is not None:
        std_data[name] = entry
        _save(std_data, std_path)
    if newllm_data is not None:
        newllm_data[name] = entry
        _save(newllm_data, newllm_path)

    files_str = " + ".join(f.replace("model_feature_", "").replace(".json", "")
                            for f in _MODE_TO_MODEL_FILES[mode])
    action = "Added" if is_new else ("Replaced" if replace else "Modified")
    print(f"[profile] {action} model '{name}' in {output_dir} ({files_str}).")
    return entry

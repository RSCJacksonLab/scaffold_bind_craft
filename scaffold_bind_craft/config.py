from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_settings(settings_path: str | Path, output_dir_override: str | Path | None = None) -> dict[str, Any]:
    path = Path(settings_path).expanduser().resolve()
    with path.open() as handle:
        settings = json.load(handle)

    settings["_settings_path"] = str(path)
    settings["_repo_root"] = str(path.parents[1] if path.parent.name == "settings" else path.parent)

    if output_dir_override is not None:
        settings["output_dir"] = str(Path(output_dir_override).expanduser().resolve())
    elif "output_dir" in settings:
        settings["output_dir"] = str(Path(settings["output_dir"]).expanduser().resolve())
    else:
        raise KeyError("settings must define output_dir, or pass --output-dir")

    Path(settings["output_dir"]).mkdir(parents=True, exist_ok=True)
    return settings


def require(settings: Mapping[str, Any], *keys: str) -> Any:
    cursor: Any = settings
    path: list[str] = []
    for key in keys:
        path.append(key)
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise KeyError(f"settings missing required key: {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


def stage_dir(settings: Mapping[str, Any], stage: str) -> Path:
    stages = settings.get("stages", {})
    configured = stages.get(stage, {}) if isinstance(stages, Mapping) else {}
    dirname = configured.get("dir", stage) if isinstance(configured, Mapping) else stage
    path = Path(str(settings["output_dir"])) / str(dirname)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base is not None:
        return (Path(base).expanduser().resolve() / path).resolve()
    return path.resolve()


def setting_path(settings: Mapping[str, Any], value: str | Path) -> Path:
    base = Path(str(settings.get("_settings_path", "."))).parent
    return resolve_path(value, base=base)

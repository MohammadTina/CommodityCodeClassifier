"""
Central configuration manager.
All components load their settings through this class.
"""

import os
import yaml
from pathlib import Path
from typing import Any


class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._configs: dict = {}
        self._load_all()

    def _load_all(self):
        """Load all yaml files from config directory."""
        if not self.config_dir.exists():
            raise FileNotFoundError(
                f"Config directory not found: {self.config_dir}"
            )
        for config_file in self.config_dir.glob("*.yaml"):
            name = config_file.stem
            with open(config_file) as f:
                content = yaml.safe_load(f)
                self._configs[name] = self._resolve_env_vars(content)

    def _resolve_env_vars(self, obj: Any) -> Any:
        """
        Recursively resolve ${ENV_VAR} placeholders in config values.
        Allows secrets like DB connection strings to live in environment
        rather than config files.
        """
        if isinstance(obj, str):
            if obj.startswith("${") and obj.endswith("}"):
                var_name = obj[2:-1]
                value = os.environ.get(var_name)
                if value is None:
                    raise EnvironmentError(
                        f"Required environment variable not set: {var_name}"
                    )
                return value
            return obj
        elif isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(i) for i in obj]
        return obj

    def get(self, config_name: str, key: str = None) -> Any:
        config = self._configs.get(config_name, {})
        if key:
            return config.get(key)
        return config

    @property
    def app(self) -> dict:
        return self._configs.get("app", {}).get("app", {})

    @property
    def llm(self) -> dict:
        return self._configs.get("llm", {}).get("llm", {})

    @property
    def retrieval(self) -> dict:
        return self._configs.get("retrieval", {}).get("retrieval", {})

    @property
    def embedding(self) -> dict:
        return self._configs.get("embedding", {}).get("embedding", {})

    @property
    def preprocessing(self) -> dict:
        return self._configs.get("preprocessing", {}).get("preprocessing", {})

    @property
    def database(self) -> dict:
        return self._configs.get("database", {}).get("database", {})

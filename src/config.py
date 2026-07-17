"""
Production configuration management utility.

Handles loading, validation, and management of configuration from JSON files.
Supports environment variable overrides for production deployments.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages application configuration with validation and environment overrides."""

    def __init__(self, config_path: Optional[str] = None, env_prefix: str = "AUTOMATION_"):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to JSON configuration file
            env_prefix: Prefix for environment variable overrides

        Raises:
            FileNotFoundError: If config_path does not exist
        """
        self.config_path = config_path
        self.env_prefix = env_prefix
        self.config: Dict[str, Any] = {}

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to JSON configuration file

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If JSON is invalid
        """
        config_path = str(config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    def apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides to configuration.

        Environment variables should follow pattern: AUTOMATION_SECTION_KEY=value
        Example: AUTOMATION_MODEL_STATE_SIZE=128
        """
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Parse environment variable name
                parts = key[len(self.env_prefix):].lower().split('_')
                if len(parts) < 2:
                    continue

                section = parts[0]
                config_key = '_'.join(parts[1:])

                if section not in self.config:
                    logger.warning(f"Section {section} not in configuration")
                    continue

                # Attempt type conversion
                try:
                    converted_value = self._convert_value(value)
                    self.config[section][config_key] = converted_value
                    logger.info(f"Overriding {section}.{config_key} = {converted_value}")
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to override {key}: {e}")

    @staticmethod
    def _convert_value(value: str) -> Any:
        """
        Convert string environment variable to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Converted value (int, float, bool, or str)
        """
        # Try boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # Try integer
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            section: Configuration section
            key: Configuration key within section
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        try:
            return self.config.get(section, {}).get(key, default)
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str, default: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Configuration section
            default: Default value if section not found

        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, default or {})

    def validate_required_fields(self, required: Dict[str, list]) -> None:
        """
        Validate that required configuration fields exist.

        Args:
            required: Dict mapping section names to required field lists

        Raises:
            ValueError: If required field is missing
        """
        for section, fields in required.items():
            if section not in self.config:
                raise ValueError(f"Required section missing: {section}")

            for field in fields:
                if field not in self.config[section]:
                    raise ValueError(
                        f"Required field missing: {section}.{field}"
                    )

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.config.copy()

    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to JSON file.

        Args:
            output_path: Path to save configuration

        Raises:
            OSError: If file write fails
        """
        try:
            output_path = str(output_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=2)

            logger.info(f"Configuration saved to {output_path}")
        except OSError as e:
            logger.error(f"Failed to save configuration: {e}")
            raise


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get or create global configuration manager.

    Args:
        config_path: Path to configuration file (only used on first call)

    Returns:
        ConfigManager instance
    """
    global _config_manager

    if _config_manager is None:
        default_path = config_path or os.environ.get('AUTOMATION_CONFIG', 'config/production.json')
        _config_manager = ConfigManager(default_path if os.path.exists(default_path) else None)
        _config_manager.apply_env_overrides()

    return _config_manager


def get_config(section: str, key: str, default: Any = None) -> Any:
    """
    Convenience function to get configuration value.

    Args:
        section: Configuration section
        key: Configuration key
        default: Default value

    Returns:
        Configuration value
    """
    manager = get_config_manager()
    return manager.get(section, key, default)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    config = ConfigManager("config/production.json")
    config.apply_env_overrides()

    print("Model configuration:")
    print(json.dumps(config.get_section("model"), indent=2))

    print("\nTraining configuration:")
    print(json.dumps(config.get_section("training"), indent=2))

"""
Sample utility functions for testing ingestion.
"""
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class FileInfo:
    """Information about a file."""
    path: str
    size: int
    extension: str


def find_files(directory: str, extension: str = ".py") -> List[str]:
    """
    Find all files with a given extension in a directory.

    Args:
        directory: Directory to search
        extension: File extension to filter by

    Returns:
        List of file paths
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files


def read_file_safely(filepath: str) -> Optional[str]:
    """
    Read a file with error handling.

    Args:
        filepath: Path to the file

    Returns:
        File contents or None if reading fails
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except (IOError, OSError, UnicodeDecodeError):
        return None


def get_file_info(filepath: str) -> Optional[FileInfo]:
    """
    Get information about a file.

    Args:
        filepath: Path to the file

    Returns:
        FileInfo object or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        return None

    return FileInfo(
        path=filepath,
        size=os.path.getsize(filepath),
        extension=os.path.splitext(filepath)[1]
    )


def batch_process(items: List[Any], processor: callable, batch_size: int = 10) -> List[Any]:
    """
    Process items in batches.

    Args:
        items: List of items to process
        processor: Function to apply to each batch
        batch_size: Size of each batch

    Returns:
        Flattened list of processed results
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = processor(batch)
        results.extend(batch_results)
    return results


class ConfigManager:
    """Manages application configuration."""

    def __init__(self, defaults: Optional[Dict[str, Any]] = None):
        """
        Initialize with optional defaults.

        Args:
            defaults: Default configuration values
        """
        self._config = defaults or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value

    def update(self, values: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        self._config.update(values)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self._config.copy()

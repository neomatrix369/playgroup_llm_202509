"""
Report writer utilities for consistent file output formatting.

Following Object Calisthenics:
- Single Responsibility: Handles report formatting only
- Small class: Minimal focused methods
- DRY: Eliminates repetitive header/footer patterns

Eliminates 30+ lines of boilerplate header/footer duplication.
"""

from pathlib import Path
from typing import TextIO, Union


class ReportWriter:
    """
    Single responsibility: Format and write structured reports.

    Replaces repetitive patterns:
    - f.write("=" * 80 + "\\n") → writer.section_header(title)
    - f.write("─" * 70 + "\\n") → writer.subsection_separator()
    """

    def __init__(self, file_handle: TextIO, width: int = 80):
        """
        Initialize report writer.

        Args:
            file_handle: Open file handle to write to
            width: Character width for headers/separators (default: 80)
        """
        self._file = file_handle
        self._width = width

    @classmethod
    def open(cls, file_path: Union[str, Path], width: int = 80) -> 'ReportWriter':
        """
        Open a file and return a ReportWriter (context manager compatible).

        Usage:
            with ReportWriter.open("report.log") as writer:
                writer.section_header("RESULTS")
                writer.write("Content here")
        """
        file_handle = open(file_path, 'w')
        return cls(file_handle, width)

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._file:
            self._file.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close file."""
        self.close()

    def section_header(self, title: str, char: str = "=") -> None:
        """
        Write a major section header.

        Example:
            ================================================================================
            🎯 SECTION TITLE
            ================================================================================
        """
        self._file.write(char * self._width + "\n")
        self._file.write(title + "\n")
        self._file.write(char * self._width + "\n")

    def section_separator(self, char: str = "=") -> None:
        """
        Write a section separator line.

        Example:
            ================================================================================
        """
        self._file.write(char * self._width + "\n")

    def subsection_header(self, title: str, width: int = 70) -> None:
        """
        Write a subsection header with separator.

        Example:
            📊 SUBSECTION TITLE
            ──────────────────────────────────────────────────────────────────────
        """
        self._file.write(title + "\n")
        self._file.write("─" * width + "\n")

    def subsection_separator(self, width: int = 70) -> None:
        """
        Write a subsection separator.

        Example:
            ──────────────────────────────────────────────────────────────────────
        """
        self._file.write("─" * width + "\n")

    def write(self, content: str) -> None:
        """Write content directly to file."""
        self._file.write(content)

    def writeln(self, content: str = "") -> None:
        """Write content with newline."""
        self._file.write(content + "\n")

    def blank_line(self) -> None:
        """Write a blank line."""
        self._file.write("\n")
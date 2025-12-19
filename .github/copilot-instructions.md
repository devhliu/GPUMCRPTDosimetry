# Copilot Agent Instructions (Python Code Style)

Use Python 3.10+.

## Style

- Follow PEP 8 and keep formatting consistent with nearby files.
- Prefer explicit, readable code over clever one-liners.
- Use 4-space indentation.
- Keep functions small and focused; avoid deep nesting.

## Imports

- Group imports in this order with a blank line between groups: standard library, third-party, local.
- Prefer `from __future__ import annotations` at the top of new/edited modules when type hints are used (match existing patterns).

## Typing

- Use type hints for public functions and non-trivial internal helpers.
- Prefer `dataclasses.dataclass` for configuration containers.
- Use `pathlib.Path` for filesystem paths.

## Errors and Logging

- Raise specific exceptions with actionable messages.
- Do not print from library code; keep side effects explicit.

## Tests

- When adding behavior, add or update `pytest` tests if the project already has tests for the area.

## Comments and Docs

- Do not add comments unless explicitly requested.
- Use clear names and docstrings for public APIs when needed; keep docstrings concise.


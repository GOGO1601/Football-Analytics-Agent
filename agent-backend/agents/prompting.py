"""
Load prompt templates from agents/prompts/<version>.md.

A version file holds one section per node, each starting with a level-2 heading:

    ## router
    ...text...

    ## synthesiser
    ...text...

Placeholders use $name (string.Template), so JSON examples with { } in the prompt need
no escaping. Every placeholder must be supplied; a missing one is an error rather than
silently leaving "$name" in the prompt.

Keeping prompts in versioned files (not in code) is what makes the A/B prompt
experiments later a configuration change: PROMPT_VERSION=v1 or v2.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_SECTION = re.compile(r"^##\s+(\S+)\s*$", re.MULTILINE)


class PromptError(KeyError):
    pass


def default_version() -> str:
    return os.environ.get("PROMPT_VERSION", "v1")


@lru_cache(maxsize=16)
def load_sections(version: str) -> dict[str, str]:
    path = PROMPTS_DIR / f"{version}.md"
    if not path.is_file():
        raise PromptError(f"Prompt version '{version}' not found at {path}")
    text = path.read_text(encoding="utf-8")
    matches = list(_SECTION.finditer(text))
    sections = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[m.group(1)] = text[m.end():end].strip()
    return sections


def render(section: str, version: str | None = None, **values: str) -> str:
    """Return the named section of a prompt version with $placeholders filled in."""
    v = version or default_version()
    sections = load_sections(v)
    if section not in sections:
        raise PromptError(f"Prompt '{v}' has no section '## {section}'. Found: {sorted(sections)}")
    try:
        return Template(sections[section]).substitute(**values)
    except KeyError as e:
        raise PromptError(f"Prompt '{v}/{section}' needs a value for ${e.args[0]}") from e
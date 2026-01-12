from pathlib import Path
from typing import Any

import yaml

PLAYBOOK_DIR = Path(__file__).parent


def list_playbooks() -> list[dict[str, Any]]:
    playbooks: list[dict[str, Any]] = []
    for path in sorted(PLAYBOOK_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        playbooks.append(
            {
                "id": data.get("id"),
                "version": data.get("version"),
                "title": data.get("title"),
            }
        )
    return playbooks

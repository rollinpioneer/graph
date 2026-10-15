from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import yaml


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path | str, delimiter: str | None = None) -> list[dict[str, str]]:
    p = Path(path)
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter or ("\t" if p.suffix == ".tsv" else ",")))


def write_csv(path: Path | str, rows: Iterable[dict], fieldnames: list[str] | None = None, delimiter: str = ",") -> None:
    materialized = list(rows)
    fields = list(fieldnames or [])
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def dump_yaml(path: Path | str, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_yaml(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def dump_json(path: Path | str, value: dict | list) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

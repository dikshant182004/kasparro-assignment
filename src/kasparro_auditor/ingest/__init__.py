"""Ingestion package: JSONL → SQLite with Pydantic validation.

Every row is validated against `schemas/input_contract.json`. Invalid
rows go to `rejected_records` so one bad record never kills the batch
(brief requirement).
"""
from kasparro_auditor.ingest.loader import import_pack
from kasparro_auditor.ingest.normalize import normalize_url

__all__ = ["import_pack", "normalize_url"]

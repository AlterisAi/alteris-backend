"""Alteris evaluation framework — stage-by-stage data quality checks.

Modules:
  checks  — Automated completeness checks (Stage 0a queries codified)
  golden  — Golden store CRUD (JSONL files in eval/golden/)
  sampler — Stratified sampling from production tables
  reviewer — Interactive CLI review tool
  runner  — Stage runner (re-run stages with golden data)
  stats   — Statistics and comparison reporting
"""

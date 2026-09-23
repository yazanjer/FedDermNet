"""Aggregated report helpers (tables + figures from ``results/``)."""

from src.report.figures import generate_report_figures
from src.report.tables import write_report_tables

__all__ = ["generate_report_figures", "write_report_tables"]

"""Example Pydantic schemas for CSV validation."""

from datetime import date
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, Field


class FinancialRow(BaseModel):
    """Example schema for financial CSV data (revenue, income, etc.)."""

    date: date
    revenue: Annotated[Decimal | None, Field(default=None)] = None
    net_income: Annotated[Decimal | None, Field(default=None)] = None
    total_assets: Annotated[Decimal | None, Field(default=None)] = None
    total_liabilities: Annotated[Decimal | None, Field(default=None)] = None

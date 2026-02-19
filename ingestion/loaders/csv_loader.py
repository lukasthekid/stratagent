"""CSV document loader with pandas and schema validation."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError

from ingestion.models import Document, DocumentMetadata


class CSVLoader:
    """Load CSV files (financials, tabular data) with optional schema validation."""

    def __init__(
        self,
        path: str | Path,
        *,
        company_name: str | None = None,
        source_date: datetime | None = None,
        schema: type[BaseModel] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        """Initialize CSV loader.

        Args:
            path: Path to the CSV file.
            company_name: Optional company name for metadata.
            source_date: Optional document date (defaults to file mtime).
            schema: Optional Pydantic model for row-level validation.
            encoding: File encoding (default: utf-8).
        """
        self.path = Path(path)
        self.company_name = company_name
        self.source_date = source_date
        self.schema = schema
        self.encoding = encoding

    def load(self) -> list[Document]:
        """Load CSV, validate schema if provided, and return as Documents."""
        if not self.path.exists():
            raise FileNotFoundError(f"CSV not found: {self.path}")

        date = self.source_date
        if date is None:
            try:
                mtime = self.path.stat().st_mtime
                date = datetime.fromtimestamp(mtime)
            except OSError:
                date = None

        df = pd.read_csv(self.path, encoding=self.encoding)

        if self.schema is not None:
            df = self._validate_schema(df)

        content = self._to_content(df)
        metadata = DocumentMetadata(
            source=str(self.path),
            date=date,
            company_name=self.company_name,
            extra={
                "rows": len(df),
                "columns": list(df.columns),
            },
        )
        return [Document(content=content, metadata=metadata)]

    def _validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate each row against the schema; raise on first invalid row."""
        valid_rows: list[dict[str, Any]] = []
        errors: list[str] = []

        for i, row in df.iterrows():
            row_dict = self._row_to_dict(row)
            try:
                validated = self.schema.model_validate(row_dict)
                valid_rows.append(validated.model_dump())
            except ValidationError as e:
                errors.append(f"Row {i}: {e}")

        if errors:
            raise ValueError(
                f"Schema validation failed for {len(errors)} row(s). "
                f"First error: {errors[0]}"
            )

        return pd.DataFrame(valid_rows)

    def _row_to_dict(self, row: pd.Series) -> dict[str, Any]:
        """Convert pandas row to dict, replacing NaN with None."""
        return {k: (None if pd.isna(v) else v) for k, v in row.items()}

    def _to_content(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to text content for Document."""
        return df.to_csv(index=False)


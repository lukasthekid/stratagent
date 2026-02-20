import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    data: dict = Field(description="Financial data as a dictionary")
    calculations: list = Field(
        description="List of calculations: growth_rate, gross_margin, net_margin, current_ratio, debt_to_equity"
    )


class FinancialCalculatorTool(BaseTool):
    name: str = "Financial Calculator Tool"
    description: str = """
    Use this tool to perform financial calculations on raw data.
    Supported calculations:
    - growth_rate: needs 'current' and 'previous' in data
    - gross_margin: needs 'revenue' and 'cogs'
    - net_margin: needs 'revenue' and 'net_income'
    - current_ratio: needs 'current_assets' and 'current_liabilities'
    - debt_to_equity: needs 'total_debt' and 'equity'
    Always use this tool instead of estimating financial ratios manually.
    """
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, data: dict, calculations: list) -> str:
        if not isinstance(data, dict):
            return "Error: data must be a dictionary."
        if not isinstance(calculations, list):
            return "Error: calculations must be a list."

        results = {}
        errors = []

        def safe_div(num: float, denom: float) -> float | None:
            if denom == 0:
                return None
            return num / denom

        try:
            if "growth_rate" in calculations:
                if "current" in data and "previous" in data:
                    prev = float(data["previous"])
                    curr = float(data["current"])
                    growth = safe_div(curr - prev, prev)
                    results["growth_rate"] = f"{growth * 100:.2f}%" if growth is not None else "N/A (div by zero)"
                else:
                    errors.append("growth_rate needs 'current' and 'previous'")

            if "gross_margin" in calculations:
                if "revenue" in data and "cogs" in data:
                    rev = float(data["revenue"])
                    margin = safe_div(rev - float(data["cogs"]), rev)
                    results["gross_margin"] = f"{margin * 100:.2f}%" if margin is not None else "N/A"
                else:
                    errors.append("gross_margin needs 'revenue' and 'cogs'")

            if "net_margin" in calculations:
                if "revenue" in data and "net_income" in data:
                    rev = float(data["revenue"])
                    margin = safe_div(float(data["net_income"]), rev)
                    results["net_margin"] = f"{margin * 100:.2f}%" if margin is not None else "N/A"
                else:
                    errors.append("net_margin needs 'revenue' and 'net_income'")

            if "current_ratio" in calculations:
                if "current_assets" in data and "current_liabilities" in data:
                    ratio = safe_div(
                        float(data["current_assets"]),
                        float(data["current_liabilities"]),
                    )
                    results["current_ratio"] = f"{ratio:.2f}" if ratio is not None else "N/A (div by zero)"
                else:
                    errors.append("current_ratio needs 'current_assets' and 'current_liabilities'")

            if "debt_to_equity" in calculations:
                if "total_debt" in data and "equity" in data:
                    ratio = safe_div(float(data["total_debt"]), float(data["equity"]))
                    results["debt_to_equity"] = f"{ratio:.2f}" if ratio is not None else "N/A (div by zero)"
                else:
                    errors.append("debt_to_equity needs 'total_debt' and 'equity'")

        except (TypeError, ValueError) as e:
            return f"Calculation error: {e}. Ensure numeric values in data."

        output = json.dumps(results, indent=2)
        if errors:
            output += "\n\nMissing inputs: " + "; ".join(errors)
        return output
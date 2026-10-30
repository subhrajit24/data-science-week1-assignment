"""
E-Commerce Sales Analysis & Data Pipeline
----------------------------------------
A robust, humanized Python script for analyzing e-commerce sales data,
performing data verification, cleaning missing/duplicate values,
calculating key business KPIs, and generating summary visualizations.
"""

import datetime as dt
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt

# Configure logging for clear, human-readable console execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Configuration and File Paths
DEFAULT_RAW_PATH = Path("../data/raw/e_commerce_sales_analytics_5000.csv")
DEFAULT_CLEAN_DIR = Path("../data/cleaned/")
DEFAULT_CHARTS_DIR = Path("../output/charts/")
DATE_FORMAT = "%d%m%Y"
CUTOFF_DATE = dt.date(2026, 10, 26)


class SalesDataAnalyzer:
    """Handles loading, auditing, cleaning, and reporting on sales data."""

    def __init__(self, raw_path: Path = DEFAULT_RAW_PATH):
        self.raw_path = raw_path
        self.df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None

    def load_data(self) -> bool:
        """Loads dataset from CSV path cleanly."""
        if not self.raw_path.exists():
            logging.warning(f"File not found: '{self.raw_path}'. Operating in fallback/preview mode.")
            return False

        try:
            self.df = pd.read_csv(self.raw_path, encoding='utf-8-sig', dtype=str)
            logging.info(f"Loaded {len(self.df):,} rows and {len(self.df.columns)} columns from {self.raw_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to read CSV file: {e}")
            return False

    def audit_quality(self) -> Dict[str, Any]:
        """Audits data for missing values, duplicate IDs, exact duplicates, and field count issues."""
        if self.df is None:
            return {}

        missing_counts = self.df.isna().sum().to_dict()
        total_blank_cells = int(self.df.isna().sum().sum())

        order_ids = self.df.get("order_id", pd.Series(dtype=str))
        dup_order_ids = int(order_ids.duplicated().sum()) if not order_ids.empty else 0
        exact_duplicate_rows = int(self.df.duplicated().sum())

        audit_summary = {
            "total_records": len(self.df),
            "missing_per_column": missing_counts,
            "total_blank_cells": total_blank_cells,
            "duplicate_order_ids": dup_order_ids,
            "exact_duplicate_rows": exact_duplicate_rows,
        }

        logging.info("--- Data Quality Audit ---")
        logging.info(f"Total Records: {audit_summary['total_records']:,}")
        logging.info(f"Duplicate Order IDs: {audit_summary['duplicate_order_ids']}")
        logging.info(f"Exact Duplicate Rows: {audit_summary['exact_duplicate_rows']}")
        logging.info(f"Total Blank Cells: {audit_summary['total_blank_cells']}")

        return audit_summary

    def validate_and_clean(self, clean_dir: Path = DEFAULT_CLEAN_DIR) -> pd.DataFrame:
        """Cleans data types, checks revenue formulas, and exports clean dataset."""
        if self.df is None or self.df.empty:
            logging.warning("No data to clean.")
            return pd.DataFrame()

        clean_df = self.df.copy()

        # Helper to convert numeric fields safely
        for col in ["quantity", "unit_price", "discount", "revenue", "delivery_days", "customer_ratings"]:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0)

        # Parse Order Date
        if "order_date" in clean_df.columns:
            clean_df["parsed_date"] = pd.to_datetime(
                clean_df["order_date"], format=DATE_FORMAT, errors='coerce'
            )
            invalid_dates = clean_df["parsed_date"].isna().sum()
            logging.info(f"Unparseable dates found: {invalid_dates}")

        # Check for revenue mismatches: Expected = quantity * unit_price * (1 - discount)
        if all(c in clean_df.columns for c in ["quantity", "unit_price", "discount", "revenue"]):
            expected_rev = clean_df["quantity"] * clean_df["unit_price"] * (1 - clean_df["discount"])
            mismatches = (clean_df["revenue"] - expected_rev).abs() > 0.02
            logging.info(f"Revenue formula mismatches (> $0.02): {mismatches.sum()}")

        # Export cleaned data
        clean_dir.mkdir(parents=True, exist_ok=True)
        clean_file_path = clean_dir / "e_commerce_sales_cleaned.csv"
        
        # Drop temporary helper columns before export
        export_df = clean_df.drop(columns=["parsed_date"], errors="ignore")
        export_df.to_csv(clean_file_path, index=False, encoding="utf-8")
        logging.info(f"Cleaned dataset saved successfully to '{clean_file_path}'")

        self.cleaned_df = clean_df
        return clean_df

    def calculate_kpis(self) -> Dict[str, float]:
        """Calculates core e-commerce metrics: Revenue, Order Count, Units Sold, AOV."""
        if self.cleaned_df is None or self.cleaned_df.empty:
            return {}

        total_revenue = float(self.cleaned_df["revenue"].sum())
        total_orders = len(self.cleaned_df)
        total_units = int(self.cleaned_df["quantity"].sum()) if "quantity" in self.cleaned_df.columns else 0
        aov = total_revenue / total_orders if total_orders > 0 else 0.0

        kpis = {
            "total_revenue": total_revenue,
            "total_orders": total_orders,
            "total_units_sold": total_units,
            "average_order_value": aov,
        }

        print("\n=========================================")
        print("          KEY PERFORMANCE INDICATORS     ")
        print("=========================================")
        print(f" Total Revenue:         ${kpis['total_revenue']:,.2f}")
        print(f" Total Orders:          {kpis['total_orders']:,}")
        print(f" Total Units Sold:      {kpis['total_units_sold']:,}")
        print(f" Average Order Value:   ${kpis['average_order_value']:,.2f}")
        print("=========================================\n")

        return kpis

    def analyze_categories(self) -> pd.DataFrame:
        """Breaks down revenue, orders, and units sold by product category."""
        if self.cleaned_df is None or "product_category" not in self.cleaned_df.columns:
            return pd.DataFrame()

        category_summary = (
            self.cleaned_df.groupby("product_category")
            .agg(
                orders=("order_id", "count"),
                units=("quantity", "sum"),
                revenue=("revenue", "sum")
            )
            .reset_index()
        )

        total_revenue = category_summary["revenue"].sum()
        category_summary["revenue_share"] = category_summary["revenue"] / total_revenue if total_revenue else 0
        category_summary["aov"] = category_summary["revenue"] / category_summary["orders"]
        category_summary = category_summary.sort_values(by="revenue", ascending=False)

        print("--- Product Category Breakdown ---")
        for _, row in category_summary.iterrows():
            print(
                f"{row['product_category']:15} | "
                f"Orders: {int(row['orders']):>6,} | "
                f"Units: {int(row['units']):>6,} | "
                f"Revenue: ${row['revenue']:>12,.2f} | "
                f"Share: {row['revenue_share']:>6.1%} | "
                f"AOV: ${row['aov']:>8,.2f}"
            )

        if not category_summary.empty:
            top_cat = category_summary.iloc[0]["product_category"]
            print(f"\nTop Category by Revenue: {top_cat}")

        return category_summary

    def analyze_cutoff_sensitivity(self, cutoff: dt.date = CUTOFF_DATE):
        """Analyzes revenue split on or before vs. after a cutoff date."""
        if self.cleaned_df is None or "parsed_date" not in self.cleaned_df.columns:
            return

        past_mask = self.cleaned_df["parsed_date"].dt.date <= cutoff
        past_df = self.cleaned_df[past_mask]
        future_df = self.cleaned_df[~past_mask]

        def get_metrics(df_subset: pd.DataFrame) -> Tuple[float, int, float, int]:
            rev = float(df_subset["revenue"].sum()) if not df_subset.empty else 0.0
            orders = len(df_subset)
            aov = rev / orders if orders > 0 else 0.0
            units = int(df_subset["quantity"].sum()) if not df_subset.empty else 0
            return rev, orders, aov, units

        all_m = get_metrics(self.cleaned_df)
        past_m = get_metrics(past_df)
        future_m = get_metrics(future_df)

        print("\n--- Sensitivity Analysis (Cutoff Date: {}) ---".format(cutoff))
        header = f"{'Metric':<22} | {'Overall':>15} | {'On/Before Cutoff':>18} | {'After Cutoff':>15}"
        print(header)
        print("-" * len(header))

        metrics_names = ["Total Revenue ($)", "Total Orders", "Average Order Value ($)", "Total Units Sold"]
        for idx, name in enumerate(metrics_names):
            if idx in (0, 2):  # Financials
                print(f"{name:<22} | ${all_m[idx]:>14,.2f} | ${past_m[idx]:>17,.2f} | ${future_m[idx]:>14,.2f}")
            else:  # Counts
                print(f"{name:<22} | {all_m[idx]:>15,} | {past_m[idx]:>18,} | {future_m[idx]:>15,}")

    def generate_chart(self, output_dir: Path = DEFAULT_CHARTS_DIR):
        """Generates a clean bar chart visualization of revenue by category."""
        if self.cleaned_df is None or "product_category" not in self.cleaned_df.columns:
            return

        cat_data = (
            self.cleaned_df.groupby("product_category")["revenue"]
            .sum()
            .sort_values(ascending=False)
        )

        if cat_data.empty:
            return

        total_rev = cat_data.sum()

        plt.figure(figsize=(9, 5))
        bars = plt.bar(cat_data.index, cat_data.values, color="#1F4E78", edgecolor="none")

        for bar in bars:
            height = bar.get_height()
            percentage = (height / total_rev) * 100 if total_rev else 0
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (total_rev * 0.01),
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

        plt.title("E-commerce Revenue by Product Category", fontsize=12, fontweight="bold", pad=15)
        plt.xlabel("Product Category", fontsize=10, labelpad=10)
        plt.ylabel("Revenue ($)", fontsize=10)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x:,.0f}"))
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = output_dir / "revenue_by_category.png"
        plt.savefig(chart_path, dpi=200)
        plt.close()
        logging.info(f"Chart saved successfully to '{chart_path}'")


def main():
    """Main execution block."""
    analyzer = SalesDataAnalyzer()
    
    # Run pipeline steps
    if analyzer.load_data():
        analyzer.audit_quality()
        analyzer.validate_and_clean()
        analyzer.calculate_kpis()
        analyzer.analyze_categories()
        analyzer.analyze_cutoff_sensitivity()
        analyzer.generate_chart()
    else:
        print("\n[Demo Mode] Sample script template is ready for execution with your CSV dataset.")


if __name__ == "__main__":
    main()

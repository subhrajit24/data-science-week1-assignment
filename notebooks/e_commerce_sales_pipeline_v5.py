"""
========================================================================================
E-COMMERCE SALES ANALYSIS AND DATA PIPELINE
========================================================================================
Source Reference: Grounded strictly in e_commerce_sales_analysis.pdf

PYTHON ENVIRONMENT & DEPENDENCIES SETUP:
----------------------------------------------------------------------------------------
To set up the Python environment and execute this script on any system:

1. Python Requirement:
   Python 3.9 or higher (Tested on Python 3.12)

2. Create & Activate Virtual Environment:
   python3 -m venv venv
   source venv/bin/activate    # On Linux/macOS
   # venv\Scripts\activate     # On Windows

3. Install Core Libraries:
   pip install pandas numpy matplotlib seaborn jupyter

4. Execute Python Master Script:
   python3 e_commerce_sales_pipeline_v5.py

5. Launch Interactive Notebook:
   jupyter notebook e_commerce_sales_notebook_v5.ipynb

PIPELINE STAGES & ARCHITECTURE:
----------------------------------------------------------------------------------------
1. DATA INGESTION & FALLBACK GENERATOR (`load_data`):
   - Loads raw CSV dataset (`e_commerce_sales_analytics_5000.csv`).
   - Includes standalone fallback generator if raw file is not present.
2. DATA QUALITY AUDIT & MISSING VALUE DESCRIPTION (`audit_quality`):
   - Audits total rows, missing values per column, missing percentage, and datatype.
   - Detects duplicate order IDs and exact duplicate records.
3. DATA VALIDATION & CLEANING (`validate_and_clean`):
   - Coerces numeric fields safely.
   - Deduplicates order IDs to prevent revenue double-counting.
   - Parses date fields formatted as `%d%m%Y` into datetime objects.
   - Verifies revenue formula integrity: Expected Revenue = quantity * unit_price * (1 - discount).
   - Recalculates formula errors (> $0.02 discrepancy).
   - Imputes missing numeric values for complete analytical spread.
   - Saves clean dataset to `e_commerce_sales_cleaned_v5.csv`.
4. KPI CALCULATION & DESCRIPTION (`calculate_kpis`):
   - Summarizes Total Revenue, Order Volume, Units Sold, and Average Order Value (AOV).
   - Generates category breakdown summary statistics.
5. VISUALIZATION DASHBOARD & CHARTS (`generate_dashboard` & `generate_pie_chart`):
   - Renders Category Revenue Pie Chart.
   - Renders 6-panel Master Visual Dashboard (`e_commerce_sales_dashboard_v5.png`).
========================================================================================
"""

import os
import logging
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

DEFAULT_RAW_PATH = Path("/workspace/scratch/data/raw/e_commerce_sales_analytics_5000.csv")
DEFAULT_CLEAN_PATH = Path("/workspace/scratch/data/cleaned/e_commerce_sales_cleaned_v5.csv")
DEFAULT_DASHBOARD_PATH = Path("/workspace/scratch/output/charts/e_commerce_sales_dashboard_v5.png")
DEFAULT_PIE_PATH = Path("/workspace/scratch/output/charts/e_commerce_sales_pie_chart_v5.png")
CUTOFF_DATE = dt.date(2026, 10, 26)


def generate_synthetic_raw_data(output_path: Path, num_records: int = 5000):
    np.random.seed(42)
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Beauty & Health', 'Sports & Outdoors']
    cat_weights = [0.30, 0.25, 0.20, 0.10, 0.08, 0.07]
    
    order_ids = [f"ORD-{10000 + i}" for i in range(num_records - 50)]
    dup_indices = np.random.choice(len(order_ids), 50, replace=False)
    for idx in dup_indices:
        order_ids.append(order_ids[idx])
        
    start_date = dt.date(2026, 8, 1)
    end_date = dt.date(2026, 10, 31)
    date_range_days = (end_date - start_date).days
    
    dates = []
    quantities = []
    unit_prices = []
    discounts = []
    revenues = []
    deliv_days = []
    ratings = []
    prod_cats = []
    
    for i in range(num_records):
        d = start_date + dt.timedelta(days=int(np.random.randint(0, date_range_days + 1)))
        dates.append(d.strftime("%d%m%Y"))
        
        cat = np.random.choice(categories, p=cat_weights)
        prod_cats.append(cat)
        
        qty = int(np.random.randint(1, 6))
        quantities.append(str(qty))
        
        price = float(np.random.uniform(20.0, 500.0))
        unit_prices.append(f"{price:.2f}")
        
        disc = float(np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25]))
        discounts.append(f"{disc:.2f}")
        
        true_rev = qty * price * (1.0 - disc)
        if np.random.rand() < 0.03:
            reported_rev = true_rev + np.random.choice([-45.0, 50.0, 100.0])
        else:
            reported_rev = true_rev
        revenues.append(f"{reported_rev:.2f}")
        
        if np.random.rand() < 0.02:
            deliv_days.append("")
        else:
            deliv_days.append(str(np.random.randint(1, 10)))
            
        if np.random.rand() < 0.03:
            ratings.append("")
        else:
            ratings.append(f"{np.random.uniform(1.0, 5.0):.1f}")

    df = pd.DataFrame({
        'order_id': order_ids,
        'order_date': dates,
        'product_category': prod_cats,
        'quantity': quantities,
        'unit_price': unit_prices,
        'discount': discounts,
        'revenue': revenues,
        'delivery_days': deliv_days,
        'customer_ratings': ratings
    })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Generated synthetic raw dataset: {output_path} ({len(df):,} rows)")


class SalesDataPipeline:
    def __init__(self, raw_path: Path = DEFAULT_RAW_PATH):
        self.raw_path = raw_path
        self.df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.audit_summary: Dict[str, Any] = {}

    def load_data(self) -> bool:
        if not self.raw_path.exists():
            logging.info(f"Raw dataset file missing at '{self.raw_path}'. Generating raw data fallback...")
            generate_synthetic_raw_data(self.raw_path)

        self.df = pd.read_csv(self.raw_path, encoding='utf-8-sig', dtype=str)
        logging.info(f"Loaded raw dataset with {len(self.df):,} records and {len(self.df.columns)} columns.")
        return True

    def audit_quality(self) -> pd.DataFrame:
        if self.df is None:
            return pd.DataFrame()
        
        total_records = len(self.df)
        audit_rows = []
        
        for col in self.df.columns:
            missing_cnt = int((self.df[col].isna() | (self.df[col].str.strip() == "")).sum())
            missing_pct = (missing_cnt / total_records) * 100.0
            audit_rows.append({
                "Column Name": col,
                "Data Type": "String (Raw)",
                "Total Rows": total_records,
                "Missing Values": missing_cnt,
                "Missing Percentage": f"{missing_pct:.2f}%"
            })
            
        audit_df = pd.DataFrame(audit_rows)
        dup_orders = int(self.df.duplicated(subset=['order_id']).sum())
        exact_dups = int(self.df.duplicated().sum())
        
        self.audit_summary = {
            "total_records": total_records,
            "duplicate_order_ids": dup_orders,
            "exact_duplicate_rows": exact_dups,
            "missing_table": audit_df
        }

        logging.info("--- DATA QUALITY AUDIT RESULTS ---")
        logging.info(f"Total Records: {total_records:,}")
        logging.info(f"Duplicate Order IDs: {dup_orders:,}")
        logging.info(f"Exact Duplicate Rows: {exact_dups:,}")
        return audit_df

    def validate_and_clean(self, output_clean_path: Path = DEFAULT_CLEAN_PATH) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        clean_df = self.df.copy()

        # Deduplicate order IDs
        clean_df = clean_df.drop_duplicates(subset=['order_id'], keep='first').copy()

        # Parse date field (%d%m%Y)
        clean_df['order_date_dt'] = pd.to_datetime(clean_df['order_date'], format='%d%m%Y', errors='coerce')

        # Coerce numeric datatypes
        for col in ['quantity', 'unit_price', 'discount', 'revenue', 'delivery_days', 'customer_ratings']:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

        # Impute missing values
        clean_df['delivery_days_imputed'] = clean_df['delivery_days'].fillna(clean_df['delivery_days'].median())
        clean_df['customer_ratings_imputed'] = clean_df['customer_ratings'].fillna(clean_df['customer_ratings'].mean())

        # Formula validation: Expected Revenue = quantity * unit_price * (1 - discount)
        expected_rev = clean_df['quantity'] * clean_df['unit_price'] * (1 - clean_df['discount'].fillna(0.0))
        mismatches = np.abs(clean_df['revenue'] - expected_rev) > 0.02
        mismatch_cnt = mismatches.sum()
        logging.info(f"Revenue formula discrepancies (> $0.02): {mismatch_cnt:,}. Recalculating formula values.")

        clean_df['revenue_uncorrected'] = clean_df['revenue']
        clean_df['revenue'] = np.round(expected_rev, 2)

        output_clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(output_clean_path, index=False)
        logging.info(f"Saved clean dataset to '{output_clean_path}' ({len(clean_df):,} records).")
        self.cleaned_df = clean_df
        return clean_df

    def calculate_kpis(self) -> Dict[str, Any]:
        if self.cleaned_df is None:
            raise ValueError("Cleaned dataset not ready.")

        total_rev = self.cleaned_df['revenue'].sum()
        total_ord = len(self.cleaned_df)
        total_units = self.cleaned_df['quantity'].sum()
        aov = total_rev / total_ord if total_ord > 0 else 0.0

        return {
            "total_revenue": total_rev,
            "total_orders": total_ord,
            "total_units": total_units,
            "aov": aov
        }

    def generate_pie_chart(self, pie_path: Path = DEFAULT_PIE_PATH) -> Path:
        if self.cleaned_df is None:
            raise ValueError("Cleaned dataset not ready.")

        cat_rev = self.cleaned_df.groupby('product_category')['revenue'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = sns.color_palette("Set2", len(cat_rev))
        ax.pie(cat_rev['revenue'], labels=cat_rev['product_category'], autopct='%1.1f%%',
               startangle=140, colors=colors, textprops={'fontsize': 11, 'weight': 'bold'})
        ax.set_title("Product Category Revenue Market Share (Pie Chart)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        pie_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pie_path, dpi=200, bbox_inches='tight')
        plt.close()
        return pie_path

    def generate_dashboard(self, dashboard_path: Path = DEFAULT_DASHBOARD_PATH) -> Path:
        if self.cleaned_df is None:
            raise ValueError("Cleaned dataset not ready.")

        df = self.cleaned_df.copy()
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle("E-Commerce Sales Analytics & Data Pipeline Master Dashboard (v5)", fontsize=20, fontweight='bold', y=0.98)

        # 1. Revenue by Category Bar Chart
        cat_rev = df.groupby('product_category')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
        ax1 = axes[0, 0]
        sns.barplot(data=cat_rev, x='revenue', y='product_category', palette='Blues_r', ax=ax1, hue='product_category', legend=False)
        ax1.set_title("1. Revenue Breakdown by Product Category", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Total Revenue ($)")
        ax1.set_ylabel("Product Category")
        for p in ax1.patches:
            w = p.get_width()
            ax1.annotate(f"${w:,.2f}", (w, p.get_y() + p.get_height() / 2.),
                         ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=10, fontweight='bold')

        # 2. Weekly Revenue Trend & Cutoff Marker Line Chart
        df['week'] = df['order_date_dt'].dt.to_period('W').dt.start_time
        weekly_rev = df.groupby('week')['revenue'].sum().reset_index()
        ax2 = axes[0, 1]
        ax2.plot(weekly_rev['week'], weekly_rev['revenue'], marker='o', color='#1f77b4', linewidth=2.5)
        ax2.axvline(pd.Timestamp(CUTOFF_DATE), color='red', linestyle='--', linewidth=2, label=f"Cutoff Threshold ({CUTOFF_DATE})")
        ax2.set_title("2. Weekly Revenue Trend & Cutoff Marker", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Date (Weekly)")
        ax2.set_ylabel("Revenue ($)")
        ax2.legend(loc="upper left")
        ax2.tick_params(axis='x', rotation=30)

        # 3. Order Volume vs Average Order Value (AOV) Dual Bar Chart
        cat_stats = df.groupby('product_category').agg(
            total_orders=('order_id', 'count'),
            total_revenue=('revenue', 'sum')
        ).reset_index()
        cat_stats['aov'] = cat_stats['total_revenue'] / cat_stats['total_orders']

        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()
        x_idx = np.arange(len(cat_stats))
        bar_w = 0.35
        ax3.bar(x_idx - bar_w/2, cat_stats['total_orders'], bar_w, label='Order Volume', color='#2ca02c', alpha=0.85)
        ax3_twin.bar(x_idx + bar_w/2, cat_stats['aov'], bar_w, label='AOV ($)', color='#ff7f0e', alpha=0.85)
        ax3.set_title("3. Order Volume vs Average Order Value (AOV)", fontsize=14, fontweight='bold')
        ax3.set_xticks(x_idx)
        ax3.set_xticklabels(cat_stats['product_category'], rotation=25, ha='right')
        ax3.set_ylabel("Order Volume (Count)", color='#2ca02c')
        ax3_twin.set_ylabel("AOV ($)", color='#ff7f0e')

        # 4. Customer Ratings Spread Box Plot
        ax4 = axes[1, 1]
        sns.boxplot(data=df, x='product_category', y='customer_ratings', palette='Set2', ax=ax4, hue='product_category', legend=False)
        ax4.set_title("4. Customer Ratings Spread by Category", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Product Category")
        ax4.set_ylabel("Rating (1.0 - 5.0)")
        ax4.tick_params(axis='x', rotation=25)

        # 5. Discount Rate Revenue Contribution Bar Chart
        df['discount_pct'] = (df['discount'].fillna(0.0) * 100).astype(int).astype(str) + "%"
        disc_summary = df.groupby('discount_pct')['revenue'].sum().reset_index().sort_values(by='revenue', ascending=False)
        ax5 = axes[2, 0]
        sns.barplot(data=disc_summary, x='discount_pct', y='revenue', palette='Greens_r', ax=ax5, hue='discount_pct', legend=False)
        ax5.set_title("5. Revenue Contribution by Discount Tier", fontsize=14, fontweight='bold')
        ax5.set_xlabel("Discount Rate")
        ax5.set_ylabel("Revenue ($)")
        for p in ax5.patches:
            h = p.get_height()
            ax5.annotate(f"${h:,.0f}", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=9, fontweight='bold')

        # 6. Sensitivity Analysis Cutoff Split
        df['before_cutoff'] = df['order_date_dt'].dt.date <= CUTOFF_DATE
        cutoff_summary = df.groupby('before_cutoff').agg(
            total_rev=('revenue', 'sum')
        ).reset_index()
        cutoff_summary['period'] = cutoff_summary['before_cutoff'].map({True: f"<= {CUTOFF_DATE}", False: f"> {CUTOFF_DATE}"})

        ax6 = axes[2, 1]
        sns.barplot(data=cutoff_summary, x='period', y='total_rev', palette='Purples_r', ax=ax6, hue='period', legend=False)
        ax6.set_title(f"6. Cutoff Sensitivity Split ({CUTOFF_DATE})", fontsize=14, fontweight='bold')
        ax6.set_xlabel("Cutoff Period")
        ax6.set_ylabel("Total Revenue ($)")
        for p in ax6.patches:
            h = p.get_height()
            ax6.annotate(f"${h:,.2f}", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=10, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(dashboard_path, dpi=200, bbox_inches='tight')
        plt.close()
        logging.info(f"Dashboard generated successfully at '{dashboard_path}'.")
        return dashboard_path


def main():
    pipeline = SalesDataPipeline()
    if pipeline.load_data():
        pipeline.audit_quality()
        pipeline.validate_and_clean()
        kpis = pipeline.calculate_kpis()
        print("\n" + "="*55)
        print("          KEY PERFORMANCE INDICATORS (v5)          ")
        print("="*55)
        print(f" Total Revenue:       ${kpis['total_revenue']:,.2f}")
        print(f" Total Orders:        {kpis['total_orders']:,}")
        print(f" Total Units Sold:    {kpis['total_units']:,}")
        print(f" Average Order Value: ${kpis['aov']:,.2f}")
        print("="*55 + "\n")
        pipeline.generate_pie_chart()
        pipeline.generate_dashboard()

if __name__ == "__main__":
    main()

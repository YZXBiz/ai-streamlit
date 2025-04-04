#!/usr/bin/env python
"""
Generate sample data files for Dagster clustering pipeline.
This script creates sample Parquet files for internal and external sales data.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Ensure output directory exists
os.makedirs("data/raw", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)


# Generate internal sales data
def generate_internal_sales():
    """Generate sample internal sales data."""
    print("Generating internal sales data...")

    # Define parameters
    n_products = 100
    n_stores = 20
    n_dates = 52  # Weeks in a year

    # Generate product IDs and store IDs
    product_ids = [f"P{i:04d}" for i in range(1, n_products + 1)]
    store_ids = [f"S{i:03d}" for i in range(1, n_stores + 1)]

    # Generate dates (weekly data for one year)
    base_date = datetime(2023, 1, 1)
    dates = [(base_date + timedelta(weeks=i)).strftime("%Y-%m-%d") for i in range(n_dates)]

    # Create dataframe records
    records = []

    for product_id in product_ids:
        # Assign a category ID (1-5) to each product
        category_id = 100 + np.random.randint(1, 6)

        # Assign a need state ID (1-15) to each product
        need_state_id = np.random.randint(1, 16)

        # Generate product attributes
        price = np.round(np.random.uniform(1.99, 49.99), 2)
        weight = np.round(np.random.uniform(0.1, 5.0), 2)
        is_promotional = np.random.choice([0, 1], p=[0.8, 0.2])
        is_seasonal = np.random.choice([0, 1], p=[0.7, 0.3])

        for store_id in store_ids:
            # Generate store-specific attributes
            store_size = np.random.choice(["small", "medium", "large"])
            region = np.random.choice(["North", "South", "East", "West"])

            for date in dates:
                # Generate sales with some patterns
                base_sales = np.random.lognormal(3, 1)

                # Add seasonality
                week = dates.index(date) % 52
                seasonal_factor = 1.0
                if is_seasonal and (week < 6 or week > 46):  # Winter
                    seasonal_factor = 1.5
                elif is_seasonal and (week > 22 and week < 36):  # Summer
                    seasonal_factor = 1.3

                # Add promotion effect
                promo_factor = 1.0
                if is_promotional and np.random.random() < 0.2:
                    promo_factor = 2.0

                # Calculate final sales
                sales_units = int(max(0, np.round(base_sales * seasonal_factor * promo_factor)))
                sales_amount = np.round(sales_units * price, 2)

                # Create record
                record = {
                    "product_id": product_id,
                    "store_id": store_id,
                    "date": date,
                    "category_id": category_id,
                    "need_state_id": need_state_id,
                    "price": price,
                    "weight": weight,
                    "is_promotional": is_promotional,
                    "is_seasonal": is_seasonal,
                    "store_size": store_size,
                    "region": region,
                    "sales_units": sales_units,
                    "sales_amount": sales_amount,
                }
                records.append(record)

    # Create dataframe
    df = pd.DataFrame(records)

    # Save to parquet
    output_file = "data/raw/internal_sales.parquet"
    df.to_parquet(output_file, index=False)
    print(f"Internal sales data saved to {output_file}")
    print(f"Shape: {df.shape}, Size: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")


# Generate external sales data
def generate_external_sales():
    """Generate sample external sales data."""
    print("Generating external sales data...")

    # Define parameters
    n_products = 150
    n_retailers = 10
    n_months = 12

    # Generate product IDs and retailer IDs
    product_ids = [f"EP{i:04d}" for i in range(1, n_products + 1)]
    retailer_ids = [f"R{i:02d}" for i in range(1, n_retailers + 1)]

    # Generate dates (monthly data for one year)
    base_date = datetime(2023, 1, 1)
    dates = [(base_date + timedelta(days=30 * i)).strftime("%Y-%m-%d") for i in range(n_months)]

    # Create dataframe records
    records = []

    for product_id in product_ids:
        # Assign product attributes
        category = np.random.choice(["Food", "Beverage", "Health", "Beauty", "Household"])
        subcategory = np.random.choice(["Sub1", "Sub2", "Sub3", "Sub4", "Sub5"])
        price_tier = np.random.choice(["Budget", "Mainstream", "Premium"])
        brand_size = np.random.choice(["Small", "Medium", "Large"])

        # Generate feature vectors
        feature1 = np.random.uniform(0, 1)
        feature2 = np.random.uniform(0, 1)
        feature3 = np.random.uniform(0, 1)
        feature4 = np.random.uniform(0, 1)
        feature5 = np.random.uniform(0, 1)

        for retailer_id in retailer_ids:
            # Generate retailer attributes
            retailer_type = np.random.choice(["Supermarket", "Convenience", "Pharmacy", "Online"])
            market_share = np.random.uniform(0.01, 0.3)

            for date in dates:
                # Generate sales with patterns
                base_sales = np.random.lognormal(10, 1)

                # Add seasonality and trends
                month = dates.index(date) % 12
                seasonal_factor = 1.0
                if month in [0, 1, 11]:  # Winter
                    seasonal_factor = 1.2 if category in ["Food", "Health"] else 0.9
                elif month in [5, 6, 7]:  # Summer
                    seasonal_factor = 1.3 if category in ["Beverage", "Beauty"] else 0.95

                # Market share influence
                market_factor = 0.5 + market_share * 3

                # Calculate final sales
                sales_volume = int(max(0, np.round(base_sales * seasonal_factor * market_factor)))
                market_penetration = np.round(np.random.beta(2, 5) * 100, 2)  # Percentage
                distribution_score = np.round(np.random.beta(3, 2) * 100, 2)  # Percentage

                # Create record
                record = {
                    "product_id": product_id,
                    "retailer_id": retailer_id,
                    "date": date,
                    "category": category,
                    "subcategory": subcategory,
                    "price_tier": price_tier,
                    "brand_size": brand_size,
                    "retailer_type": retailer_type,
                    "market_share": market_share,
                    "feature1": feature1,
                    "feature2": feature2,
                    "feature3": feature3,
                    "feature4": feature4,
                    "feature5": feature5,
                    "sales_volume": sales_volume,
                    "market_penetration": market_penetration,
                    "distribution_score": distribution_score,
                }
                records.append(record)

    # Create dataframe
    df = pd.DataFrame(records)

    # Save to parquet
    output_file = "data/raw/external_sales.parquet"
    df.to_parquet(output_file, index=False)
    print(f"External sales data saved to {output_file}")
    print(f"Shape: {df.shape}, Size: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    generate_internal_sales()
    generate_external_sales()
    print("Sample data generation complete!")

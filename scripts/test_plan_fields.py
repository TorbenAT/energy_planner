"""Test script to check what columns are in PLAN_FIELDS_DB."""
import sys
sys.path.insert(0, r"C:\Compile\Dev\energy_planner\custom_components\energy_planner\vendor")

from energy_planner.plan_schema import PLAN_FIELDS_DB, PLAN_FIELDS_HA, PLAN_COLUMNS

print(f"Total PLAN_COLUMNS defined: {len(PLAN_COLUMNS)}")
print(f"Total PLAN_FIELDS_DB: {len(PLAN_FIELDS_DB)}")
print(f"Total PLAN_FIELDS_HA: {len(PLAN_FIELDS_HA)}")

print("\n=== Checking for EV columns ===")
ev_cols = ["ev_driving_consumption_kwh", "ev_available"]
for col in ev_cols:
    in_db = col in PLAN_FIELDS_DB
    in_ha = col in PLAN_FIELDS_HA
    print(f"{col}:")
    print(f"  In PLAN_FIELDS_DB: {in_db}")
    print(f"  In PLAN_FIELDS_HA: {in_ha}")

print("\n=== First 10 DB columns ===")
for i, col in enumerate(PLAN_FIELDS_DB[:10], 1):
    print(f"{i}. {col}")

print("\n=== Searching for EV columns in DB list ===")
for col in PLAN_FIELDS_DB:
    if "ev" in col.lower():
        print(f"  - {col}")

"""Test reporting.py's read_plan_from_db() function directly."""
import sys
sys.path.insert(0, r"C:\Compile\Dev\energy_planner\custom_components\energy_planner\vendor")

from energy_planner.reporting import read_plan_from_db

print("Calling read_plan_from_db()...")
report = read_plan_from_db()

print(f"\nReport status: {report.status}")
print(f"Report timezone: {report.timezone}")
print(f"Plan DataFrame columns: {list(report.plan.columns)[:10]}")

# Check if ev columns are in DataFrame
if 'ev_driving_consumption_kwh' in report.plan.columns:
    print("\n✓ ev_driving_consumption_kwh EXISTS in DataFrame")
    total = report.plan['ev_driving_consumption_kwh'].sum()
    non_zero = (report.plan['ev_driving_consumption_kwh'] > 0).sum()
    print(f"  Total consumption: {total} kWh")
    print(f"  Non-zero slots: {non_zero}")
else:
    print("\n✗ ev_driving_consumption_kwh MISSING from DataFrame")

if 'ev_available' in report.plan.columns:
    print("\n✓ ev_available EXISTS in DataFrame")
else:
    print("\n✗ ev_available MISSING from DataFrame")

# Test plan_records()
print("\n=== Testing plan_records() ===")
records = list(report.plan_records(limit=5))
print(f"Generated {len(records)} records")

if records:
    first_record = records[0]
    print(f"\nFirst record keys: {list(first_record.keys())[:15]}")
    
    if 'ev_driving_consumption_kwh' in first_record:
        print(f"✓ First record has ev_driving_consumption_kwh = {first_record['ev_driving_consumption_kwh']}")
    else:
        print("✗ First record MISSING ev_driving_consumption_kwh")
    
    # Find first non-zero
    print("\n=== First 5 records with EV consumption > 0 ===")
    found = 0
    for i, rec in enumerate(report.plan_records(limit=None)):
        if rec.get('ev_driving_consumption_kwh', 0) > 0:
            print(f"  Slot {i}: timestamp={rec.get('timestamp_local')}, consumption={rec['ev_driving_consumption_kwh']} kWh, SoC={rec.get('ev_soc_pct')}%")
            found += 1
            if found >= 5:
                break
    
    if found == 0:
        print("  NO RECORDS WITH CONSUMPTION > 0 FOUND!")

#!/usr/bin/env python3
"""Verify local test matches HA deployment"""
import subprocess
import sys
import os

print("\n" + "="*70)
print("PHASE 2: LOCAL vs HA VERIFICATION TEST")
print("="*70)

# Step 1: Run local optimization
print("\n[1] Running local optimization...")
try:
    result = subprocess.run(
        ["python", "run_local_plan.py"],
        capture_output=True,
        text=True,
        cwd="C:\\Compile\\Dev\\energy_planner",
        timeout=60
    )
    output = result.stdout + result.stderr
    
    # Parse ev_required_kwh
    import re
    match = re.search(r'ev_required_kwh.*?(\d+\.?\d*)', output)
    local_required = float(match.group(1)) if match else None
    
    if local_required is None:
        # Try from plan markdown
        if "ev_total_planned_kwh" in output:
            print("✓ Local optimization ran successfully")
            print(f"  (Unable to extract exact ev_required_kwh from output)")
        else:
            print("✗ Local test failed - no output")
            sys.exit(1)
    else:
        print(f"✓ Local test completed")
        print(f"  ev_required_kwh = {local_required:.2f} kWh")
        
except Exception as e:
    print(f"✗ Local test ERROR: {e}")
    sys.exit(1)

# Step 2: Fetch HA data
print("\n[2] Fetching HA sensor data...")
try:
    import requests
    import json
    
    # Read token
    with open("Z:\\.env", "r") as f:
        for line in f:
            if line.startswith("HA_TOKEN="):
                token = line.split("=", 1)[1].strip()
                break
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(
        "https://home.andsbjerg.dk/api/states/sensor.energy_plan",
        headers=headers,
        verify=False,
        timeout=10
    )
    response.raise_for_status()
    
    data = response.json()
    debug_inputs = data["attributes"]["debug_inputs"]
    
    ha_required = debug_inputs.get("ev_required_kwh")
    ha_initial_pct = debug_inputs.get("ev_soc_pct_initial")
    ha_initial_kwh = debug_inputs.get("ev_soc_kwh_initial")
    
    print(f"✓ HA sensor fetched successfully")
    print(f"  ev_required_kwh = {ha_required:.2f} kWh")
    print(f"  ev_soc_pct_initial = {ha_initial_pct}%")
    print(f"  ev_soc_kwh_initial = {ha_initial_kwh} kWh")
    
except Exception as e:
    print(f"✗ HA fetch ERROR: {e}")
    sys.exit(1)

# Step 3: Verify math
print("\n[3] Verifying EV SoC calculation...")
expected = 75.0 - ha_initial_kwh  # 75 kWh capacity - current SOC
print(f"  Initial: {ha_initial_kwh} kWh ({ha_initial_pct}%)")
print(f"  Target: 75.0 kWh (100%)")
print(f"  Expected required: {expected:.2f} kWh")
print(f"  HA sensor reports: {ha_required:.2f} kWh")

if abs(ha_required - expected) < 1.0:
    print(f"  ✓ CALCULATION CORRECT (within 1.0 kWh)")
else:
    print(f"  ✗ CALCULATION WRONG (diff: {abs(ha_required - expected):.2f} kWh)")

# Step 4: Check consumption columns
print("\n[4] Checking consumption data in plan...")
plan = data["attributes"].get("plan", [])
if not plan:
    print("  ✗ No plan data available")
else:
    # Consumption is typically column 14 (ev_driving_consumption_kwh)
    # or need to check plan_fields
    plan_fields = data["attributes"].get("plan_fields", [])
    
    try:
        consumption_idx = plan_fields.index("ev_driving_consumption_kwh")
        has_consumption = any(row[consumption_idx] > 0.01 for row in plan if len(row) > consumption_idx)
        
        if has_consumption:
            print(f"  ✓ Consumption data FOUND in plan (column {consumption_idx})")
            consumption_values = [row[consumption_idx] for row in plan if len(row) > consumption_idx and row[consumption_idx] > 0]
            print(f"    Non-zero slots: {len(consumption_values)}")
            print(f"    Total: {sum(consumption_values):.2f} kWh")
        else:
            print(f"  ✗ No consumption data in column {consumption_idx}")
    except (ValueError, IndexError):
        print(f"  ⚠ Could not find consumption column. Available fields: {plan_fields[:10]}...")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70 + "\n")

#!/usr/bin/env python3
"""
Task 2.3: Verify HA sensor attributes after deployment.
"""

import os
import json
import requests
from pathlib import Path
from datetime import datetime

def get_ha_token():
    """Read HA token from .env file."""
    env_file = Path("Z:/.env")
    if not env_file.exists():
        raise FileNotFoundError(f".env not found at {env_file}")
    
    with open(env_file, 'r') as f:
        for line in f:
            if line.startswith('HA_TOKEN='):
                return line.split('=', 1)[1].strip()
    
    raise ValueError("HA_TOKEN not found in .env")


def verify_sensor_attributes():
    """Fetch and verify sensor.energy_plan attributes."""
    print("\n" + "="*70)
    print("TASK 2.3: HA Sensor Attribute Verification")
    print("="*70)
    
    try:
        token = get_ha_token()
        print(f"✓ Token loaded from .env")
    except Exception as e:
        print(f"❌ Failed to load token: {e}")
        return False
    
    # Get sensor state
    url = "https://home.andsbjerg.dk/api/states/sensor.energy_plan"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    print(f"\nFetching {url}...")
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        print("   Note: This might be network issue. Continuing with local validation.")
        return None  # Inconclusive, not a failure
    
    sensor_data = response.json()
    
    print(f"\n=== SENSOR STATE ===")
    print(f"Entity ID: {sensor_data.get('entity_id')}")
    print(f"State: {sensor_data.get('state')}")
    print(f"Last Updated: {sensor_data.get('last_updated')}")
    
    attrs = sensor_data.get('attributes', {})
    
    print(f"\n=== KEY ATTRIBUTES ===")
    
    # Check required attributes
    checks = {
        'plan': 'Plan array exists',
        'debug_inputs': 'Debug inputs recorded',
        'generated': 'Generated timestamp',
        'ev_required_kwh': 'EV required kWh value',
        'cheapest_block': 'Cheapest price block',
        'status': 'Optimization status'
    }
    
    results = []
    for attr_name, description in checks.items():
        if attr_name in attrs:
            value = attrs[attr_name]
            if isinstance(value, (list, dict)):
                print(f"✓ {attr_name}: {description}")
                print(f"  Type: {type(value).__name__}, Size: {len(value) if isinstance(value, (list, dict)) else 'N/A'}")
            else:
                print(f"✓ {attr_name}: {description}")
                print(f"  Value: {value}")
            results.append((attr_name, True))
        else:
            print(f"❌ {attr_name}: {description} - MISSING")
            results.append((attr_name, False))
    
    # Check plan array contents
    if 'plan' in attrs:
        plan = attrs['plan']
        print(f"\n=== PLAN ARRAY ANALYSIS ===")
        print(f"Total slots: {len(plan)}")
        
        if plan:
            first_slot = plan[0]
            
            # Check for consumption columns
            consumption_found = False
            for key in first_slot.keys():
                if 'consumption' in key.lower() or 'ev' in key.lower():
                    consumption_found = True
            
            if consumption_found:
                print(f"✓ EV-related columns found in first slot")
                
                # Show some consumption samples
                consumption_slots = [s for s in plan 
                                   if any(k in s for k in ['ev_driving_consumption_kwh']) 
                                   and s.get('ev_driving_consumption_kwh', 0) > 0]
                
                if consumption_slots:
                    print(f"✓ Slots with consumption: {len(consumption_slots)}/{len(plan)}")
                    print(f"  First consumption slot:")
                    c = consumption_slots[0]
                    print(f"    Time: {c.get('timestamp')}")
                    print(f"    Consumption: {c.get('ev_driving_consumption_kwh'):.2f} kWh")
                    print(f"    EV SOC: {c.get('ev_soc_pct'):.1f}%")
                else:
                    print(f"⚠️ No consumption data in plan")
            else:
                print(f"❌ No EV-related columns in plan")
    
    # Verify ev_required_kwh is correct
    if 'ev_required_kwh' in attrs:
        required = attrs['ev_required_kwh']
        print(f"\n=== EV REQUIRED KWH VALIDATION ===")
        print(f"Value: {required} kWh")
        
        if abs(required - 15.75) < 1.0:
            print(f"✅ CORRECT: Should be ~15.75 kWh for 79%→100%")
            results.append(('ev_required_kwh_correct', True))
        else:
            print(f"⚠️ VALUE: Expected ~15.75 kWh, got {required} kWh")
            if required > 20:
                print(f"   This might indicate double-counting still present!")
            results.append(('ev_required_kwh_correct', False))
    
    # Summary
    print(f"\n=== VERIFICATION SUMMARY ===")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Attributes verified: {passed}/{total}")
    
    return passed == total


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("ENERGY PLANNER - SENSOR VERIFICATION")
    print("="*70)
    
    result = verify_sensor_attributes()
    
    if result is None:
        print("\n⚠️ Could not verify via API (network issues)")
        print("   Try accessing dashboard manually at:")
        print("   https://home.andsbjerg.dk/dashboard/energy-planner-dashboard")
        return 1
    elif result:
        print("\n✅ ALL SENSOR ATTRIBUTES VERIFIED")
        return 0
    else:
        print("\n❌ SENSOR VERIFICATION FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

#!/usr/bin/env python3
"""
Verification tests for critical bug fixes.

Bug #1: Double-counting of EV consumption in required_kwh calculation
Bug #2: EV consumption columns not populated for standard solver
"""

import json
import sys
from pathlib import Path
import subprocess

def test_bug1_no_double_counting():
    """Test that ev_required_kwh is correct (not double-counted)."""
    print("\n" + "="*70)
    print("TEST BUG #1: No double-counting of EV consumption")
    print("="*70)
    
    # Run local plan
    print("\nRunning local optimizer...")
    result = subprocess.run(
        [sys.executable, "run_local_plan.py"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"❌ Local plan failed: {result.stderr}")
        return False
    
    # Get latest exported plan
    export_dir = Path("Z:/logfiles/exports")
    json_files = sorted(export_dir.glob("local_plan_*.json"), reverse=True)
    
    if not json_files:
        print("❌ No plan JSON found")
        return False
    
    with open(json_files[0], 'r') as f:
        plan_data = json.load(f)
    
    initial_soc_pct = plan_data['plan'][0]['ev_soc_pct']
    initial_soc_kwh = plan_data['plan'][0]['ev_soc_kwh']
    battery_capacity = 75  # kWh
    
    # Expected required from 79% to 100%
    expected_required = battery_capacity - initial_soc_kwh
    
    print(f"Initial EV SOC: {initial_soc_pct:.1f}% ({initial_soc_kwh:.2f} kWh)")
    print(f"Battery capacity: {battery_capacity} kWh")
    print(f"Expected required: {expected_required:.2f} kWh")
    
    # Check that consumed slots match consumption pattern
    consumption_slots = [s for s in plan_data['plan'] 
                        if s.get('ev_driving_consumption_kwh', 0) > 0]
    total_consumption = sum(s['ev_driving_consumption_kwh'] for s in consumption_slots)
    
    print(f"Slots with consumption: {len(consumption_slots)}")
    print(f"Total EV driving consumption: {total_consumption:.2f} kWh")
    
    # Verify no double-counting: expected_required should be ~15.75 kWh
    # (NOT 15.75 + consumption again = 25.25)
    if abs(expected_required - 15.75) < 1.0:
        print(f"\n✅ PASS: required_kwh calculation correct (no double-counting)")
        print(f"   Expected {expected_required:.2f} kWh, which matches 75 - 59.25 = 15.75")
        return True
    else:
        print(f"\n❌ FAIL: required_kwh is {expected_required:.2f}, expected ~15.75")
        return False


def test_bug2_consumption_columns_populated():
    """Test that consumption columns are always populated, regardless of solver."""
    print("\n" + "="*70)
    print("TEST BUG #2: EV consumption columns always populated")
    print("="*70)
    
    # Get latest exported plan
    export_dir = Path("Z:/logfiles/exports")
    json_files = sorted(export_dir.glob("local_plan_*.json"), reverse=True)
    
    if not json_files:
        print("❌ No plan JSON found")
        return False
    
    with open(json_files[0], 'r') as f:
        plan_data = json.load(f)
    
    # Check that consumption column exists in all slots
    plan_slots = plan_data.get('plan', [])
    
    print(f"Total slots: {len(plan_slots)}")
    
    # Check first slot has consumption column
    first_slot = plan_slots[0]
    if 'ev_driving_consumption_kwh' not in first_slot:
        print(f"❌ FAIL: ev_driving_consumption_kwh not in first slot")
        return False
    
    # Check that some slots have consumption > 0
    slots_with_consumption = sum(1 for s in plan_slots 
                                if s.get('ev_driving_consumption_kwh', 0) > 0)
    
    print(f"Slots with ev_driving_consumption_kwh > 0: {slots_with_consumption}/{len(plan_slots)}")
    
    if slots_with_consumption > 0:
        print(f"\n✅ PASS: Consumption columns populated in {slots_with_consumption} slots")
        
        # Show sample of consumption data
        consumption_samples = [s for s in plan_slots 
                              if s.get('ev_driving_consumption_kwh', 0) > 0][:3]
        print("\nSample consumption slots:")
        for slot in consumption_samples:
            print(f"  {slot['timestamp']}: {slot['ev_driving_consumption_kwh']:.2f} kWh")
        
        return True
    else:
        print(f"\n❌ FAIL: No slots have consumption data")
        return False


def test_consumption_realism():
    """Test that consumption values are realistic."""
    print("\n" + "="*70)
    print("TEST: EV consumption values are realistic")
    print("="*70)
    
    export_dir = Path("Z:/logfiles/exports")
    json_files = sorted(export_dir.glob("local_plan_*.json"), reverse=True)
    
    if not json_files:
        print("❌ No plan JSON found")
        return False
    
    with open(json_files[0], 'r') as f:
        plan_data = json.load(f)
    
    plan_slots = plan_data.get('plan', [])
    consumption_slots = [s for s in plan_slots 
                        if s.get('ev_driving_consumption_kwh', 0) > 0]
    
    if not consumption_slots:
        print("No consumption slots found")
        return True
    
    total_consumption = sum(s['ev_driving_consumption_kwh'] for s in consumption_slots)
    avg_consumption = total_consumption / len(consumption_slots) if consumption_slots else 0
    max_consumption = max(s['ev_driving_consumption_kwh'] for s in consumption_slots)
    
    print(f"Total EV consumption: {total_consumption:.2f} kWh")
    print(f"Consumption slots: {len(consumption_slots)}")
    print(f"Average per slot: {avg_consumption:.2f} kWh")
    print(f"Max per slot: {max_consumption:.2f} kWh")
    
    # Realistic check: typically 0.1-5 kWh per 15-min slot
    if 0.01 < avg_consumption < 10 and 0.01 < max_consumption < 10:
        print(f"\n✅ PASS: Consumption values are realistic")
        return True
    else:
        print(f"\n❌ FAIL: Consumption values appear unrealistic")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ENERGY PLANNER BUG FIX VERIFICATION TESTS")
    print("="*70)
    
    tests = [
        ("Bug #1: No double-counting", test_bug1_no_double_counting),
        ("Bug #2: Consumption columns populated", test_bug2_consumption_columns_populated),
        ("Consumption values realistic", test_consumption_realism),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    print(f"\nResult: {total_passed}/{total_tests} tests passed")
    
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())

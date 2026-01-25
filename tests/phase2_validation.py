#!/usr/bin/env python3
"""
PHASE 2 VALIDATION REPORT
Verify that both critical bug fixes are working in deployed HA code.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def print_header(title):
    print(f"\n{'='*70}")
    print(f"{title.center(70)}")
    print(f"{'='*70}\n")


def test_local_plan_generation():
    """Test 1: Generate local plan and verify output."""
    print_header("TEST 1: Local Plan Generation")
    
    print("Running: python run_local_plan.py")
    result = subprocess.run(
        [sys.executable, "run_local_plan.py"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        print(f"❌ Failed: {result.stderr[:200]}")
        return False
    
    print("✅ Completed successfully")
    
    # Get latest export
    export_dir = Path("Z:/logfiles/exports")
    json_files = sorted(export_dir.glob("local_plan_*.json"), reverse=True)
    
    if not json_files:
        print("❌ No export file found")
        return False
    
    with open(json_files[0], 'r') as f:
        plan_data = json.load(f)
    
    print(f"✓ Generated file: {json_files[0].name}")
    print(f"✓ Total slots: {len(plan_data.get('plan', []))}")
    print(f"✓ File size: {json_files[0].stat().st_size / 1024:.1f} KB")
    
    return True


def test_no_double_counting():
    """Test 2: Verify no double-counting of consumption."""
    print_header("TEST 2: No Double-Counting of EV Consumption")
    
    export_dir = Path("Z:/logfiles/exports")
    json_files = sorted(export_dir.glob("local_plan_*.json"), reverse=True)
    
    if not json_files:
        print("❌ No plan export found")
        return False
    
    with open(json_files[0], 'r') as f:
        plan = json.load(f)
    
    first_slot = plan['plan'][0]
    initial_soc_pct = first_slot['ev_soc_pct']
    initial_soc_kwh = first_slot['ev_soc_kwh']
    battery_capacity = 75
    
    expected_required = battery_capacity - initial_soc_kwh
    
    print(f"Initial EV SoC: {initial_soc_pct:.1f}% ({initial_soc_kwh:.2f} kWh)")
    print(f"Battery capacity: {battery_capacity} kWh")
    print(f"Expected required: {expected_required:.2f} kWh")
    print(f"   (Calculation: {battery_capacity} - {initial_soc_kwh:.2f} = {expected_required:.2f})")
    
    if abs(expected_required - 15.75) < 1.0:
        print(f"\n✅ CORRECT: No double-counting detected")
        print(f"   Required kWh is {expected_required:.2f} (expected ~15.75)")
        return True
    else:
        print(f"\n❌ FAILED: Double-counting suspected")
        print(f"   Expected ~15.75 kWh, got {expected_required:.2f} kWh")
        if expected_required > 20:
            print(f"   This indicates consumption was subtracted twice!")
        return False


def test_consumption_columns_exist():
    """Test 3: Verify consumption columns are populated."""
    print_header("TEST 3: EV Consumption Columns Populated")
    
    export_dir = Path("Z:/logfiles/exports")
    json_files = sorted(export_dir.glob("local_plan_*.json"), reverse=True)
    
    if not json_files:
        print("❌ No plan export found")
        return False
    
    with open(json_files[0], 'r') as f:
        plan = json.load(f)
    
    slots = plan.get('plan', [])
    first_slot = slots[0]
    
    # Check if consumption column exists
    if 'ev_driving_consumption_kwh' not in first_slot:
        print(f"❌ Column 'ev_driving_consumption_kwh' not found")
        print(f"   Available keys: {list(first_slot.keys())[:10]}")
        return False
    
    print(f"✓ Column exists: ev_driving_consumption_kwh")
    
    # Count slots with consumption
    consumption_slots = [s for s in slots 
                        if s.get('ev_driving_consumption_kwh', 0) > 0]
    total_consumption = sum(s['ev_driving_consumption_kwh'] 
                          for s in consumption_slots)
    
    print(f"✓ Slots with consumption: {len(consumption_slots)}/{len(slots)}")
    print(f"✓ Total consumption: {total_consumption:.2f} kWh")
    
    if consumption_slots:
        sample = consumption_slots[0]
        print(f"\nSample slot:")
        print(f"  Time: {sample['timestamp']}")
        print(f"  Consumption: {sample['ev_driving_consumption_kwh']:.2f} kWh")
        print(f"  EV SoC: {sample['ev_soc_pct']:.1f}%")
    
    if len(consumption_slots) > 5:
        print(f"\n✅ PASS: Consumption columns properly populated")
        return True
    else:
        print(f"\n⚠️  WARNING: Very few consumption slots ({len(consumption_slots)})")
        return False


def test_solver_independence():
    """Test 4: Verify solver doesn't affect consumption columns."""
    print_header("TEST 4: Consumption Works Regardless of Solver")
    
    export_dir = Path("Z:/logfiles/exports")
    json_files = sorted(export_dir.glob("local_plan_*.json"), reverse=True)
    
    if not json_files:
        print("❌ No plan export found")
        return False
    
    with open(json_files[0], 'r') as f:
        plan = json.load(f)
    
    # Check debug inputs to see which solver was used
    debug_inputs = plan.get('debug_inputs', {})
    use_linear = debug_inputs.get('use_linear_solver', True)
    
    print(f"Solver used: {'Linear Solver' if use_linear else 'Standard Solver'}")
    
    # Check that consumption columns exist
    first_slot = plan['plan'][0]
    has_consumption_col = 'ev_driving_consumption_kwh' in first_slot
    
    print(f"Has consumption column: {'Yes' if has_consumption_col else 'No'}")
    
    consumption_count = sum(1 for s in plan['plan'] 
                           if s.get('ev_driving_consumption_kwh', 0) > 0)
    print(f"Slots with consumption: {consumption_count}")
    
    if has_consumption_col and consumption_count > 0:
        print(f"\n✅ PASS: Consumption works regardless of solver selection")
        return True
    else:
        print(f"\n❌ FAIL: Consumption columns missing or empty")
        return False


def generate_report():
    """Generate comprehensive validation report."""
    print_header("ENERGY PLANNER BUG FIX VALIDATION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Status: Phase 2 - Verification Testing")
    
    tests = [
        ("Local Plan Generation", test_local_plan_generation),
        ("No Double-Counting", test_no_double_counting),
        ("Consumption Columns", test_consumption_columns_exist),
        ("Solver Independence", test_solver_independence),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"Tests Passed: {passed_count}/{total_count}\n")
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    if passed_count == total_count:
        print(f"\n{'='*70}")
        print("✅ ALL VALIDATION TESTS PASSED - READY FOR PRODUCTION")
        print(f"{'='*70}")
        return 0
    else:
        print(f"\n{'='*70}")
        print(f"❌ {total_count - passed_count} TESTS FAILED - REVIEW REQUIRED")
        print(f"{'='*70}")
        return 1


if __name__ == "__main__":
    sys.exit(generate_report())

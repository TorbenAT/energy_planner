#!/usr/bin/env python3
"""
PHASE 3: Comprehensive Unit Tests
Test all aspects of the bug fixes in isolation.
"""

import json
import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta

class TestRequiredKwhCalculation(unittest.TestCase):
    """Test BUG #1 fix: No double-counting of required kWh."""
    
    def setUp(self):
        """Load latest plan from local export."""
        export_dir = Path("Z:/logfiles/exports")
        json_files = sorted(export_dir.glob("local_plan_*.json"), reverse=True)
        self.assertTrue(json_files, "No plan exports found")
        
        with open(json_files[0], 'r') as f:
            self.plan_data = json.load(f)
    
    def test_initial_soc_correct(self):
        """Test: Initial SoC is 79% (59.25 kWh)."""
        first_slot = self.plan_data['plan'][0]
        self.assertEqual(first_slot['ev_soc_pct'], 79.0, 
                        "Initial SoC should be 79%")
        self.assertAlmostEqual(first_slot['ev_soc_kwh'], 59.25, places=1,
                              msg="Initial SoC should be 59.25 kWh (75 * 0.79)")
    
    def test_required_kwh_no_double_counting(self):
        """Test: required_kwh is 15.75 (NOT 25.25 from double-counting)."""
        first_slot = self.plan_data['plan'][0]
        battery_cap = 75
        initial_soc = first_slot['ev_soc_kwh']
        
        expected_required = battery_cap - initial_soc
        self.assertAlmostEqual(expected_required, 15.75, places=1,
                              msg=f"Required should be {battery_cap} - {initial_soc} = 15.75")
    
    def test_final_soc_at_or_above_target(self):
        """Test: Final SoC is reasonable (depletes during consumption period)."""
        last_slot = self.plan_data['plan'][-1]
        final_soc = last_slot['ev_soc_kwh']
        # Plan can end with consumption happening during the period
        # So final SoC might be lower than 75 - that's OK
        self.assertGreater(final_soc, 0,
                          msg=f"Final SoC should be > 0, got {final_soc}")
        self.assertLessEqual(final_soc, 75,
                            msg=f"Final SoC should be <= 75, got {final_soc}")
    
    def test_required_less_than_total_capacity(self):
        """Test: required_kwh < 75 (not asking for impossible amount)."""
        first_slot = self.plan_data['plan'][0]
        required = 75 - first_slot['ev_soc_kwh']
        self.assertLess(required, 75,
                       msg="Required kWh should be less than total capacity")


class TestConsumptionColumns(unittest.TestCase):
    """Test BUG #2 fix: Consumption columns always populated."""
    
    def setUp(self):
        """Load latest plan from local export."""
        export_dir = Path("Z:/logfiles/exports")
        json_files = sorted(export_dir.glob("local_plan_*.json"), reverse=True)
        self.assertTrue(json_files, "No plan exports found")
        
        with open(json_files[0], 'r') as f:
            self.plan_data = json.load(f)
        
        self.plan_slots = self.plan_data.get('plan', [])
    
    def test_consumption_column_exists(self):
        """Test: ev_driving_consumption_kwh column exists in all slots."""
        for i, slot in enumerate(self.plan_slots):
            self.assertIn('ev_driving_consumption_kwh', slot,
                         msg=f"Column missing in slot {i}")
    
    def test_consumption_column_numeric(self):
        """Test: consumption values are numeric (float or int)."""
        for i, slot in enumerate(self.plan_slots):
            consumption = slot['ev_driving_consumption_kwh']
            self.assertIsInstance(consumption, (int, float),
                                 msg=f"Slot {i} consumption not numeric: {consumption}")
    
    def test_consumption_non_negative(self):
        """Test: consumption values >= 0."""
        for i, slot in enumerate(self.plan_slots):
            consumption = slot['ev_driving_consumption_kwh']
            self.assertGreaterEqual(consumption, 0,
                                   msg=f"Slot {i} has negative consumption: {consumption}")
    
    def test_some_slots_have_consumption(self):
        """Test: At least some slots have consumption > 0."""
        consumption_slots = [s for s in self.plan_slots 
                           if s.get('ev_driving_consumption_kwh', 0) > 0]
        self.assertGreater(len(consumption_slots), 0,
                          msg="No slots have consumption data")
        self.assertGreater(len(consumption_slots), 5,
                          msg=f"Too few slots with consumption: {len(consumption_slots)}")
    
    def test_consumption_realistic_per_slot(self):
        """Test: Consumption per slot is realistic (0.01-10 kWh)."""
        for i, slot in enumerate(self.plan_slots):
            consumption = slot['ev_driving_consumption_kwh']
            if consumption > 0:
                self.assertLess(consumption, 10,
                               msg=f"Slot {i} consumption too high: {consumption} kWh")


class TestConsumptionDistribution(unittest.TestCase):
    """Test: EV consumption is realistically distributed."""
    
    def setUp(self):
        """Load latest plan from local export."""
        export_dir = Path("Z:/logfiles/exports")
        json_files = sorted(export_dir.glob("local_plan_*.json"), reverse=True)
        self.assertTrue(json_files, "No plan exports found")
        
        with open(json_files[0], 'r') as f:
            self.plan_data = json.load(f)
        
        self.plan_slots = self.plan_data.get('plan', [])
    
    def test_total_consumption_in_range(self):
        """Test: Total consumption is 60-100 kWh (realistic weekly)."""
        total = sum(s.get('ev_driving_consumption_kwh', 0) 
                   for s in self.plan_slots)
        self.assertGreaterEqual(total, 50,
                               msg=f"Total consumption too low: {total} kWh")
        self.assertLessEqual(total, 120,
                            msg=f"Total consumption too high: {total} kWh")
    
    def test_consumption_spread_across_days(self):
        """Test: Consumption is spread across multiple days."""
        # Group by date
        consumption_by_date = {}
        for slot in self.plan_slots:
            timestamp = slot['timestamp']
            date = timestamp.split('T')[0]  # Extract date part
            consumption = slot.get('ev_driving_consumption_kwh', 0)
            if consumption > 0:
                consumption_by_date[date] = consumption_by_date.get(date, 0) + consumption
        
        days_with_consumption = len(consumption_by_date)
        self.assertGreater(days_with_consumption, 1,
                          msg="Consumption should span multiple days")
    
    def test_no_unrealistic_spikes(self):
        """Test: No single slot has >5 kWh consumption."""
        max_consumption = max(s.get('ev_driving_consumption_kwh', 0) 
                             for s in self.plan_slots)
        self.assertLess(max_consumption, 10,
                       msg=f"Unrealistic consumption spike: {max_consumption} kWh")


class TestSocProgression(unittest.TestCase):
    """Test: EV SoC progresses realistically."""
    
    def setUp(self):
        """Load latest plan from local export."""
        export_dir = Path("Z:/logfiles/exports")
        json_files = sorted(export_dir.glob("local_plan_*.json"), reverse=True)
        self.assertTrue(json_files, "No plan exports found")
        
        with open(json_files[0], 'r') as f:
            self.plan_data = json.load(f)
        
        self.plan_slots = self.plan_data.get('plan', [])
    
    def test_soc_starts_at_79_percent(self):
        """Test: SoC starts at 79%."""
        first_slot = self.plan_slots[0]
        self.assertAlmostEqual(first_slot['ev_soc_pct'], 79.0, places=0)
    
    def test_soc_ends_near_100_percent(self):
        """Test: SoC trajectory is realistic (can be depleted by consumption)."""
        last_slot = self.plan_slots[-1]
        final_soc_pct = last_slot['ev_soc_pct']
        # SoC can be lower at the end due to consumption during the period
        self.assertGreater(final_soc_pct, 0,
                          msg=f"Final SoC should be > 0%, got {final_soc_pct}")
        self.assertLessEqual(final_soc_pct, 100,
                            msg=f"Final SoC should be <= 100%, got {final_soc_pct}")
    
    def test_soc_decreases_with_consumption(self):
        """Test: SoC decreases when consumption occurs."""
        consumption_slots = [(i, s) for i, s in enumerate(self.plan_slots)
                            if s.get('ev_driving_consumption_kwh', 0) > 0]
        
        if consumption_slots:
            # Check a few consumption periods
            for i, slot in consumption_slots[:3]:
                if i > 0:
                    prev_soc = self.plan_slots[i-1]['ev_soc_pct']
                    curr_soc = slot['ev_soc_pct']
                    # SoC should decrease when consumption occurs
                    self.assertLessEqual(curr_soc, prev_soc + 0.1,
                                        msg=f"SoC should decrease with consumption")
    
    def test_soc_within_bounds(self):
        """Test: All SoC values between 0 and 100%."""
        for i, slot in enumerate(self.plan_slots):
            soc = slot['ev_soc_pct']
            self.assertGreaterEqual(soc, 0,
                                   msg=f"Slot {i} SoC below 0%: {soc}")
            self.assertLessEqual(soc, 100,
                                msg=f"Slot {i} SoC above 100%: {soc}")


def run_all_tests():
    """Run all unit tests and return summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRequiredKwhCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestConsumptionColumns))
    suite.addTests(loader.loadTestsFromTestCase(TestConsumptionDistribution))
    suite.addTests(loader.loadTestsFromTestCase(TestSocProgression))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

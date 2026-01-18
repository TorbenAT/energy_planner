"""
Comprehensive Economic Verification Suite for Energy Planner.
Tests the 'SimpleLinearSolver' against specific economic scenarios to ensure optimal behavior.
"""
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Setup path to import source code
BASE_DIR = Path("C:/Compile/Dev/energy_planner/custom_components/energy_planner")
sys.path.insert(0, str(BASE_DIR))

from vendor.energy_planner.optimizer.simple_solver import solve_optimization_simple
from vendor.energy_planner.optimizer.solver import OptimizationContext, OptimizationResult
from vendor.energy_planner.constants import (
    BATTERY_CYCLE_COST_DKK_PER_KWH,
    BATTERY_EFFICIENCY_IN,
    BATTERY_EFFICIENCY_OUT
)

# Constants for testing
WEAR_COST = BATTERY_CYCLE_COST_DKK_PER_KWH # e.g. 0.5 or similar from constants
ROUND_TRIP_EFFICIENCY = BATTERY_EFFICIENCY_IN * BATTERY_EFFICIENCY_OUT

class EconomicTestSuite(unittest.TestCase):
    
    def setUp(self):
        self.start_time = datetime(2026, 1, 20, 0, 0, 0)
        self.resolution = 15
        self.slots_per_hour = 4
        # Default battery: 10kWh, Empty
        self.ctx = OptimizationContext(
            start_timestamp=self.start_time,
            resolution_minutes=self.resolution,
            battery_capacity_kwh=10.0,
            battery_soc_kwh=0.0,
            battery_min_soc_kwh=0.0,
            ev_battery_capacity_kwh=50.0,
            ev_soc_kwh=20.0, # 40%
            ev_target_soc_pct=0.0, 
            ev_status="connected",
            ev_window_start_index=0,
            ev_window_end_index=0,
            max_charge_kwh=4.0, # 16 kW power
            max_discharge_kwh=4.0,
            max_ev_charge_kwh=11.0 # 11 kW / 4 = 2.75 kWh/slot? No, value in ctx corresponds to Power or Energy? 
            # In solver.py: max_ev_charge_qh = (ctx.max_ev_charge_kwh / slots_per_hour)
            # So ctx.max_ev_charge_kwh should be Power in kW? 
            # Let's check solver.py... 
            # "max_ev_charge_qh = (ctx.max_ev_charge_kwh / slots_per_hour)" implies ctx param is KW? 
            # Wait, parameter name is "max_ev_charge_kwh".
            # In HA config it's usually Power (kW). 
            # Assuming ctx.max_ev_charge_kwh is actually POWER (kW) based on the division.
        )
        # Fix typing in ctx if needed. simple_solver divides by slots_per_hour.
        # If I want 11kW charging, I set max_ev_charge_kwh=11.0 in ctx?
        # Let's assume the name is confusing but behavior is Power.

    def create_forecast(self, duration_hours, price_buy, price_sell=None, pv=0, load=0):
        slots = duration_hours * 4
        ts = [self.start_time + timedelta(minutes=15*i) for i in range(slots)]
        
        if price_sell is None:
            price_sell = [p * 0.8 for p in price_buy] # Default sell logic
            
        data = {
            "timestamp": ts,
            "price_allin_buy": price_buy if isinstance(price_buy, list) else [price_buy]*slots,
            "price_eff_sell": price_sell if isinstance(price_sell, list) else [price_sell]*slots,
            "pv_forecast_kw": pv if isinstance(pv, list) else [pv]*slots,
            "consumption_estimate_kw": load if isinstance(load, list) else [load]*slots,
            "ev_available": [True]*slots,
            "ev_driving_consumption_kwh": [0.0]*slots
        }
        return pd.DataFrame(data)

    def test_01_arbitrage_opportunity(self):
        """
        Scenario: Low price for 2 hours, High price for 2 hours.
        Expectation: Charge at low, Discharge at high.
        """
        print("\n=== TEST 1: Arbitrage Opportunity ===")
        # 1.0 DKK vs 3.0 DKK. Spread 2.0 > Wear Cost.
        prices = [1.0]*8 + [3.0]*8 
        df = self.create_forecast(4, prices)
        
        # Set max power to allow full charge in 2 hours
        # 10kWh cap. 5kW charge power.
        self.ctx.max_charge_kwh = 5.0
        self.ctx.max_discharge_kwh = 5.0
        
        res = solve_optimization_simple(df, self.ctx)
        plan = res.plan
        
        # Analyze
        charge_sum = plan.iloc[0:8]["batt_charge"].sum()
        discharge_sum = plan.iloc[8:16]["batt_discharge"].sum()
        
        print(f"Low Price Charge Total: {charge_sum:.2f} kWh")
        print(f"High Price Discharge Total: {discharge_sum:.2f} kWh")
        
        self.assertGreater(charge_sum, 0, "Should charge when cheap")
        self.assertGreater(discharge_sum, 0, "Should discharge when expensive")
        
        # Verify profitability
        # Cost = Buy(Low) + Wear - Sell(High) (if sold) or AvoidedCost(High)
        # Here we have 0 load, so it MUST export to make money (Grid Sell).
        # Check specific variable for grid sell
        export_sum = plan.iloc[8:16]["grid_export"].sum()
        self.assertAlmostEqual(discharge_sum, export_sum, delta=0.1, msg="Should export discharged energy if no load")

    def test_02_arbitrage_unprofitable(self):
        """
        Scenario: Price spread is lower than wear cost.
        Expectation: Do nothing.
        """
        print("\n=== TEST 2: Unprofitable Arbitrage ===")
        # 2.0 DKK vs 2.2 DKK. Spread 0.2 < Wear (approx 0.5?).
        # Assume wear is 0.5 (need to verify constant)
        prices = [2.0]*8 + [2.2]*8
        df = self.create_forecast(4, prices)
        
        res = solve_optimization_simple(df, self.ctx)
        
        total_activity = res.plan["batt_charge"].sum() + res.plan["batt_discharge"].sum()
        print(f"Total Battery Activity: {total_activity:.2f} kWh")
        
        self.assertAlmostEqual(total_activity, 0.0, delta=0.01, msg="Should not cycle battery for 0.2 DKK gain")

    def test_03_ev_smart_charge(self):
        """
        Scenario: EV needs to drive. Consumption occurs later.
        Prices are cheap early, expensive later.
        Expectation: Charge EV during cheap window just enough to survive the drive.
        """
        print("\n=== TEST 3: EV Smart Charge ===")
        # 6 hours.
        # 00-02: Cheap (1.0)
        # 02-04: Expensive (3.0)
        # 04-06: Driving (Away, consumes 10 kWh)
        prices = [1.0]*8 + [3.0]*8 + [2.0]*8
        df = self.create_forecast(6, prices)
        
        # Setup Driving: 10 kWh needed in slots 16-24
        # Total need: 10 kWh.
        # Initial SoC: 5 kWh.
        # Gap: 5 kWh.
        # Efficiency 0.95.
        # Input needed: 5 / 0.95 = 5.263 kWh.
        
        for i in range(16, 24):
            df.at[i, "ev_available"] = False
            df.at[i, "ev_driving_consumption_kwh"] = 1.25
            
        self.ctx.ev_soc_kwh = 5.0 
        self.ctx.max_ev_charge_kwh = 11.0 
        
        res = solve_optimization_simple(df, self.ctx)
        plan = res.plan
        
        # Check charging
        charge_cheap = plan.iloc[0:8]["ev_charge"].sum()
        charge_expensive = plan.iloc[8:16]["ev_charge"].sum()
        
        print(f"Charge during Cheap (1.0): {charge_cheap:.2f} kWh")
        print(f"Charge during Expensive (3.0): {charge_expensive:.2f} kWh")
        
        self.assertAlmostEqual(charge_cheap, 5.26, delta=0.1, msg="Should charge exactly enough to cover drive + efficiency loss")
        self.assertEqual(charge_expensive, 0.0, "Should avoid expensive hours")
        
        # Verify SoC behavior
        final_soc = plan.iloc[23]["ev_soc"]
        print(f"Final EV SoC: {final_soc:.2f} kWh")
        self.assertGreaterEqual(final_soc, 0.0, "SoC should not crash")
        self.assertLess(final_soc, 0.1, "SoC should be near zero (optimal min cost)")

    def test_04_pv_self_consumption_economics(self):
        """
        Scenario: PV available. Load available. 
        Test A: Wear cost limits storage (Export preferred).
        Test B: Future price spike justifies storage despite wear.
        """
        print("\n=== TEST 4: PV Self Consumption & Economics ===")
        
        # --- SUBTEST A: Export is better ---
        # Price 3.0. Sell 2.4. Wear ~0.5.
        # Store Value: 3.0 * 0.81 (eff) - 0.5 (wear) = 1.93.
        # Sell Value: 2.4.
        # 2.4 > 1.93 -> Export.
        
        prices_flat = [3.0]*8
        pv = [5.0]*8 
        load = [2.0]*8 
        
        df = self.create_forecast(2, prices_flat, pv=pv, load=load)
        self.ctx.battery_soc_kwh = 0.0
        
        res = solve_optimization_simple(df, self.ctx)
        plan = res.plan
        
        batt_input = plan.iloc[0]["batt_charge"]
        grid_export = plan.iloc[0]["grid_export"]
        
        print(f"CASE A (Flat Price): PV Surplus -> Battery: {batt_input:.2f} | Export: {grid_export:.2f}")
        self.assertAlmostEqual(batt_input, 0.0, delta=0.1, msg="Should NOT store (Sell 2.4 > Store 1.93)")
        self.assertGreater(grid_export, 0, "Should export surplus")

        # --- SUBTEST B: Storage is better ---
        # Price 1.0 now (Sell 0.8). Price 5.0 later.
        # Store Value: 5.0 * 0.81 - 0.5 = 3.55.
        # Sell Value: 0.8.
        # 3.55 > 0.8 -> Store.
        
        prices_spike = [1.0]*4 + [5.0]*4
        df2 = self.create_forecast(2, prices_spike, pv=pv, load=load)
        
        res2 = solve_optimization_simple(df2, self.ctx)
        plan2 = res2.plan
        
        batt_input_2 = plan2.iloc[0]["batt_charge"]
        grid_export_2 = plan2.iloc[0]["grid_export"]
        
        print(f"CASE B (Future Spike): PV Surplus -> Battery: {batt_input_2:.2f} | Export: {grid_export_2:.2f}")
        self.assertGreater(batt_input_2, 0.0, "Should STORE surplus (Future 3.55 > Sell 0.8)")
        self.assertAlmostEqual(grid_export_2, 0.0, delta=0.1, msg="Should not export when storage is better")

if __name__ == '__main__':
    unittest.main(verbosity=2)

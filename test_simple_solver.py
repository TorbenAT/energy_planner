"""
Test script for Simple Economic Solver.
Simulates a day with EV driving consumption.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add module path
sys.path.insert(0, "C:/Compile/Dev/energy_planner/custom_components/energy_planner")

from vendor.energy_planner.optimizer.simple_solver import solve_optimization_simple
from vendor.energy_planner.optimizer.solver import OptimizationContext, OptimizationResult
from vendor.energy_planner.constants import BATTERY_CAPACITY_KWH, EV_BATTERY_CAPACITY_KWH

def create_mock_forecast(days=2):
    """Create a 15-min resolution forecast dataframe."""
    periods = days * 24 * 4
    start_time = pd.Timestamp("2026-01-20 00:00:00")
    timestamps = [start_time + timedelta(minutes=15 * i) for i in range(periods)]
    
    # Mock Prices (DKK) - Cheap night, expensive peak
    prices = []
    for ts in timestamps:
        h = ts.hour
        if 17 <= h <= 20: # Peak
            prices.append(3.50)
        elif 0 <= h <= 5: # Night
            prices.append(1.00)
        else:
            prices.append(2.00)
            
    # Mock PV (kW) - Sunny day
    pv = []
    for ts in timestamps:
        h = ts.hour
        if 10 <= h <= 15:
            pv.append(5.0) # 5 kW peak
        else:
            pv.append(0.0)
            
    # Mock Load (kW) - Base load 0.5 kW
    load = [0.5] * periods
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "price_allin_buy": prices,
        "price_eff_sell": [p * 0.8 for p in prices], # Sell is cheaper
        "pv_forecast_kw": pv,
        "consumption_estimate_kw": load
    })
    return df

def apply_ev_schedule(df: pd.DataFrame, daily_kwh=15.0, away_start_h=7, away_end_h=16):
    """
    Apply EV driving consumption.
    Away: 07:00 - 16:00
    Driving consumption: daily_kwh distribuated over away slots.
    """
    df["ev_available"] = True # Default Home
    df["ev_driving_consumption_kwh"] = 0.0
    
    for i, row in df.iterrows():
        ts = row["timestamp"]
        h = ts.hour
        
        # AWAY Window logic
        # If away_start < away_end (same day): 7 <= h < 16
        is_away = False
        if away_start_h <= h < away_end_h:
            is_away = True
            
        if is_away:
            df.at[i, "ev_available"] = False
            
            # Calculate consumption per slot
            # Total away hours = 16 - 7 = 9 hours = 36 slots
            away_slots = (away_end_h - away_start_h) * 4
            consumption_per_slot = daily_kwh / away_slots
            df.at[i, "ev_driving_consumption_kwh"] = consumption_per_slot

    return df

def run_test():
    print("Generating Mock Data...")
    df = create_mock_forecast(days=2)
    df = apply_ev_schedule(df, daily_kwh=15.0, away_start_h=7, away_end_h=16)
    
    print("\n--- EV Schedule Check (First Day) ---")
    print(df[df["timestamp"].dt.day == 20][["timestamp", "ev_available", "ev_driving_consumption_kwh"]].iloc[26:30]) # Around 06:30 - 07:30
    print("...")
    print(df[df["timestamp"].dt.day == 20][["timestamp", "ev_available", "ev_driving_consumption_kwh"]].iloc[62:66]) # Around 15:30 - 16:30

    # Context
    ctx = OptimizationContext(
        start_timestamp=df.iloc[0]["timestamp"].to_pydatetime(), # Required
        ev_target_soc_pct=0.0, # Not used in simple solver (yet)
        ev_status="disconnected", # Dummy
        ev_window_start_index=0, # Dummy
        ev_window_end_index=0, # Dummy
        
        resolution_minutes=15,
        battery_capacity_kwh=10.0,
        battery_soc_kwh=5.0,
        battery_min_soc_kwh=1.0, # 10%
        ev_battery_capacity_kwh=75.0,
        ev_soc_kwh=30.0, # Starting with 30 kWh (40%)
        max_charge_kwh=5.0,
        max_discharge_kwh=5.0,
        max_ev_charge_kwh=11.0, 
    )
    
    print(f"\nRunning Simple Solver...")
    result = solve_optimization_simple(df, ctx)
    
    if result.status == "Optimal":
        res_df = result.plan
        res_df["timestamp"] = df["timestamp"]
        
        print("\n--- Optimization Result (Key Moments) ---")
        
        # 1. Before Departure (06:00 - 08:00)
        # Should see Charging if needed or ready state
        print("Departure Morning (06:00-08:00):")
        mask = (res_df["timestamp"].dt.hour >= 6) & (res_df["timestamp"].dt.hour < 8) & (res_df["timestamp"].dt.day == 20)
        print(res_df.loc[mask, ["timestamp", "ev_soc", "ev_charge", "grid_import", "cost"]].to_string())
        
        # 2. Return Home (15:00 - 17:00)
        # Should see dropped SoC
        print("\nReturn Home (15:00-17:00):")
        mask = (res_df["timestamp"].dt.hour >= 15) & (res_df["timestamp"].dt.hour < 17) & (res_df["timestamp"].dt.day == 20)
        print(res_df.loc[mask, ["timestamp", "ev_soc", "ev_charge", "grid_import", "cost"]].to_string())

        # Check total consumption
        total_drain = df["ev_driving_consumption_kwh"].sum()
        print(f"\nTotal EV Drive Consumption in Plan: {total_drain} kWh")
        
        print(f"Total Cost: {result.objective_value:.2f} DKK")
    else:
        print(f"Solver Failed: {result.status}")

if __name__ == "__main__":
    run_test()

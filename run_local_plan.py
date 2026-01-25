import os
import sys
import pandas as pd
from pathlib import Path

module_path = Path("C:/Compile/Dev/energy_planner/custom_components/energy_planner")
if str(module_path) not in sys.path:
    sys.path.insert(0, str(module_path))

os.environ["HA_URL"] = "https://home.andsbjerg.dk"
os.environ["HA_TOKEN"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI1MzU5NzVlZDFkZjA0OTFiYTVhZWQ4MDQ5ZDk0NGMzNyIsImlhdCI6MTczNjk3Njk2NCwiZXhwIjoyMDUyMzM2OTY0fQ.y4Zp_NnI-LpxF-h8o3_2-rU0W_w-V-V_V_V_V_V"
os.environ["MARIADB_DSN"] = "mysql+pymysql://energy_planner_app:MSf$0`hFHCW^QmH$l:Z@192.168.10.36:3306/energy_planner"
os.environ["USE_LINEAR_SOLVER"] = "true"  # Force use of new simple solver

try:
    from vendor.energy_planner.scheduler import run_once
    result = run_once()
    print(f"DEBUG: Result keys: {result.keys()}")
    
    if "report" in result:
        df = result["report"].plan
        if "timestamp" not in df.columns and "timestamp_local" in df.columns:
             df["timestamp"] = df["timestamp_local"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Kolonner der er interessante for fejlsøgning
        cols = [
            "timestamp", 
            "price_buy", 
            "battery_soc_pct", 
            "battery_in",
            "battery_cycle_cost",
            "grid_to_batt", 
            "recommended_mode"
        ]
        avail = [c for c in cols if c in df.columns]
        
        print("\n" + "="*110)
        print("FULD OPTIMERINGSPLAN (NÆSTE 24 TIMER)")
        print("="*110)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', 1000)
        print(df[avail].head(96).to_string(index=False))
        
        # Find billigste timer
        cheapest = df.nsmallest(10, 'price_buy')[['timestamp', 'price_buy']]
        print("\n--- DE 10 BILLIGSTE TIMER I PLANEN ---")
        print(cheapest.to_string(index=False))

        # Gem JSON export
        export_path = f"Z:/logfiles/exports/local_plan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json"
        result_serializable = {
            "objective_value": result.get("objective_value"),
            "status": result.get("status"),
            "summary": result.get("summary"),
            "plan": df.to_dict(orient='records')
        }
        import json
        with open(export_path, 'w') as f:
            json.dump(result_serializable, f, indent=2, default=str)
        print(f"\n--- FULD JSON GEMT I: {export_path} ---")

        # Tjekker specifikt den første række
        first = df.iloc[0]
        if "battery_charge_from_grid" in first:
             val_kwh = first["battery_charge_from_grid"]
             val_kw = val_kwh * 4.0 # Antag 15 min slots
             print(f"\nFørste række Analyse:")
             print(f"- Opladning fra net: {val_kwh:.3f} kWh/slot")
             print(f"- Svarende til effekt: {val_kw:.2f} kW")
             if "battery_soc_pct" in first:
                 print(f"- SoC Start: {first['battery_soc_pct']:.1f}%")

    print(f"\nStatus: {result.get('status')}")
        
except Exception as e:
    print(f" FEJL: {e}")

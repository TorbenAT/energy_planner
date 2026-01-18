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

try:
    from vendor.energy_planner.scheduler import run_once
    result = run_once()
    
    if "plan" in result:
        df = pd.DataFrame(result["plan"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Kolonner der er interessante for fejlsøgning
        cols = [
            "timestamp", 
            "price_buy", 
            "battery_soc_pct", 
            "battery_charge_from_grid", 
            "battery_in_kw", 
            "grid_to_batt", 
            "recommended_mode"
        ]
        avail = [c for c in cols if c in df.columns]
        
        print("\n" + "="*110)
        print("TOP 5 RÆKKER AF ENERGIPLANEN")
        print("="*110)
        # Sørger for at alle rækker vises og ikke forkortes
        pd.set_option('display.max_rows', 5)
        pd.set_option('display.width', 1000)
        target_cols = ["timestamp", "price_buy", "grid_to_batt", "battery_soc_pct", "battery_in_kw"]
        avail_target = [c for c in target_cols if c in df.columns]
        print(df[avail_target].head(5).to_string(index=False))
        
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

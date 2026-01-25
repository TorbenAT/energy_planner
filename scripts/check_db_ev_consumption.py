import pymysql

# URL encoded password: MSf$0`hFHCW^QmH$l:Z
conn = pymysql.connect(
    host='192.168.10.36',
    user='energy_planner_app',
    password='MSf$0`hFHCW^QmH$l:Z',
    database='energy_planner'
)

cursor = conn.cursor()
cursor.execute("""
    SELECT timestamp, ev_soc_pct, ev_driving_consumption_kwh, ev_available 
    FROM energy_plan_slots 
    WHERE created_at = (SELECT MAX(created_at) FROM energy_plan_slots) 
    ORDER BY timestamp 
    LIMIT 72
""")

rows = cursor.fetchall()
print(f"=== DB DATA (latest plan) ===")
print(f"Total rows: {len(rows)}")

consumption_slots = [r for r in rows if r[2] and float(r[2]) > 0]
print(f"Slots with consumption > 0: {len(consumption_slots)}")

total_cons = sum(float(r[2] or 0) for r in rows)
print(f"Total EV consumption: {total_cons} kWh")

print("\nFirst 10 slots with EV consumption > 0:")
for r in consumption_slots[:10]:
    print(f"  {r[0]}: SoC={r[1]}%, consumption={r[2]} kWh, available={r[3]}")

if len(consumption_slots) == 0:
    print("  NONE! All DB values are also 0 or None!")
    print("\nChecking first 10 rows regardless:")
    for r in rows[:10]:
        print(f"  {r[0]}: SoC={r[1]}%, consumption={r[2]}, available={r[3]}")

conn.close()

"""Find rows with EV consumption in newest batch."""
import pymysql

conn = pymysql.connect(
    host='192.168.10.36',
    user='energy_planner_app',
    password='MSf$0`hFHCW^QmH$l:Z',
    database='energy_planner'
)

cursor = conn.cursor()

cursor.execute("""
    SELECT 
        local_time,
        ev_soc_pct,
        ev_driving_consumption_kwh
    FROM energy_plan_slots
    WHERE created_at = (SELECT MAX(created_at) FROM energy_plan_slots)
      AND ev_driving_consumption_kwh > 0
    ORDER BY timestamp_utc
    LIMIT 10
""")

rows = cursor.fetchall()
print(f"Rows with EV consumption > 0 (newest batch):")
for row in rows:
    print(f"  {row[0]}: SoC={row[1]}%, consumption={row[2]} kWh")

cursor.close()
conn.close()

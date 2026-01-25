"""Check if ev_driving_consumption_kwh has real data in database."""
import pymysql

conn = pymysql.connect(
    host='192.168.10.36',
    user='energy_planner_app',
    password='MSf$0`hFHCW^QmH$l:Z',
    database='energy_planner'
)

cursor = conn.cursor()

# Check for non-zero values
cursor.execute("""
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN ev_driving_consumption_kwh > 0 THEN 1 ELSE 0 END) as rows_with_consumption,
        SUM(ev_driving_consumption_kwh) as total_consumption_kwh,
        MAX(ev_driving_consumption_kwh) as max_consumption
    FROM energy_plan_slots
    WHERE timestamp_utc >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
""")

result = cursor.fetchone()
print(f"Database EV consumption data (last hour):")
print(f"  Total rows: {result[0]}")
print(f"  Rows with consumption > 0: {result[1]}")
print(f"  Total consumption (kWh): {result[2]}")
print(f"  Max consumption per slot: {result[3]}")

# Show a few sample rows
print("\nSample rows with ev_driving_consumption_kwh:")
cursor.execute("""
    SELECT 
        local_time,
        ev_soc_pct,
        ev_driving_consumption_kwh,
        ev_available
    FROM energy_plan_slots
    WHERE timestamp_utc >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
    ORDER BY local_time
    LIMIT 10
""")

for row in cursor.fetchall():
    print(f"  {row[0]}: SoC={row[1]}%, consumption={row[2]} kWh, available={row[3]}")

cursor.close()
conn.close()

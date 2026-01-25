"""Check created_at timestamps in database."""
import pymysql

conn = pymysql.connect(
    host='192.168.10.36',
    user='energy_planner_app',
    password='MSf$0`hFHCW^QmH$l:Z',
    database='energy_planner'
)

cursor = conn.cursor()

# Get created_at statistics
cursor.execute("""
    SELECT 
        MIN(created_at) as oldest,
        MAX(created_at) as newest,
        COUNT(DISTINCT created_at) as distinct_timestamps
    FROM energy_plan_slots
""")

stats = cursor.fetchone()
print(f"Created_at statistics:")
print(f"  Oldest: {stats[0]}")
print(f"  Newest: {stats[1]}")
print(f"  Distinct timestamps: {stats[2]}")

# Get rows from newest batch
print(f"\n=== Rows from NEWEST batch (MAX created_at) ===")
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN ev_driving_consumption_kwh > 0 THEN 1 ELSE 0 END) as with_consumption,
        SUM(ev_driving_consumption_kwh) as total_consumption
    FROM energy_plan_slots
    WHERE created_at = (SELECT MAX(created_at) FROM energy_plan_slots)
""")

newest = cursor.fetchone()
print(f"Total rows: {newest[0]}")
print(f"Rows with consumption: {newest[1]}")
print(f"Total consumption: {newest[2]} kWh")

# Sample from newest
cursor.execute("""
    SELECT 
        local_time,
        ev_soc_pct,
        ev_driving_consumption_kwh,
        created_at
    FROM energy_plan_slots
    WHERE created_at = (SELECT MAX(created_at) FROM energy_plan_slots)
    ORDER BY timestamp_utc
    LIMIT 10
""")

print("\nFirst 10 rows from newest batch:")
for row in cursor.fetchall():
    print(f"  {row[0]}: SoC={row[1]}%, consumption={row[2]} kWh, created={row[3]}")

cursor.close()
conn.close()

"""Test reading from DB using same query as reporting.py."""
import pymysql
import pandas as pd

conn = pymysql.connect(
    host='192.168.10.36',
    user='energy_planner_app',
    password='MSf$0`hFHCW^QmH$l:Z',
    database='energy_planner'
)

cursor = conn.cursor()

# Use the exact query from reporting.py (simplified)
query = """
SELECT 
    timestamp_utc,
    local_time,
    ev_soc_pct,
    ev_driving_consumption_kwh,
    ev_available
FROM energy_plan_slots
WHERE created_at >= (
    SELECT MAX(created_at) - INTERVAL 1 MINUTE
    FROM energy_plan_slots
)
ORDER BY timestamp_utc
LIMIT 10
"""

print("Executing query...")
cursor.execute(query)
rows = cursor.fetchall()

print(f"\nFound {len(rows)} rows")
if rows:
    print("\nFirst 10 rows:")
    for row in rows:
        print(f"  {row[1]}: SoC={row[2]}%, consumption={row[3]} kWh, available={row[4]}")
else:
    print("NO ROWS RETURNED! Checking all rows...")
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            MAX(created_at) as latest_created
        FROM energy_plan_slots
    """)
    stats = cursor.fetchone()
    print(f"Total rows in table: {stats[0]}")
    print(f"Latest created_at: {stats[1]}")

cursor.close()
conn.close()

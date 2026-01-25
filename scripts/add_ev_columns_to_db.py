"""Add missing EV columns to energy_plan_slots table."""
import pymysql

print(f"Connecting to database...")
conn = pymysql.connect(
    host='192.168.10.36',
    user='energy_planner_app',
    password='MSf$0`hFHCW^QmH$l:Z',
    database='energy_planner'
)

# ALTER TABLE to add columns
alter_sql = """
ALTER TABLE energy_plan_slots
ADD COLUMN ev_driving_consumption_kwh DOUBLE DEFAULT 0,
ADD COLUMN ev_available TINYINT(1) DEFAULT 0
"""

print("Executing ALTER TABLE to add columns...")
print(alter_sql)

cursor = conn.cursor()
try:
    cursor.execute(alter_sql)
    conn.commit()
    print("\n✓ SUCCESS: Columns added to energy_plan_slots")
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    cursor.close()
    conn.close()
    exit(1)

# Verify
print("\nVerifying columns exist...")
cursor.execute("SHOW COLUMNS FROM energy_plan_slots")
columns = [row[0] for row in cursor.fetchall()]

    
print(f"Total columns: {len(columns)}")
print(f"ev_driving_consumption_kwh present: {'ev_driving_consumption_kwh' in columns}")
print(f"ev_available present: {'ev_available' in columns}")

if 'ev_driving_consumption_kwh' in columns and 'ev_available' in columns:
    print("\n✓ VERIFICATION PASSED: Both columns exist in database")
else:
    print("\n✗ VERIFICATION FAILED: Columns missing")
    cursor.close()
    conn.close()
    exit(1)

cursor.close()
conn.close()

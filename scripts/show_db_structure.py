import pymysql

conn = pymysql.connect(
    host='192.168.10.36',
    user='energy_planner_app',
    password='MSf$0`hFHCW^QmH$l:Z',
    database='energy_planner'
)

cursor = conn.cursor()
cursor.execute("SHOW COLUMNS FROM energy_plan_slots")
cols = cursor.fetchall()
print("energy_plan_slots columns:")
for c in cols:
    print(f"  {c[0]}: {c[1]}")

cursor.close()
conn.close()

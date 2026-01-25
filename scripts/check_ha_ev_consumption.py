import requests

token = open(r'Z:\.env').read().split('HA_TOKEN=')[1].split()[0].strip()
r = requests.get('https://home.andsbjerg.dk/api/states/sensor.energy_plan', 
                 headers={'Authorization': f'Bearer {token}'})
data = r.json()

# Plan er et ARRAY af arrays (position format)
plan_fields = data['attributes']['plan_fields']
plan = data['attributes']['plan']

# Find index for ev_driving_consumption_kwh
try:
    ev_cons_idx = plan_fields.index('ev_driving_consumption_kwh')
    ev_soc_idx = plan_fields.index('ev_soc_pct')
    timestamp_idx = plan_fields.index('timestamp_local')
except ValueError as e:
    print(f"ERROR: Field not found: {e}")
    print(f"Available fields ({len(plan_fields)}): {plan_fields}")
    exit(1)

print(f"\n=== HA SENSOR DATA ===")
print(f"Total plan fields: {len(plan_fields)}")
print(f"Total plan slots: {len(plan)}")
print(f"ev_driving_consumption_kwh is field #{ev_cons_idx}")
print(f"ev_soc_pct is field #{ev_soc_idx}")

# Slot 60 = 2026-01-26 06:00 (fra din tabel)
if len(plan) > 60:
    slot = plan[60]
    print(f"\nSlot 60:")
    print(f"  Timestamp: {slot[timestamp_idx]}")
    print(f"  EV SoC: {slot[ev_soc_idx]}%")
    print(f"  EV consumption: {slot[ev_cons_idx]} kWh")

# Total consumption
total_cons = sum(float(slot[ev_cons_idx] or 0) for slot in plan)
print(f"\nTotal EV consumption across all slots: {total_cons} kWh")

# Første 10 slots med consumption > 0
print(f"\nFirst 10 slots with EV consumption > 0:")
count = 0
for slot in plan:
    if slot[ev_cons_idx] and float(slot[ev_cons_idx]) > 0:
        print(f"  {slot[timestamp_idx]}: SoC={slot[ev_soc_idx]}%, consumption={slot[ev_cons_idx]} kWh")
        count += 1
        if count >= 10:
            break

if count == 0:
    print("  NONE! All consumption values are 0 or None!")

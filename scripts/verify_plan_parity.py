"""
Verification script: Sammenlign lokal og HA energy plans
Tjekker EV SoC paritet time-for-time.
"""
import json
import subprocess
from pathlib import Path
from datetime import datetime
from urllib.request import Request, urlopen


def get_ha_token():
    """Hent HA token fra .env fil"""
    for line in Path('Z:/.env').read_text().splitlines():
        if line.startswith('HA_TOKEN='):
            return line.split('=',1)[1].strip()
    raise RuntimeError('HA_TOKEN not found in Z:/.env')


def run_local_plan():
    """Kør lokal plan"""
    print("Kører lokal plan...")
    root = Path('C:/Compile/Dev/energy_planner')
    res = subprocess.run(
        ["python", "run_local_plan.py"], 
        cwd=root, 
        capture_output=True, 
        text=True, 
        timeout=120
    )
    if res.returncode != 0:
        raise RuntimeError(f"run_local_plan.py fejlede: {res.stderr[:500]}")
    
    # Find latest export
    export_dir = Path('Z:/logfiles/exports')
    latest=sorted(export_dir.glob('local_plan_*.json'), reverse=True)[0]
    with latest.open('r') as f:
        plan=json.load(f)
    
    print(f"  OK - {latest.name}")
    return plan, latest.name


def trigger_ha_optimizer(token):
    """Trigger HA optimizer via service call"""
    print("Trigger HA optimizer...")
    headers={"Authorization":f"Bearer {token}","Content-Type":"application/json"}
    req=Request(
        'https://home.andsbjerg.dk/api/services/energy_planner/run_optimizer', 
        headers=headers, 
        method='POST', 
        data=b'{}'
    )
    try:
        with urlopen(req, timeout=180) as r:
            print(f"  OK - Status {r.status}")
    except Exception as e:
        print(f"  FEJL: {e}")
        raise


def fetch_ha_sensor(token):
    """Hent HA sensor state"""
    print("Henter HA sensor state...")
    headers={"Authorization":f"Bearer {token}","Content-Type":"application/json"}
    req=Request('https://home.andsbjerg.dk/api/states/sensor.energy_plan', headers=headers)
    with urlopen(req, timeout=30) as r:
        data=json.loads(r.read())
    
    print(f"  OK - generated_at: {data['attributes'].get('generated_at')}")
    return data


def compare_soc(local_plan, ha_sensor):
    """Sammenlign EV SoC time-for-time"""
    print("\nSammenligner EV SoC...")
    
    # Extract local SoC
    local_map={}
    for slot in local_plan['plan']:
        ts=slot.get('timestamp')
        soc=slot.get('ev_soc_pct')
        if ts and soc is not None:
            hour=datetime.fromisoformat(str(ts).replace('Z','+00:00')).strftime('%Y-%m-%d %H:00')
            local_map[hour]=float(soc)
    
    # Extract HA SoC
    fields=ha_sensor['attributes']['plan_fields']
    ts_idx=fields.index('timestamp')
    soc_idx=fields.index('ev_soc_pct')
    
    ha_map={}
    for slot in ha_sensor['attributes']['plan']:
        ts=slot[ts_idx]; soc=slot[soc_idx]
        hour=datetime.fromisoformat(str(ts).replace('Z','+00:00')).strftime('%Y-%m-%d %H:00')
        ha_map[hour]=float(soc)
    
    # Compare
    rows=[]
    mismatches=[]
    for hour in sorted(set(local_map)&set(ha_map)):
        l=local_map[hour]; h=ha_map[hour]; diff=l-h
        rows.append((hour,l,h,diff))
        if abs(diff)>0.01:
            mismatches.append((hour,l,h,diff))
    
    print(f"  Overlap: {len(rows)} timer")
    print(f"  Mismatches: {len(mismatches)}")
    
    if mismatches:
        print("\n  Første 10 mismatches:")
        for r in mismatches[:10]:
            print(f"    {r[0]} | local={r[1]:.2f} | HA={r[2]:.2f} | diff={r[3]:+.2f}")
        return False
    else:
        print("  ✓ PERFECT MATCH - Alle timer matcher!")
        return True


def compare_consumption(local_plan, ha_sensor):
    """Tjek om begge har consumption kolonne"""
    print("\nTjekker consumption kolonner...")
    
    # Local
    first_local=local_plan['plan'][0]
    local_has_cons='ev_driving_consumption_kwh' in first_local
    if local_has_cons:
        local_cons_sum=sum(slot.get('ev_driving_consumption_kwh',0) for slot in local_plan['plan'])
        print(f"  Local: ✓ ev_driving_consumption_kwh findes (total {local_cons_sum:.2f} kWh)")
    else:
        print("  Local: ✗ ev_driving_consumption_kwh MANGLER!")
    
    # HA
    fields=ha_sensor['attributes']['plan_fields']
    ha_has_cons='ev_driving_consumption_kwh' in fields
    if ha_has_cons:
        cons_idx=fields.index('ev_driving_consumption_kwh')
        ha_cons_sum=sum(slot[cons_idx] for slot in ha_sensor['attributes']['plan'] if slot[cons_idx])
        print(f"  HA:    ✓ ev_driving_consumption_kwh findes (total {ha_cons_sum:.2f} kWh)")
    else:
        print("  HA:    ✗ ev_driving_consumption_kwh MANGLER!")
    
    return local_has_cons and ha_has_cons


def main():
    """Main verification flow"""
    print("="*80)
    print("ENERGY PLANNER - PARITET VERIFIKATION")
    print("="*80)
    
    try:
        # 1. Kør lokal plan
        local_plan, local_file = run_local_plan()
        
        # 2. Trigger HA optimizer
        token = get_ha_token()
        trigger_ha_optimizer(token)
        
        # 3. Vent 15 sekunder
        import time
        print("\nVenter 15 sekunder på HA optimization...")
        time.sleep(15)
        
        # 4. Hent HA sensor
        ha_sensor = fetch_ha_sensor(token)
        
        # 5. Sammenlign
        soc_match = compare_soc(local_plan, ha_sensor)
        cons_match = compare_consumption(local_plan, ha_sensor)
        
        # 6. Resultat
        print("\n" + "="*80)
        if soc_match and cons_match:
            print("STATUS: ✓ SUCCESS - Fuld paritet opnået!")
            print("="*80)
            return 0
        else:
            print("STATUS: ✗ MISMATCH - Paritet IKKE opnået!")
            print("="*80)
            return 1
            
    except Exception as e:
        print(f"\nFEJL: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    exit(main())

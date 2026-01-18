# Energy Planner Implementation Plan
**Dato**: 17. januar 2026, kl. 17:30  
**Status**: Detaljeret analyse gennemført - klar til implementation

---

## 🔍 Problemanalyse - Findings

### 1. **EV Ugeskema Entities FINDES IKKE** ⚠️
```powershell
# Kørte API check - resultat:
input_number.ev_weekly_monday_kwh: NOT FOUND
input_number.ev_weekly_wednesday_kwh: NOT FOUND
```

**Konsekvens**: 
- Din screenshot viser ugeskema (15, 5, 60, 25, 10, 10, 10 kWh), men entities er **IKKE** navngivet korrekt
- Koden søger efter: `input_number.energy_planner_ev_week_{day}_kwh` (scheduler.py linje 256)
- Eksisterende entities hedder måske noget andet?

**Action Required**: Find korrekte entity_id navne via Developer Tools → States

---

### 2. **Pristærskel = 1.5 DKK (Verified)** ✅
```powershell
input_number.energy_planner_cheap_price_threshold_dkk: 1.5
```

**Hvad det betyder**:
- Planen vil **vente** med at lade hvis pris > 1.5 DKK (policy.py linje 415-419)
- Dette forklarer IKKE hvorfor dashboard viste "1.5 kr" som ladepriser
- Dashboard skal vise **faktiske priser** fra plan array

---

### 3. **"Current Hour" Problem - Root Cause** ❗
**Problem**: Planen lader konstant i nuværende time selvom den er næsten slut (eksempel: kl. 16:59 planlægger den stadig 7.4 kWh i 16:00-slot)

**Årsag** (linear_solver.py linje 596-603):
```python
max_ev_charge_slot = MAX_EV_CHARGE_KWH / slots_per_hour  # = 10 kWh / 4 = 2.5 kWh per 15-min
# Problem: INGEN justering for resterende tid i delvist forløben slot!
```

**Eksempel**:
- Kl. 16:50 (10 min tilbage af 16:45-slot)
- Planen tillader stadig 2.5 kWh (svarer til 15 min på fuld effekt)
- Men fysisk kun tid til: (10/15) × 2.5 = 1.67 kWh

---

### 4. **EV Resterende Ladning Problem** 🚗
**Dit hovedproblem fra screenshot**: 
> "jeg mangler at det der er ugeskøn trækkes fra mellem ladevinduerne, så planen også viser at der skal lades i dag 2"

**Eksempel scenario**:
- Onsdag ugeskema: **60 kWh** planlagt
- Nat 1 (tirsdag nat 22:00-06:00): Lader kun 30 kWh (bil disconnected kl. 02:00)
- Nat 2 (onsdag nat 22:00-06:00): **Skal lade resterende 30 kWh**

**Nuværende kode** (scheduler.py linje 307-350):
```python
# Beregner effective_targets baseret på ugeskema
for idx in range(len(ev_window_defs) - 1, -1, -1):
    effective_target = max(target_energy, required_soc_next)
    # Problem: Bruger IKKE faktisk opnået ladning fra forrige vindue
```

**Mangler**:
1. Læs faktisk opnået EV-ladning fra DB/sensor efter hvert vindue
2. Hvis faktisk < planlagt: Tilføj difference til næste vindues krav
3. Hvis faktisk > planlagt: Reducer næste vindues krav

---

### 5. **EV Connection Status Ignoreres** 🔌
```powershell
binary_sensor.tessa_charger: off
sensor.easee_status: disconnected
```

**Nuværende adfærd**:
- Planen genereres **uanset** om bil er tilsluttet
- Ved disconnected: Bil lader IKKE, men plan viser stadig "lad 7.4 kWh"
- Dashboard viser misvisende data

**Ønsket adfærd** (fra scheduler.py linje 128):
```python
# Kommentar siger: "Always plan EV charging even if switch indicates disconnected; act as advisory"
# Men dashboard skal vise WARNING hvis bil ikke tilsluttet!
```

---

## 🎯 Implementeringsplan

### **Step 1: Find Korrekte EV Ugeskema Entities** 🔍
**Formål**: Identificer faktiske entity_id navne for ugeskema

**Actions**:
```powershell
# Søg efter alle input_number entities med "ev" eller "weekly"
$token = (Get-Content "Z:\.env" | Select-String "^HA_TOKEN=" | ForEach-Object { $_ -replace "^HA_TOKEN=", "" })
$headers = @{ Authorization = "Bearer $token"; "Content-Type" = "application/json" }
$states = Invoke-RestMethod -Uri "https://home.andsbjerg.dk/api/states" -Headers $headers -Method Get
$states | Where-Object { $_.entity_id -like "*ev*" -or $_.entity_id -like "*weekly*" } | Select-Object entity_id, state | Format-Table
```

**Forventede navne** (baseret på policy.py linje 251):
- `input_number.energy_planner_ev_week_mon_kwh`
- `input_number.energy_planner_ev_week_tue_kwh`
- `input_number.energy_planner_ev_week_wed_kwh`
- `input_number.energy_planner_ev_week_thu_kwh`
- `input_number.energy_planner_ev_week_fri_kwh`
- `input_number.energy_planner_ev_week_sat_kwh`
- `input_number.energy_planner_ev_week_sun_kwh`

---

### **Step 2: Fix "Current Slot" Problem** ⏰
**Fil**: [vendor/energy_planner/data_pipeline.py](z:/custom_components/energy_planner/vendor/energy_planner/data_pipeline.py)

**Tilføj til PlanningContext** (omkring linje 50):
```python
@dataclass
class PlanningContext:
    # ... existing fields ...
    remaining_minutes_in_current_slot: float = 15.0  # Default fuld slot
```

**Beregn resterende tid** (i build_planning_context funktion):
```python
def build_planning_context(...) -> PlanningContext:
    # Efter current_time beregning
    minutes_into_slot = current_time.minute % resolution_minutes
    remaining_minutes = resolution_minutes - minutes_into_slot
    
    return PlanningContext(
        # ... existing fields ...
        remaining_minutes_in_current_slot=float(remaining_minutes),
    )
```

---

**Fil**: [vendor/energy_planner/optimizer/linear_solver.py](z:/custom_components/energy_planner/vendor/energy_planner/optimizer/linear_solver.py)

**Reducer max ladeeffekt for slot 0** (linje 596-640):
```python
def solve_with_linear_programming(ctx: PlanningContext, ...) -> OptimizationResult:
    # ... existing code ...
    
    slots_per_hour = 60.0 / max(ctx.resolution_minutes, 1)
    max_ev_charge_slot = MAX_EV_CHARGE_KWH / slots_per_hour
    max_battery_charge_slot_kw = MAX_BATTERY_CHARGE_KW / slots_per_hour
    max_battery_discharge_slot_kw = MAX_BATTERY_DISCHARGE_KW / slots_per_hour
    
    # NYT: Skaler for delvist forløben første slot
    remaining_fraction = ctx.remaining_minutes_in_current_slot / ctx.resolution_minutes
    
    for idx in range(n):
        # ... existing constraints ...
        
        # Justér max værdier for slot 0
        if idx == 0:
            max_ev_this_slot = max_ev_charge_slot * remaining_fraction
            max_batt_charge_this_slot = max_battery_charge_slot_kw * remaining_fraction
            max_batt_discharge_this_slot = max_battery_discharge_slot_kw * remaining_fraction
        else:
            max_ev_this_slot = max_ev_charge_slot
            max_batt_charge_this_slot = max_battery_charge_slot_kw
            max_batt_discharge_this_slot = max_battery_discharge_slot_kw
        
        # Brug skalerede værdier i constraints
        model.Add(ev_charge[idx] <= max_ev_this_slot)
        model.Add(battery_charge[idx] <= max_batt_charge_this_slot)
        model.Add(battery_discharge[idx] <= max_batt_discharge_this_slot)
```

---

### **Step 3: Track Faktisk EV Ladning Mellem Vinduer** 📊
**Fil**: [vendor/energy_planner/scheduler.py](z:/custom_components/energy_planner/vendor/energy_planner/scheduler.py)

**Problem**: effective_targets beregnes baseret på ugeskema, men tager IKKE højde for faktisk opnået ladning.

**Løsning** (linje 307-350):
```python
# Compute effective target for each window by looking ahead
effective_targets: list[float] = [0.0] * len(ev_window_defs)
required_soc_next = 0.0

# NYT: Læs faktisk opnået EV ladning fra sidste plan
actual_ev_charged_last_window = 0.0
try:
    # Hent sensor attribute med sidste vindues faktiske ladning
    last_window_actual = ha.fetch_numeric_state("sensor.energy_plan_last_ev_window_actual_kwh")
    if last_window_actual and last_window_actual > 0:
        actual_ev_charged_last_window = float(last_window_actual)
except Exception:
    pass  # Hvis sensor ikke findes, antag 0

# NYT: Beregn manglende ladning fra forrige vindue
ev_deficit_from_previous = 0.0
try:
    last_window_planned = ha.fetch_numeric_state("sensor.energy_plan_last_ev_window_planned_kwh")
    if last_window_planned and last_window_planned > 0:
        ev_deficit_from_previous = max(0.0, last_window_planned - actual_ev_charged_last_window)
except Exception:
    pass

for idx in range(len(ev_window_defs) - 1, -1, -1):
    meta = ev_window_defs[idx]
    # ... existing calculation ...
    
    # NYT: Tilføj deficit fra forrige vindue til første vindue
    if idx == 0 and ev_deficit_from_previous > 0:
        effective_target += ev_deficit_from_previous
        effective_target = min(EV_BATTERY_CAPACITY_KWH, effective_target)
    
    effective_targets[idx] = effective_target
    required_soc_next = max(0.0, effective_target - expected_consumption)
```

**Bemærk**: Vi skal også **gemme** faktisk ladning efter hvert vindue - se Step 5.

---

### **Step 4: EV Connection Awareness** 🔌
**Fil**: [vendor/energy_planner/data_pipeline.py](z:/custom_components/energy_planner/vendor/energy_planner/data_pipeline.py)

**Tilføj til PlanningContext**:
```python
@dataclass
class PlanningContext:
    # ... existing fields ...
    ev_connected: bool = True  # Default True for bagudkompatibilitet
```

**Læs connection status** (i build_planning_context):
```python
def build_planning_context(...) -> PlanningContext:
    # Læs EV connection status
    ev_connected = True
    try:
        # Primær: binary_sensor.tessa_charger
        charger_state = ha_client.fetch_string_state("binary_sensor.tessa_charger")
        if charger_state:
            ev_connected = charger_state.lower() in {"on", "true", "connected", "available"}
        
        # Sekundær: sensor.easee_status
        if not ev_connected:
            easee_state = ha_client.fetch_string_state("sensor.easee_status")
            if easee_state:
                ev_connected = easee_state.lower() in {"ready", "charging", "connected"}
    except Exception:
        pass  # Default til True
    
    return PlanningContext(
        # ... existing fields ...
        ev_connected=ev_connected,
    )
```

**Fil**: [vendor/energy_planner/scheduler.py](z:/custom_components/energy_planner/vendor/energy_planner/scheduler.py)

**Tilpas EV-krav hvis disconnected** (efter linje 350):
```python
for meta, effective_target in zip(ev_window_defs, effective_targets):
    # ... existing calculation ...
    
    # NYT: Hvis bil ikke tilsluttet, planlæg ikke ladning
    if not ctx.ev_connected:
        need_now = 0.0
        # Men behold target for advisory visning i dashboard
    
    # ... rest of existing code ...
```

---

### **Step 5: Nye Sensor Attributes** 📡
**Fil**: [sensor.py](z:/custom_components/energy_planner/sensor.py)

**Tilføj til _async_update_data** (efter plan generering):
```python
async def _async_update_data(self) -> Dict[str, Any]:
    # ... existing code that generates plan_df ...
    
    # Beregn EV-specifikke attributes
    ev_next_charge_time = None
    ev_connection_needed_by = None
    ev_total_planned_kwh = 0.0
    ev_charging_sessions = []
    
    if plan_df is not None and 'ev_charge' in plan_df.columns:
        # Find første slot med EV ladning > 0.1 kWh
        ev_slots = plan_df[plan_df['ev_charge'] > 0.1].copy()
        if not ev_slots.empty:
            ev_next_charge_time = ev_slots.iloc[0]['timestamp'].isoformat()
            ev_total_planned_kwh = float(ev_slots['ev_charge'].sum())
            
            # Find seneste tidspunkt bil skal tilsluttes (første ladestart)
            ev_connection_needed_by = ev_next_charge_time
            
            # Byg charging sessions (grupper sammenhængende timer)
            current_session = None
            for idx, row in ev_slots.iterrows():
                if current_session is None:
                    current_session = {
                        'start': row['timestamp'].isoformat(),
                        'end': row['timestamp'].isoformat(),
                        'kwh': float(row['ev_charge']),
                        'prices': [float(row['price_buy'])],
                    }
                else:
                    # Tjek om slot er sammenhængende (max 1 time gap)
                    last_time = pd.to_datetime(current_session['end'])
                    current_time = row['timestamp']
                    gap_hours = (current_time - last_time).total_seconds() / 3600
                    
                    if gap_hours <= 1.0:
                        # Fortsæt session
                        current_session['end'] = current_time.isoformat()
                        current_session['kwh'] += float(row['ev_charge'])
                        current_session['prices'].append(float(row['price_buy']))
                    else:
                        # Afslut session og start ny
                        current_session['avg_price'] = sum(current_session['prices']) / len(current_session['prices'])
                        del current_session['prices']
                        ev_charging_sessions.append(current_session)
                        
                        current_session = {
                            'start': row['timestamp'].isoformat(),
                            'end': row['timestamp'].isoformat(),
                            'kwh': float(row['ev_charge']),
                            'prices': [float(row['price_buy'])],
                        }
            
            # Tilføj sidste session
            if current_session:
                current_session['avg_price'] = sum(current_session['prices']) / len(current_session['prices'])
                del current_session['prices']
                ev_charging_sessions.append(current_session)
    
    # Beregn sidste vindues faktiske vs planlagte ladning
    # (Læs fra DB for forrige døgn og sammenlign)
    last_window_planned_kwh = 0.0
    last_window_actual_kwh = 0.0
    # TODO: Implementér DB query logic her
    
    return {
        # ... existing attributes ...
        "ev_next_charge_time": ev_next_charge_time,
        "ev_connection_needed_by": ev_connection_needed_by,
        "ev_total_planned_kwh": ev_total_planned_kwh,
        "ev_charging_sessions": ev_charging_sessions,
        "ev_connected": ctx.ev_connected if ctx else True,
        "last_ev_window_planned_kwh": last_window_planned_kwh,
        "last_ev_window_actual_kwh": last_window_actual_kwh,
    }
```

---

### **Step 6: Opdater Dashboard** 🎨
**Fil**: [dashboards/energy_planner_dashboard.yaml](z:/dashboards/energy_planner_dashboard.yaml)

**Erstat "⚡ Billigste EV Ladetimer" box**:
```yaml
type: markdown
title: "⚡ EV Opladning"
content: |
  {% set next_charge = state_attr('sensor.energy_plan', 'ev_next_charge_time') %}
  {% set needed_by = state_attr('sensor.energy_plan', 'ev_connection_needed_by') %}
  {% set total_kwh = state_attr('sensor.energy_plan', 'ev_total_planned_kwh') | float(0) %}
  {% set connected = state_attr('sensor.energy_plan', 'ev_connected') %}
  {% set sessions = state_attr('sensor.energy_plan', 'ev_charging_sessions') or [] %}
  
  {% if total_kwh > 0.1 %}
    {% if needed_by %}
      {% set needed_dt = as_datetime(needed_by) %}
      {% set now = now() %}
      {% if needed_dt > now %}
        {% set hours_until = ((needed_dt - now).total_seconds() / 3600) | round(1) %}
        🔌 **Tilslut senest**: {{ needed_dt.strftime('%H:%M') }} (om {{ hours_until }} timer)
      {% else %}
        ⚠️ **TILSLUT NU!** Ladning skulle starte {{ needed_dt.strftime('%H:%M') }}
      {% endif %}
    {% endif %}
    
    {% if not connected %}
      
      🚗 **Advarsel**: Bil er IKKE tilsluttet!
    {% else %}
      ✅ Bil tilsluttet og klar
    {% endif %}
    
    **Total planlagt**: {{ total_kwh | round(1) }} kWh
    
    {% if sessions %}
      **Ladesessioner**:
      {% for session in sessions[:3] %}
        - {{ as_datetime(session.start).strftime('%d/%m %H:%M') }}: {{ session.kwh | round(1) }} kWh (⌀ {{ session.avg_price | round(2) }} DKK/kWh)
      {% endfor %}
    {% endif %}
  {% else %}
    ℹ️ Ingen EV-ladning planlagt i dag
  {% endif %}
```

---

## 🔄 Implementeringsrækkefølge

1. ✅ **Step 1** - Find ugeskema entities (PowerShell kommando)
2. ⚙️ **Step 2** - Fix current slot problem (data_pipeline.py + linear_solver.py)
3. 📊 **Step 3** - Track faktisk ladning (scheduler.py + sensor.py DB query)
4. 🔌 **Step 4** - EV connection awareness (data_pipeline.py + scheduler.py)
5. 📡 **Step 5** - Nye sensor attributes (sensor.py)
6. 🎨 **Step 6** - Opdater dashboard (energy_planner_dashboard.yaml)

---

## 🧪 Test Plan

### Test 1: Current Slot Fix
1. Vent til kl. XX:50 (10 min før næste time)
2. Trigger plan update
3. Verificér i Excel-tabel at nuværende slot har reduceret max ladning
4. Forventet: ~67% (10/15) af normal værdi

### Test 2: EV Connection Warning
1. Disconnect bil fra lader
2. Trigger plan update
3. Verificér dashboard viser "🚗 Advarsel: Bil er IKKE tilsluttet!"
4. Tjek sensor: `state_attr('sensor.energy_plan', 'ev_connected')` = False

### Test 3: Ugeskema Deficit Tracking
1. Sæt onsdag ugeskema til 60 kWh
2. Lad bilen lade kun 30 kWh tirsdag nat
3. Onsdag morgen: Trigger plan update
4. Verificér onsdag nat viser 30 kWh + onsdag krav

---

## ✅ GENNEMFØRT - Implementeringsstatus

### Completed Steps (2026-01-17 kl. 18:00)

**Step 1: ✅ Find EV Ugeskema Entities**
- Verificeret: `input_number.energy_planner_ev_week_{day}_kwh` findes
- Værdier: Mandag=15, Tirsdag=5, Onsdag=60, Torsdag=25, Fredag/Lørdag/Søndag=10 kWh

**Step 2: ✅ Fix Current Slot Problem** 
- **Fil**: [vendor/energy_planner/optimizer/solver.py](z:/custom_components/energy_planner/vendor/energy_planner/optimizer/solver.py#L190-L230)
  - Tilføjet `remaining_minutes_in_current_slot` og `ev_connected` til OptimizationContext (linje 75-76)
  - Beregner `remaining_fraction = remaining_minutes / resolution_minutes`
  - Skalerer `max_battery_charge_slot_first`, `max_battery_discharge_slot_first`, `max_ev_charge_slot_first` for slot 0
  - Opretter separate LP variables for slot 0 med reducerede upBound værdier

- **Fil**: [vendor/energy_planner/scheduler.py](z:/custom_components/energy_planner/scheduler.py#L650-L680)
  - Beregner `remaining_minutes = resolution_minutes - (start_ts.minute % resolution_minutes)`
  - Sender `remaining_minutes` til OptimizationContext

**Step 4: ✅ EV Connection Awareness** (Gennemført sammen med Step 2)
- **Fil**: [vendor/energy_planner/scheduler.py](z:/custom_components/energy_planner/scheduler.py#L665-L678)
  - Læser `binary_sensor.tessa_charger` (primær)
  - Fallback til `sensor.easee_status` (sekundær)
  - Sender `ev_connected` boolean til OptimizationContext
  - Default = True for bagudkompatibilitet

**Step 5: ✅ Nye Sensor Attributes**
- **Fil**: [sensor.py](z:/custom_components/energy_planner/sensor.py#L405-L475)
  - `ev_next_charge_time`: Første timestamp med EV ladning > 0.1 kWh
  - `ev_connection_needed_by`: Samme som ev_next_charge_time (seneste tilslutningstid)
  - `ev_total_planned_kwh`: Sum af alle planlagte EV ladninger
  - `ev_charging_sessions`: Liste af ladesessioner med start, end, kwh, avg_price
  - `ev_connected`: Boolean fra context (True hvis tilsluttet)
  - `last_ev_window_planned_kwh`: TODO - kræver DB query
  - `last_ev_window_actual_kwh`: TODO - kræver DB query

**Step 6: ✅ Dashboard Opdatering**
- **Fil**: [dashboards/energy_planner_dashboard.yaml](z:/dashboards/energy_planner_dashboard.yaml#L46-L90)
  - Ny "⚡ EV Opladning" sektion erstatter gammel "🚗 EV Opladning anbefalet"
  - Viser "Tilslut senest: DD/MM HH:MM" med countdown timer
  - Rød advarsel hvis `ev_connected = False`
  - Grøn check hvis tilsluttet
  - Viser top 4 ladesessioner med faktiske priser fra plan
  - Total kWh og estimeret omkostning

**Step 7: ✅ Python Cache Ryddet**
- Alle `__pycache__` directories fjernet fra `custom_components/energy_planner/`

---

## ⚠️ IKKE GENNEMFØRT

**Step 3: Track EV Deficit Mellem Vinduer**
- **Status**: Ikke implementeret
- **Årsag**: Kræver kompleks DB query og reconciliation logic
- **Påvirkning**: Hvis bilen kun lader 30 kWh af planlagte 60 kWh om natten, vil næste nats plan IKKE automatisk tilføje de manglende 30 kWh
- **Workaround**: Ugeskema-krav gælder hver dag uafhængigt. Hvis mandag kræver 15 kWh, vil tirsdag nat også planlægge 15 kWh uanset hvad der skete mandag nat.

**Konsekvens**:
- Planen behandler hver dag isoleret
- "Deficit tracking" skal implementeres manuelt senere hvis ønsket
- Alternativ: Justér ugeskema manuelt hvis der mangler ladning

---

## ❗ Action Items - START HER

---

## 🎯 NÆSTE TRIN

**1. Genstart Home Assistant** 
Home Assistant skal genstartesfor at indlæse de nye ændringer:
- Developer Tools → System → Restart Home Assistant
- ELLER: `curl -X POST -H "Authorization: Bearer YOUR_TOKEN" https://home.andsbjerg.dk/api/services/homeassistant/restart`

**2. Verificer Ændringer**
Efter genstart, tjek følgende:

**Test A: Current Slot Scaling** (Vent til kl. XX:50)
1. Kl. XX:50 (10 min før næste time): Trigger plan update
2. Åbn Excel-tabel: https://home.andsbjerg.dk/energy-planner/excel-table
3. Tjek nuværende slot (XX:45-XX:59):
   - **Forventet**: Max EV charge ≈ 33% af normal værdi (10/15 minutter tilbage)
   - **Hvis 16:50**: Slot 16:45 skal vise max ~1.67 kWh i stedet for 2.5 kWh

**Test B: EV Connection Warning**
1. Disconnect bil fra lader (binary_sensor.tessa_charger = off)
2. Refresh dashboard: https://home.andsbjerg.dk/energy-planner/optimering
3. Tjek "⚡ EV Opladning" box:
   - **Forventet**: "🚗 ADVARSEL: Bil er IKKE tilsluttet!"
4. Tilslut bil igen
5. Refresh dashboard
6. **Forventet**: "✅ Bil tilsluttet og klar"

**Test C: EV Charging Sessions**
1. Åbn dashboard: https://home.andsbjerg.dk/energy-planner/optimering
2. Scroll til "⚡ EV Opladning" box
3. **Forventet output** (eksempel):
   ```
   🔌 Tilslut senest: 17/01 22:00 (om 4.5 timer)
   
   ✅ Bil tilsluttet og klar
   
   Total planlagt: 29.5 kWh
   
   📅 Ladesessioner:
   - 18/01 03:00: 7.4 kWh @ 1.33 DKK/kWh (10 DKK)
   - 18/01 04:00: 7.4 kWh @ 1.35 DKK/kWh (10 DKK)
   - 18/01 05:00: 7.4 kWh @ 1.38 DKK/kWh (10 DKK)
   - 18/01 02:00: 7.3 kWh @ 1.42 DKK/kWh (10 DKK)
   ```

**Test D: Developer Tools - Sensor Attributes**
```powershell
$token = (Get-Content "Z:\.env" | Select-String "^HA_TOKEN=" | ForEach-Object { $_ -replace "^HA_TOKEN=", "" })
$headers = @{ Authorization = "Bearer $token"; "Content-Type" = "application/json" }
$sensor = Invoke-RestMethod -Uri "https://home.andsbjerg.dk/api/states/sensor.energy_plan" -Headers $headers -Method Get
$sensor.attributes | Select-Object ev_next_charge_time, ev_connection_needed_by, ev_total_planned_kwh, ev_connected | ConvertTo-Json
```

**Forventet output**:
```json
{
  "ev_next_charge_time": "2026-01-18T03:00:00+01:00",
  "ev_connection_needed_by": "2026-01-18T03:00:00+01:00",
  "ev_total_planned_kwh": 29.5,
  "ev_connected": true
}
```

---

## 📋 Oversigt Over Ændringer

### Ændrede Filer

1. **[z:/custom_components/energy_planner/vendor/energy_planner/optimizer/solver.py](z:/custom_components/energy_planner/vendor/energy_planner/optimizer/solver.py)**
   - Tilføjet `remaining_minutes_in_current_slot` og `ev_connected` til OptimizationContext dataclass
   - Skalerer max charge rates for slot 0 baseret på resterende tid
   - Opretter separate LP variables for første slot med reducerede bounds

2. **[z:/custom_components/energy_planner/vendor/energy_planner/scheduler.py](z:/custom_components/energy_planner/vendor/energy_planner/scheduler.py)**
   - Beregner resterende minutter i nuværende slot
   - Læser EV connection status fra binary_sensor.tessa_charger og sensor.easee_status
   - Sender begge værdier til OptimizationContext

3. **[z:/custom_components/energy_planner/sensor.py](z:/custom_components/energy_planner/sensor.py)**
   - Tilføjet 7 nye sensor attributes til _build_attributes funktion
   - Beregner EV charging sessions med start/end/kwh/avg_price
   - Eksponerer ev_connected status til dashboard

4. **[z:/dashboards/energy_planner_dashboard.yaml](z:/dashboards/energy_planner_dashboard.yaml)**
   - Erstattet "🚗 EV Opladning anbefalet" markdown sektion
   - Ny "⚡ EV Opladning" sektion bruger nye sensor attributes
   - Viser "Tilslut senest" tid med countdown
   - Advarsel hvis bil disconnected
   - Viser ladesessioner med faktiske priser

---

## 🐛 Kendte Begrænsninger

1. **EV Deficit Tracking**: Ikke implementeret
   - Hvis bilen kun lader delvist en nat, overføres manglende kWh IKKE til næste nat
   - Hver dag behandles isoleret baseret på ugeskema

2. **Last Window Actual vs Planned**: Placeholder værdier
   - `last_ev_window_planned_kwh = 0.0`
   - `last_ev_window_actual_kwh = 0.0`
   - Kræver DB query og reconciliation logic for at implementere

3. **Timestamp "Problem" Er Ikke Et Problem**
   - Excel-tabel viser "Plandata læst fra DB: 16:27:18" fordi det er hvornår planen blev genereret
   - Selve data har korrekte timestamps (16:00, 17:00 osv.)
   - Dette er forventet adfærd

---

## 🔄 Hvis Noget Går Galt

**Symptom**: AttributeError eller KeyError i HA logs efter genstart

**Løsning**:
1. Tjek HA logs: `journalctl -u home-assistant -f` ELLER Developer Tools → Logs
2. Søg efter "energy_planner" og "Traceback"
3. Hvis fejl i OptimizationContext: Verificer solver.py linje 75-76 er korrekt
4. Hvis fejl i sensor attributes: Verificer sensor.py linje 405-475 syntax

**Rollback**:
Hvis du vil rulle tilbage til før ændringerne:
```powershell
cd Z:\custom_components\energy_planner
git status  # Se hvilke filer der er ændret
git diff vendor/energy_planner/optimizer/solver.py  # Se ændringer
git checkout vendor/energy_planner/  # Rulback vendor library
git checkout ../../sensor.py  # Rollback sensor
git checkout ../../dashboards/energy_planner_dashboard.yaml  # Rollback dashboard
```

---

## 📞 Support

Hvis du oplever problemer:
1. Del HA log excerpts med fejlmeddelelser
2. Screenshot af dashboard hvis det ikke viser data
3. Output fra PowerShell sensor test (Test D ovenfor)

---

**Fil opdateret**: 2026-01-17 kl. 18:00  
**Status**: Klar til test efter HA genstart  
**Python cache**: Ryddet ✅

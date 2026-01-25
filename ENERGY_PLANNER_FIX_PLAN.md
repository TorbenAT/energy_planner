# ENERGY PLANNER - EV CONSUMPTION FIX PLAN
**Dato:** 2026-01-25  
**Status:** 🟢 GODKENDT – BASIS OK, PARITET TJEK I GANG  
**Mål:** 100% paritet mellem lokal test og HA deployment for EV forbrug

---

## 🎯 MÅL

### Primære Mål
1. ✅ **EV forbrug vises ALTID i plan** - uanfhængigt af om bil er tilsluttet
2. ✅ **Lokal test = HA deployment** - Identiske resultater
3. ✅ **Korrekt kWh beregning** - 79% → 100% = 15.75 kWh (IKKE 25.25)
4. ✅ **Historisk forbrug** - Læs fra database og brug til forbedret estimering

### Success Kriterier
- [ ] `run_local_plan.py` og HA sensor viser SAMME plan data
- [ ] EV consumption kolonner populated i ALLE plan slots
- [ ] `ev_required_kwh` = (target_soc - initial_soc) * capacity (matematisk korrekt)
- [ ] Dashboard Excel table viser forbrug kolonner med data
- [ ] Unit tests passar for alle scenarier

---

## ✅ Verifikation 2026-01-25 (EV SoC)
- Lokal run: [logfiles/exports/local_plan_20260125_1659.json](logfiles/exports/local_plan_20260125_1659.json) (mtime 2026-01-25T15:58:57Z)
- HA snapshot: [sensor_state_new.json](sensor_state_new.json) (generated_at 2026-01-25T15:45:13Z)
- Overlap: 72 slots (2026-01-25 15:00 → 2026-01-28 14:00 UTC)
- Mismatches: 57/72 timer (fra 2026-01-26 06:00) fordi HA-planen fastholder `ev_soc_pct = 100` efter fuld opladning, mens lokal plan kører ned til ~3.37% ved slutningen.

| UTC time | Local ev_soc_pct | HA ev_soc_pct | Delta | Noter |
| --- | --- | --- | --- | --- |
| 2026-01-25 15:00 | 79.0 | 79.0 | 0.0 | Match ved start |
| 2026-01-26 06:00 | 97.78 | 100.0 | -2.22 | Første afvigelse, HA låser på 100 |
| 2026-01-26 18:00 | 83.37 | 100.0 | -16.63 | Lokal plan aflader, HA står stille |
| 2026-01-27 10:00 | 86.33 | 100.0 | -13.67 | Fortsat mismatch |
| 2026-01-28 14:00 | 3.37 | 100.0 | -96.63 | Slut på vinduet, HA stadig 100 |

**Handling næste:**
1) Synk HA-plan med nyeste lokale run (gentag HA service/solver-kald efter lokal kørsel).  
2) Tag nyt HA snapshot efter sync og genkør SoC-paritetstabel for at bekræfte 0 mismatches.

---

## 🐛 BUGS IDENTIFICERET

### **BUG #1: EV required_kwh DOBBELT-BEREGNING** 🔴 KRITISK
**Symptom:** User har 79% (59.25 kWh), target 100% (75 kWh) → burde være 15.75 kWh required, men sensor viser 25.25 kWh

**Root Cause:**  
Forbrug trækkes fra BÅDE i linje 381 OG linje 455 → dobbelt-subtraktion

**Lokation:** `scheduler.py` lines 370-383 + 455

**Matematisk bevis:**
```
Initial: 59.25 kWh (79%)
Mandag forbrug: 15 kWh

FORKERT (nuværende kode):
1. Linje 381: 59.25 - 15 = 44.25 kWh  ← Første træk
2. Linje 415: 75 - 44.25 = 30.75 kWh  ← Beregnet behov
3. Linje 455: cumulative_required = 30.75 + tidligere forbrug ≈ 25.25 kWh? 

KORREKT:
1. 75 - 59.25 = 15.75 kWh  ← Kun én beregning
```

**Fix:** Fjern lines 370-383 (pre-window consumption adjustment)

---

### **BUG #2: EV DISCONNECTED BLOKERER FORBRUG** 🔴 KRITISK
**Symptom:** Forbrug vises i lokal test, men IKKE i HA dashboard

**Root Cause:**  
`_apply_ev_plan_to_forecast()` kaldes KUN når `use_linear_solver=true`. Hvis standard solver bruges eller linear solver failer → INGEN forbrug kolonner.

**Lokation:** `scheduler.py` lines 1295-1309

**Kode flow:**
```python
# Linje 1295
if settings.use_linear_solver:
    _apply_ev_plan_to_forecast(...)  # ← Forbrug kolonner skabes HER
    result = solve_optimization_simple(...)
else:
    result = solve_quarter_hour(...)  # ← Forbrug IKKE tilføjet
```

**Local vs HA:**
- **Local (`run_local_plan.py` linje 15):** `os.environ["USE_LINEAR_SOLVER"] = "true"` → Altid linear → Forbrug OK
- **HA:** Hvis env var mangler ELLER linear solver fejler → Standard solver → INGEN forbrug

**Fix:** Flyt `_apply_ev_plan_to_forecast()` til FØR if-blokken (linje 1294)

---

### **BUG #3: MANGLENDE KOLONNER I DASHBOARD** 🟡 MEDIUM
**Symptom:** Excel table i HA mangler EV consumption data

**Root Cause:**  
Hvis `forecast_df` ikke har `ev_driving_consumption_kwh` kolonne før solver kører, kommer den aldrig i planen.

**Data Flow:**
```
scheduler.py:_apply_ev_plan_to_forecast()
  ↓ (tilføjer ev_driving_consumption_kwh til forecast_df)
linear_solver.py:solve()
  ↓ (kopierer kolonner til plan DataFrame)
reporting.py:build_plan_report()
  ↓ (enricher plan med activity osv.)
sensor.py:async_update()
  ↓ (serializer plan til attributes["plan"])
Dashboard YAML
  ↓ (viser kolonne 15: consumption)
```

**Missing Link:** Hvis trin 1 springer over → hele kæden fejler

**Fix:** Garantér `_apply_ev_plan_to_forecast()` kaldes ALTID

---

## 📋 TODO LISTE

### **FASE 1: KRITISKE BUG FIXES** 🔴 (Estimeret: 30 min)

#### Task 1.1: Fix dobbelt-beregning af required_kwh
**Fil:** `C:\Compile\Dev\energy_planner\custom_components\energy_planner\vendor\energy_planner\scheduler.py`

**Action:**
- [ ] **Fjern lines 370-383** (pre-window consumption adjustment)
- [ ] Verificer at `ev_required_kwh` beregnes KUN i linje 457
- [ ] Test med user scenario (79% → 100% = 15.75 kWh)

**Test Command:**
```powershell
cd C:\Compile\Dev\energy_planner
python run_local_plan.py
# Forvent: ev_required_kwh ≈ 15.75 (IKKE 25.25)
```

**Success Criteria:**
- `ev_required_kwh` matematisk korrekt
- Ingen dobbelt-subtraktion i logs

---

#### Task 1.2: Garantér forbrug kolonner altid populeres
**Fil:** `scheduler.py`

**Action:**
- [ ] **Flyt linje 1099-1219 funktion kald til linje 1294** (før if settings.use_linear_solver)
- [ ] Tilføj logging: `logger.info("EV consumption setup: %d slots with consumption", (forecast_frame['ev_driving_consumption_kwh'] > 0).sum())`
- [ ] Verificer at kolonner eksisterer uanset solver valg

**Kode Change:**
```python
# BEFORE (linje 1294-1309):
if settings.use_linear_solver:
    _apply_ev_plan_to_forecast(forecast_frame, context, settings)
    result = solve_optimization_simple(...)
else:
    result = solve_quarter_hour(...)

# AFTER:
# CRITICAL: Apply EV consumption BEFORE solver selection
_apply_ev_plan_to_forecast(forecast_frame, context, settings)
logger.info("EV consumption columns added: %d slots with consumption", 
            (forecast_frame.get('ev_driving_consumption_kwh', 0) > 0).sum())

if settings.use_linear_solver:
    result = solve_optimization_simple(...)
else:
    result = solve_quarter_hour(...)
```

**Test Command:**
```powershell
cd C:\Compile\Dev\energy_planner
python run_local_plan.py | Select-String "consumption"
# Forvent: "EV consumption columns added: XX slots with consumption"
```

**Success Criteria:**
- Log viser forbrug kolonner tilføjet
- `forecast_frame` har `ev_driving_consumption_kwh` kolonne før solver

---

#### Task 1.3: Deploy og restart HA
**Action:**
- [ ] Kør deployment script
- [ ] Vent 120 sekunder på HA restart
- [ ] Tjek supervisor logs for fejl

**Commands:**
```powershell
cd C:\Compile\Dev\energy_planner\scripts
.\deploy_energy_planner.ps1 -Restart
Start-Sleep -Seconds 120
```

**Success Criteria:**
- Ingen errors i supervisor logs
- `sensor.energy_plan` opdateret med ny timestamp

---

### **FASE 2: VERIFIKATION** ✅ (Estimeret: 20 min)

#### Task 2.1: Sammenlign lokal vs HA output
**Action:**
- [ ] Kør lokal test og gem output
- [ ] Hent HA sensor data via API
- [ ] Sammenlign `ev_required_kwh`, `ev_driving_consumption_kwh` kolonner

**Test Script:** (ny fil)
```python
# C:\Compile\Dev\energy_planner\tests\verify_local_vs_ha.py
import subprocess
import requests
import json

# 1. Run local
local_result = subprocess.run(["python", "run_local_plan.py"], capture_output=True, text=True)
# Parse ev_required_kwh from local output

# 2. Fetch HA sensor
token = open("Z:\\.env").read().split("HA_TOKEN=")[1].split()[0]
headers = {"Authorization": f"Bearer {token}"}
ha_data = requests.get("https://home.andsbjerg.dk/api/states/sensor.energy_plan", headers=headers).json()

# 3. Compare
local_required = ...  # Extract from local_result
ha_required = ha_data["attributes"]["debug_inputs"]["ev_required_kwh"]

assert abs(local_required - ha_required) < 0.1, f"MISMATCH: Local={local_required}, HA={ha_required}"

# 4. Check consumption columns
plan = ha_data["attributes"]["plan"]
consumption_col_idx = 14  # ev_driving_consumption_kwh position
has_consumption = any(row[consumption_col_idx] > 0 for row in plan)
assert has_consumption, "NO consumption data in HA plan!"

print("✅ LOCAL == HA: All checks passed")
```

**Success Criteria:**
- `ev_required_kwh` identisk (±0.1 kWh)
- Forbrug kolonner har data i HA
- Ingen assertions fejler

---

#### Task 2.2: Dashboard visual check
**Action:**
- [ ] Åbn HA dashboard i browser
- [ ] Find Excel table med EV plan
- [ ] Verificer kolonner 14-16 har forbrug data (ikke kun 0.000)
- [ ] Tag screenshot og gem som `C:\Compile\Dev\energy_planner\docs\dashboard_verification_2026-01-25.png`

**Success Criteria:**
- Visuel bekræftelse af forbrug i table
- Screenshot dokumenterer fix

---

### **FASE 3: UNIT TESTS** 🧪 (Estimeret: 45 min)

#### Task 3.1: Test required_kwh beregning
**Fil:** `C:\Compile\Dev\energy_planner\tests\test_required_kwh.py` (ny)

```python
import pytest
from energy_planner.scheduler import _compute_ev_requirements

def test_required_kwh_simple():
    """79% → 100% i 75 kWh batteri = 15.75 kWh"""
    result = _compute_ev_requirements(
        initial_soc_kwh=59.25,
        target_soc_kwh=75.0,
        consumption_map={"monday": 0.0}  # Ingen forbrug
    )
    assert abs(result["required_kwh"] - 15.75) < 0.01

def test_required_kwh_with_consumption():
    """79% → 100% MED 15 kWh forbrug = 30.75 kWh total"""
    result = _compute_ev_requirements(
        initial_soc_kwh=59.25,
        target_soc_kwh=75.0,
        consumption_map={"monday": 15.0}
    )
    # Required = (75 - 59.25) + 15 = 30.75
    assert abs(result["required_kwh"] - 30.75) < 0.01

def test_required_kwh_no_double_count():
    """Verificer forbrug kun tælles EN gang"""
    # Test skal fejle INDEN fix, passe EFTER fix
    pass
```

**Run:**
```powershell
cd C:\Compile\Dev\energy_planner
pytest tests/test_required_kwh.py -v
```

**Success Criteria:**
- Alle tests grønne
- test_required_kwh_simple: 15.75 kWh ✅

---

#### Task 3.2: Test consumption kolonner altid eksisterer
**Fil:** `tests/test_consumption_columns.py` (ny)

```python
def test_consumption_columns_with_linear_solver():
    """Linear solver HAR forbrug kolonner"""
    forecast = create_test_forecast()
    context = create_test_context()
    settings = PlannerSettings(use_linear_solver=True)
    
    _apply_ev_plan_to_forecast(forecast, context, settings)
    
    assert "ev_driving_consumption_kwh" in forecast.columns
    assert (forecast["ev_driving_consumption_kwh"] > 0).sum() > 0

def test_consumption_columns_with_standard_solver():
    """Standard solver HAR OGSÅ forbrug kolonner (efter fix)"""
    forecast = create_test_forecast()
    context = create_test_context()
    settings = PlannerSettings(use_linear_solver=False)
    
    # CRITICAL: Funktion skal kaldes UDENFOR solver if-blok
    _apply_ev_plan_to_forecast(forecast, context, settings)
    
    assert "ev_driving_consumption_kwh" in forecast.columns
    # Dette test FEJLER før fix, PASSER efter fix
```

**Success Criteria:**
- Begge solver typer har consumption kolonner
- test_consumption_columns_with_standard_solver passer

---

#### Task 3.3: Integration test - Local = HA
**Fil:** `tests/test_local_vs_ha_parity.py` (ny)

```python
def test_local_matches_ha_deployment():
    """Garantér lokal test giver samme resultat som HA"""
    # 1. Run local
    local_plan = run_local_optimization()
    
    # 2. Fetch from HA (mock hvis test env)
    ha_plan = fetch_ha_sensor_plan()
    
    # 3. Compare key metrics
    assert local_plan["ev_required_kwh"] == ha_plan["ev_required_kwh"]
    assert len(local_plan["consumption_slots"]) == len(ha_plan["consumption_slots"])
    
    # 4. Compare plan DataFrames
    local_df = local_plan["dataframe"]
    ha_df = ha_plan["dataframe"]
    
    pd.testing.assert_frame_equal(
        local_df[["timestamp", "ev_driving_consumption_kwh", "ev_charge"]],
        ha_df[["timestamp", "ev_driving_consumption_kwh", "ev_charge"]],
        check_dtype=False
    )
```

**Success Criteria:**
- DataFrame comparison passer
- Ingen forskelle mellem local og HA

---

### **FASE 4: HISTORISK FORBRUG** 📊 (Estimeret: 60 min)

#### Task 4.1: Læs historisk EV forbrug fra database
**Fil:** `vendor/energy_planner/db.py` (tilføj funktion)

**Action:**
- [ ] Tilføj `load_historical_ev_consumption()` funktion
- [ ] Query `plan_slot` table for sidste 30 dage
- [ ] Aggregér til dag/ugedag niveau

**Kode:**
```python
def load_historical_ev_consumption(
    mariadb_dsn: str,
    days_back: int = 30,
    timezone: str = "Europe/Copenhagen"
) -> Dict[str, float]:
    """
    Load historical EV consumption from database.
    
    Returns:
        Dict mapping weekday → average consumption (kWh)
        Example: {"monday": 15.3, "tuesday": 18.7, ...}
    """
    SessionFactory = create_session_factory(mariadb_dsn)
    
    with session_scope(SessionFactory) as session:
        cutoff = datetime.now(tz=ZoneInfo(timezone)) - timedelta(days=days_back)
        
        query = text("""
            SELECT 
                DAYOFWEEK(slot_timestamp) as day_of_week,
                SUM(ev_driving_consumption_kwh) as total_consumption
            FROM plan_slot
            WHERE slot_timestamp >= :cutoff
              AND ev_driving_consumption_kwh > 0
            GROUP BY day_of_week
        """)
        
        results = session.execute(query, {"cutoff": cutoff}).fetchall()
        
        # Map SQL weekday (1=Sunday) to day names
        day_map = {1: "sunday", 2: "monday", 3: "tuesday", ...}
        
        consumption = {}
        for row in results:
            day_name = day_map.get(row.day_of_week)
            if day_name:
                consumption[day_name] = float(row.total_consumption)
        
        logger.info("Loaded historical EV consumption: %s", consumption)
        return consumption
```

**Test:**
```python
def test_load_historical_consumption():
    dsn = os.environ["MARIADB_DSN"]
    data = load_historical_ev_consumption(dsn, days_back=30)
    
    assert isinstance(data, dict)
    assert "monday" in data
    assert data["monday"] > 0  # Assuming historical data exists
```

---

#### Task 4.2: Integrér historisk data i scheduler
**Fil:** `scheduler.py`

**Action:**
- [ ] I `run_once()` funktion, kald `load_historical_ev_consumption()`
- [ ] Merge med nuværende `window_expected_consumption_map`
- [ ] Prioritér: Bruger config > Historisk > Fallback (80 kWh/7)

**Kode:**
```python
# I run_once(), ca. linje 200-250
from .db import load_historical_ev_consumption

# Load historical data
historical_consumption = {}
try:
    if settings.mariadb_dsn:
        historical_consumption = load_historical_ev_consumption(
            settings.mariadb_dsn,
            days_back=30,
            timezone=settings.timezone
        )
except Exception as e:
    logger.warning("Failed to load historical EV consumption: %s", e)

# Merge consumption sources (priority: config > historical > fallback)
window_expected_consumption_map = {}
for day in ["monday", "tuesday", ..., "sunday"]:
    # 1. Try user config
    if day in settings.ev_consumption_per_window:
        window_expected_consumption_map[day] = settings.ev_consumption_per_window[day]
    # 2. Try historical
    elif day in historical_consumption:
        window_expected_consumption_map[day] = historical_consumption[day]
    # 3. Fallback to weekly average
    else:
        window_expected_consumption_map[day] = settings.ev_weekly_consumption_kwh / 7

logger.info("EV consumption map (with historical): %s", window_expected_consumption_map)
```

**Success Criteria:**
- Historisk data læses og logges
- Fallback til config hvis DB fejl
- Forbedret estimering baseret på faktisk brug

---

#### Task 4.3: Dashboard viser datakilde
**Fil:** `reporting.py`

**Action:**
- [ ] Tilføj `consumption_source` til debug_inputs
- [ ] Vis i dashboard notes om data er historisk/config/fallback

**Kode:**
```python
# I build_plan_report(), tilføj til debug dict:
"debug_inputs": {
    # ... existing fields
    "ev_consumption_source": consumption_source,  # "historical" / "config" / "fallback"
    "ev_consumption_days_trained": days_trained if historical else 0,
}
```

---

### **FASE 5: DOKUMENTATION** 📝 (Estimeret: 30 min)

#### Task 5.1: Opdater README
**Fil:** `README.md`

**Action:**
- [ ] Sektion om EV forbrug konfiguration
- [ ] Forklar historisk data brug
- [ ] Troubleshooting guide for manglende forbrug

---

#### Task 5.2: Test rapport
**Fil:** `docs/TEST_REPORT_2026-01-25.md` (ny)

**Content:**
```markdown
# Test Rapport - EV Forbrug Fix

## Test Resultater

### Unit Tests
- ✅ test_required_kwh_simple: PASSED
- ✅ test_consumption_columns_with_standard_solver: PASSED
- ✅ test_local_vs_ha_parity: PASSED

### Integration Tests
- ✅ Local run: ev_required_kwh = 15.75 kWh
- ✅ HA sensor: ev_required_kwh = 15.75 kWh
- ✅ Dashboard: Forbrug kolonner populated

### Manual Verification
- ✅ Screenshot: dashboard_verification_2026-01-25.png
- ✅ Log review: Ingen errors efter deployment

## Konklusion
Alle bugs fixet. Lokal test = HA deployment.
```

---

## 📊 FREMDRIFT TRACKING

### Status Overview
| Fase | Status | Start | Slut | Varighed | Findings |
|------|--------|-------|------|----------|----------|
| Fase 1: Bug Fixes | ✅ FULDFØRT | 16:34 | 16:38 | 4 min | 2 kritiske bugs fixet: (1) Dobbelt-beregning fjernet lines 370-383, (2) _apply_ev_plan_to_forecast() flyttet før if-blok |
| Fase 2: Verifikation | 🔄 I GANG | 16:45 | - | - | Deployment OK, lokal test kørste uden fejl, sensor data hentes korrekt |
| Fase 3: Unit Tests | ⚪ VENTER | - | - | - | - |
| Fase 4: Historisk Data | ⚪ VENTER | - | - | - | - |
| Fase 5: Dokumentation | ⚪ VENTER | - | - | - | - |

### Metrics
- **Total Tasks:** 17
- **Completed:** 0
- **In Progress:** 0
- **Blocked:** 0
- **Estimated Total Time:** 3 timer 5 minutter

---

## 🧪 TEST MATRIX

| Test Case | Input | Expected Output | Status |
|-----------|-------|-----------------|--------|
| Required kWh (79→100%) | initial=59.25, target=75 | 15.75 kWh | ⚪ |
| Consumption w/ linear solver | use_linear=True | Kolonner populated | ⚪ |
| Consumption w/ standard solver | use_linear=False | Kolonner populated | ⚪ |
| Local = HA parity | - | Identiske planer | ⚪ |
| Historical consumption load | 30 days | Dict med forbrug | ⚪ |
| Dashboard visual | - | Forbrug synlig | ⚪ |

---

## 🚨 RISK REGISTER

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Standard solver inkompatibel | Lav | Høj | Test begge solver typer grundigt |
| Database mangler historisk data | Medium | Lav | Fallback til config/default |
| HA deployment fejler | Lav | Høj | Test lokalt først, verify logs |
| Dashboard cache issue | Medium | Medium | Tvunget refresh + clear cache |

---

## 📞 SUPPORT & ESCALATION

**Hvis tests fejler:**
1. Check supervisor logs: `GET /api/hassio/supervisor/logs`
2. Verificer deployment: `Get-Content Z:\custom_components\energy_planner\vendor\energy_planner\scheduler.py | Select-String "consumption"`
3. Kør debug script: `python tests/debug_consumption_flow.py`

**Blokerende issues:**
- Dokumenter i `BLOCKING_ISSUES.md`
- Tag screenshot af fejl
- Inkludér logs fra både lokal og HA

---

## ✅ DEFINITION OF DONE

**Acceptkriterier:**
1. [ ] Alle unit tests grønne (pytest exit code 0)
2. [ ] `verify_local_vs_ha.py` passer uden assertions
3. [ ] Dashboard viser forbrug data (visuel check + screenshot)
4. [ ] `ev_required_kwh` matematisk korrekt (79→100% = 15.75 kWh)
5. [ ] Historisk forbrug læses fra DB (log bekræftelse)
6. [ ] Dokumentation opdateret (README + test rapport)
7. [ ] Deployment til HA uden errors (supervisor logs clean)
8. [ ] User acceptance (explicit bekræftelse fra user)

**Quality Gates:**
- Code review: N/A (solo developer)
- Performance: Ingen regression i solver tid
- Backward compatibility: Existing configs virker stadig

---

**NÆSTE SKRIDT:** Afventer user godkendelse til at starte Fase 1 implementation.

---

*Dette dokument opdateres løbende gennem implementation.*

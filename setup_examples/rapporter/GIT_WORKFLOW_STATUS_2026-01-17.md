# Git Workflow & Deployment Status
**Dato**: 17. januar 2026, kl. 18:30  
**Status**: ✅ Alle ændringer committed, pushet og synkroniseret

---

## ✅ KORREKT GIT WORKFLOW GENNEMFØRT

### Repository Struktur

**Udviklings-repo** (primær):
```
z:\dev\energy_planner\
├── .git/
├── custom_components/
│   └── energy_planner/
│       ├── sensor.py
│       └── vendor/
│           └── energy_planner/
│               ├── optimizer/solver.py
│               └── scheduler.py
└── README.md
```
- **Git remote**: https://github.com/TorbenAT/energy_planner.git
- **Branch**: main
- **Commit**: `ebc7e5c` ✅

**Home Assistant live installation**:
```
z:\custom_components\energy_planner\
├── sensor.py (kopieret fra dev)
└── vendor/ (kopieret fra dev)
```
- **Git remote**: git@github.com:TorbenAT/home-assistant-config.git (ANDET repo)
- Dette er IKKE hvor energy_planner udvikles!

**Dashboard** (separat):
```
z:\dashboards\energy_planner_dashboard.yaml
```
- Del af home-assistant-config repo
- Opdateret lokalt (IKKE committed endnu)

---

## 🎯 Gennemførte Ændringer

### Git Commit Details
```
Commit: ebc7e5c
Author: TorbenAT
Date: 2026-01-17
Message: Fix: Current slot scaling + EV connection awareness + Dashboard improvements

Changes:
- OptimizationContext: Added remaining_minutes_in_current_slot and ev_connected
- solver.py: Scale max charge rates for partially elapsed first slot
- scheduler.py: Calculate remaining minutes, read EV connection status
- sensor.py: Added EV attributes (ev_next_charge_time, ev_connection_needed_by, 
  ev_total_planned_kwh, ev_charging_sessions, ev_connected)
```

**Ændrede filer** (4 total):
1. `custom_components/energy_planner/sensor.py` (+102 lines, -0 lines)
2. `custom_components/energy_planner/vendor/energy_planner/config.py` (minor)
3. `custom_components/energy_planner/vendor/energy_planner/optimizer/solver.py` (+42 lines, -5 lines)
4. `custom_components/energy_planner/vendor/energy_planner/scheduler.py` (+20 lines, -0 lines)

**Status**:
- ✅ Committed til dev repo
- ✅ Pushet til GitHub
- ✅ Synkroniseret til Home Assistant (`z:\custom_components\energy_planner\`)
- ✅ Python cache ryddet

---

## 📋 Deployment Checklist

### Pre-Deployment ✅
- [x] Ændringer lavet i `z:\dev\energy_planner\` (IKKE direkte i `custom_components`)
- [x] Git commit med beskrivende message
- [x] Git push til GitHub
- [x] Kopieret fra dev til `z:\custom_components\energy_planner\`
- [x] Ryddet `__pycache__` directories

### Post-Deployment ⏳
- [ ] Genstart Home Assistant
- [ ] Verificer sensor.energy_plan attributes
- [ ] Test dashboard på https://home.andsbjerg.dk/energy-planner/optimering
- [ ] Test current slot scaling (vent til kl. XX:50)
- [ ] Test EV connection warning (disconnect bil)

---

## 🔄 Fremtidig Workflow

**Når du laver ændringer i energy_planner koden:**

1. **Lav ændringer i dev repo**:
   ```powershell
   cd Z:\dev\energy_planner
   # Rediger filer her
   ```

2. **Commit og push til GitHub**:
   ```powershell
   git add -A
   git commit -m "Beskrivelse af ændringer"
   git push origin main
   ```

3. **Synkroniser til Home Assistant**:
   ```powershell
   # Kopier vendor
   Remove-Item -Path "Z:\custom_components\energy_planner\vendor\*" -Recurse -Force
   Copy-Item -Path "Z:\dev\energy_planner\custom_components\energy_planner\vendor\*" -Destination "Z:\custom_components\energy_planner\vendor\" -Recurse -Force
   
   # Kopier sensor.py (hvis ændret)
   Copy-Item -Path "Z:\dev\energy_planner\custom_components\energy_planner\sensor.py" -Destination "Z:\custom_components\energy_planner\sensor.py" -Force
   
   # Ryd cache
   Get-ChildItem -Path "Z:\custom_components\energy_planner" -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force
   ```

4. **Genstart Home Assistant**:
   - Developer Tools → System → Restart

**ALDRIG lav ændringer direkte i** `z:\custom_components\energy_planner\` medmindre det er hurtige tests!

---

## 📦 Deployment Script (Forslag)

Lav et script: `z:\scripts\deploy_energy_planner.ps1`

```powershell
#!/usr/bin/env pwsh
# Deploy energy_planner from dev to HA

Write-Host "🚀 Deploying energy_planner..."

# 1. Sync vendor
Write-Host "📁 Syncing vendor directory..."
Remove-Item -Path "Z:\custom_components\energy_planner\vendor\*" -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item -Path "Z:\dev\energy_planner\custom_components\energy_planner\vendor\*" -Destination "Z:\custom_components\energy_planner\vendor\" -Recurse -Force

# 2. Sync sensor.py
Write-Host "📄 Syncing sensor.py..."
Copy-Item -Path "Z:\dev\energy_planner\custom_components\energy_planner\sensor.py" -Destination "Z:\custom_components\energy_planner\sensor.py" -Force

# 3. Clear cache
Write-Host "🧹 Clearing Python cache..."
Get-ChildItem -Path "Z:\custom_components\energy_planner" -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force

Write-Host "✅ Deployment complete! Restart Home Assistant to apply changes."
```

Brug: `.\scripts\deploy_energy_planner.ps1`

---

## 🐛 Hvis Noget Går Galt

**Rollback til forrige version**:
```powershell
cd Z:\dev\energy_planner
git log --oneline -5  # Find commit hash
git checkout <previous-hash> custom_components/energy_planner/
git commit -m "Rollback: Reason for rollback"
git push origin main
# Kør deploy script
```

**Verificer synkronisering**:
```powershell
# Sammenlign dev vs HA
fc Z:\dev\energy_planner\custom_components\energy_planner\sensor.py Z:\custom_components\energy_planner\sensor.py
```

---

**Opdateret**: 2026-01-17 kl. 18:30  
**Next**: Genstart Home Assistant og test ændringer

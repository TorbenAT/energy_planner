# Migration fra v0.1.0 til v0.2.0

Version 0.2.0 skifter fra YAML-baseret til UI-baseret konfiguration med reload support.

## Hvad skal du gøre?

### 1. Fjern YAML sensor konfiguration

I filen `integrationer/energy_planer.yaml`, **kommenter ud eller fjern** denne blok:

```yaml
sensor:
  - platform: energy_planner
    name: Energy Plan
    plan_limit: 288
    markdown_limit: 3
    markdown_max_length: 3000
    scan_interval: "00:05:00"
```

**VIGTIG**: Behold alle andre ting i filen (utility_meter, input_number osv.)

### 2. Opdater integration via HACS

1. HACS → Energy Planner → Update til v0.2.0
2. Genstart Home Assistant

### 3. Tilføj integrationen via UI

1. Settings → Devices & Services → ADD INTEGRATION
2. Søg "Energy Planner"
3. Klik Submit

### 4. Konfigurer indstillinger

1. Settings → Devices & Services → Energy Planner → CONFIGURE
2. Indstil værdier (se dine gamle YAML værdier):
   - Plan limit: 288
   - Markdown limit: 3
   - Markdown max length: 3000
   - Scan interval: 5 minutter

### 5. Genstart Home Assistant

Nu kører integrationen med UI-konfiguration!

## Fordele ved v0.2.0

✅ **Reload support** - Genindlæs uden fuld genstart  
✅ **UI konfiguration** - Ret indstillinger via UI  
✅ **Standard Home Assistant måde** - Som andre integrationer  

## Hvis noget går galt

Gå tilbage til v0.1.0:
1. HACS → Energy Planner → Redownload
2. Vælg version v0.1.0
3. Genaktiver YAML konfigurationen
4. Genstart

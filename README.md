# Energy Planner

Home Assistant integration til intelligent energi- og ladeoptimering.

## Installation via HACS

1. Tilføj dette repository som custom repository i HACS:
   - HACS → ⋮ (menu) → Custom repositories
   - URL: `https://github.com/TorbenAT/energy_planner`
   - Kategori: Integration

2. Download integrationen fra HACS

3. Genstart Home Assistant

## Konfiguration

### Første gang (UI)

1. Gå til **Settings → Devices & Services**
2. Klik **+ ADD INTEGRATION**
3. Søg efter "Energy Planner"
4. Klik "Submit" for at tilføje integrationen med standardindstillinger

### Ændring af indstillinger

1. Gå til **Settings → Devices & Services**
2. Find "Energy Planner"
3. Klik **CONFIGURE**
4. Rediger indstillinger:
   - **Plan limit**: Antal 15-minutters intervaller i prognosen (standard: 288 = 3 dage)
   - **Markdown limit**: Antal rækker i markdown tabel (standard: 96)
   - **Markdown max length**: Maks længde af markdown (0 = ingen begrænsning)
   - **Scan interval**: Opdateringsinterval i minutter (standard: 15)
5. Klik **SUBMIT**

### .env fil (påkrævet)

Integrationen kræver en `.env` fil i `custom_components/energy_planner/.env` med følgende variabler:

```env
# Home Assistant
HA_BASE_URL=https://your-domain.com:443/api
HA_API_KEY=your_long_lived_access_token
HA_SECRETS_PATH=/config/secrets.yaml

# Database
MARIADB_DSN=mysql+pymysql://user:password@host:3306/energy_planner

# Configuration
ENERGY_TIMEZONE=Europe/Copenhagen
LOOKAHEAD_HOURS=72
RESOLUTION_MINUTES=60

# Sensors
HOUSE_CONSUMPTION_SENSOR=sensor.house_load_quarter_hour
EV_SOC_SENSOR=sensor.ev_battery
BATTERY_SOC_SENSOR=sensor.battery_soc
# ... (se .env.example for alle variabler)
```

Se `.env.example` for alle mulige konfigurationsvariabler.

## Reload under udvikling

For at genindlæse integrationen uden fuld genstart:

1. **Via UI**: Developer Tools → YAML → Genindlæs Integrationer (vælg Energy Planner)
2. **Via Service**: Kald service `homeassistant.reload_config_entry` med entry_id

Dette reloader koden og opdaterer sensoren med nye indstillinger.

## Services

### `energy_planner.update_plan`
Opdater energy plan nu (tvinger coordinator til at køre).

### `energy_planner.run_optimizer`
Kør den fulde optimizer pipeline og opdater database.

## Sensor

**`sensor.energy_plan`**
- State: Status tekst for den aktuelle plan
- Attributes:
  - `plan`: Array med planlagte 15-minutters intervaller
  - `hourly_summary`: Timebaseret opsummering
  - `markdown_*`: Markdown tabeller for visning i dashboards

## Udvikling

Projektet bruger:
- `vendor/energy_planner/`: Core optimeringslogik
- `coordinator.py`: Data opdateringskoordinator
- `sensor.py`: Home Assistant sensor platform
- `config_flow.py`: UI-baseret konfiguration

## Licens

Privat projekt - ikke offentlig licens.

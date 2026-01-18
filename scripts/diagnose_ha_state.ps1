# Diagnose script til at hente alle relevante HA sensorer for Energy Planner
$token = (Get-Content "Z:\.env" | Select-String "^HA_TOKEN=" | ForEach-Object { ($_ -replace "^HA_TOKEN=", "").Trim() })
if (-not $token) {
    Write-Error "HA_TOKEN ikke fundet i Z:\.env"
    exit
}

$headers = @{
    Authorization = "Bearer $token"
    "Content-Type" = "application/json"
}

$sensors = @(
    "sensor.energi_data_service",
    "sensor.energi_data_service_salg",
    "sensor.deye12_sun12k_battery_capacity",
    "input_number.deye12_sun12k_battery_capacity_watts",
    "input_number.deye12_sun12k_battery_charge_watts",
    "input_number.deye12_sun12k_battery_discharge_watts",
    "input_number.battery_minimum",
    "input_number.battery_maximum",
    "sensor.solcast_pv_forecast_forecast_today",
    "sensor.solcast_pv_forecast_forecast_tomorrow",
    "sensor.tessa_battery",
    "input_number.tesla_charge_procent_limit",
    "select.ev_smart_charging_charge_start_time",
    "select.ev_smart_charging_charge_completion_time",
    "sensor.easee_status",
    "switch.ev_smart_charging_smart_charging_activated",
    "sensor.deye12_sun12k_total_consumption",
    "sensor.deye12_sun12k_total_pv_production",
    "sensor.deye12_sun12k_total_charge_of_the_battery",
    "sensor.deye12_sun12k_total_discharge_of_the_battery",
    "sensor.deye12_sun12k_total_energy_bought",
    "sensor.deye12_sun12k_total_energy_sold",
    "input_number.energy_planner_house_expected_daily_kwh",
    "sensor.house_load_quarter_hour",
    "input_number.energy_planner_optimistic_charging_pct",
    "input_number.energy_planner_cheap_price_threshold_dkk",
    "input_select.energy_planner_profile",
    "input_boolean.energy_planner_multi_windows"
)

Write-Host "Henter data fra Home Assistant..." -ForegroundColor Cyan

$results = foreach ($s in $sensors) {
    try {
        $resp = Invoke-RestMethod -Uri "https://home.andsbjerg.dk/api/states/$s" -Headers $headers -Method Get
        [PSCustomObject]@{
            Entity = $s
            State = $resp.state
            Unit = $resp.attributes.unit_of_measurement
            FriendlyName = $resp.attributes.friendly_name
        }
    } catch {
        [PSCustomObject]@{
            Entity = $s
            State = "ERROR"
            Unit = ""
            FriendlyName = "Kunne ikke hentes"
        }
    }
}

$results | Format-Table -AutoSize

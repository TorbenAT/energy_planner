#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Health check for Energy Planner integration
    
.DESCRIPTION
    Verifies that sensor.energy_plan has all required attributes and data.
    Useful for post-restart verification and troubleshooting.
    
.EXAMPLE
    .\check_energy_plan_health.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "`n=== Energy Planner Health Check ===" -ForegroundColor Cyan

# Read token
if (-not (Test-Path "Z:\.env")) {
    Write-Error ".env file not found"
    exit 1
}

$token = Get-Content "Z:\.env" | Select-String "^HA_TOKEN=" | ForEach-Object { $_ -replace "^HA_TOKEN=", "" }
if (-not $token) {
    Write-Error "HA_TOKEN not found in .env file"
    exit 1
}

$headers = @{
    Authorization = "Bearer $token"
    "Content-Type" = "application/json"
}

# Fetch sensor
try {
    $sensor = Invoke-RestMethod -Uri "https://home.andsbjerg.dk/api/states/sensor.energy_plan" -Headers $headers -Method Get
}
catch {
    Write-Host "[✗] Failed to fetch sensor.energy_plan" -ForegroundColor Red
    Write-Error $_
    exit 1
}

Write-Host "[✓] Sensor accessible" -ForegroundColor Green

# Check critical attributes
$requiredAttrs = @(
    "plan",
    "plan_fields",
    "plan_today",
    "plan_tomorrow", 
    "plan_day3",
    "hourly",
    "hourly_fields",
    "objective",
    "notes",
    "generated_at"
)

$kpis = @(
    "grid_import_kwh",
    "grid_export_kwh",
    "house_consumption_kwh",
    "net_dkk",
    "cheapest_block"
)

Write-Host "`n--- Required Attributes ---" -ForegroundColor Cyan
$missingCount = 0
foreach ($attr in $requiredAttrs) {
    if ($null -ne $sensor.attributes.$attr) {
        $value = $sensor.attributes.$attr
        if ($value -is [array]) {
            $count = $value.Count
            Write-Host "[✓] $attr (array, $count items)" -ForegroundColor Green
        }
        else {
            Write-Host "[✓] $attr" -ForegroundColor Green
        }
    }
    else {
        Write-Host "[✗] $attr MISSING!" -ForegroundColor Red
        $missingCount++
    }
}

Write-Host "`n--- KPIs ---" -ForegroundColor Cyan
foreach ($kpi in $kpis) {
    if ($null -ne $sensor.attributes.$kpi) {
        $value = $sensor.attributes.$kpi
        Write-Host "[✓] $kpi = $value" -ForegroundColor Green
    }
    else {
        Write-Host "[✗] $kpi MISSING!" -ForegroundColor Red
        $missingCount++
    }
}

# Summary
Write-Host "`n--- Summary ---" -ForegroundColor Cyan
Write-Host "State: $($sensor.state)" -ForegroundColor $(if ($sensor.state -eq "Optimal") { "Green" } else { "Yellow" })
Write-Host "Last updated: $($sensor.last_updated)" -ForegroundColor Gray

if ($missingCount -eq 0) {
    Write-Host "`n✓ All checks passed!" -ForegroundColor Green
    Write-Host "Excel sheet should display data correctly." -ForegroundColor Green
}
else {
    Write-Host "`n✗ $missingCount attributes missing!" -ForegroundColor Red
    Write-Host "Excel sheet will likely show 'Ingen data'" -ForegroundColor Red
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Check logs for errors: logfiles\home-assistant*.log" -ForegroundColor Yellow
    Write-Host "2. Clear cache: .\scripts\clear_energy_planner_cache.ps1" -ForegroundColor Yellow
    Write-Host "3. Restart Home Assistant" -ForegroundColor Yellow
}

Write-Host ""

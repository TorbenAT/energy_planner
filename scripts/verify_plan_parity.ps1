<#
.SYNOPSIS
    Verificer paritet mellem lokal energy plan og HA deployment

.DESCRIPTION
    Kører lokal plan, trigger HA optimizer, og sammenligner EV SoC time-for-time.
    Tjekker også om consumption kolonner findes i begge outputs.

.EXAMPLE
    .\verify_plan_parity.ps1
#>

param()

$ErrorActionPreference = "Stop"

Write-Host "=" * 80
Write-Host "ENERGY PLANNER - PARITET VERIFIKATION" -ForegroundColor Cyan
Write-Host "=" * 80

# 1. Kør lokal plan
Write-Host "`nKører lokal plan..." -ForegroundColor Yellow
Push-Location "C:\Compile\Dev\energy_planner"
try {
    python run_local_plan.py | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "run_local_plan.py fejlede med exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

# Find latest export
$latestLocal = Get-ChildItem "Z:\logfiles\exports\local_plan_*.json" | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1

Write-Host "  OK - $($latestLocal.Name)" -ForegroundColor Green

# 2. Hent HA token
$token = (Get-Content "Z:\.env" | Select-String "^HA_TOKEN=" | ForEach-Object { ($_ -replace "^HA_TOKEN=", "").Trim() })
if (-not $token) {
    throw "HA_TOKEN not found in Z:\.env"
}

# 3. Trigger HA optimizer
Write-Host "`nTrigger HA optimizer..." -ForegroundColor Yellow
$headers = @{ 
    Authorization = "Bearer $token"
    "Content-Type" = "application/json" 
}

try {
    Invoke-RestMethod -Uri "https://home.andsbjerg.dk/api/services/energy_planner/run_optimizer" `
                      -Headers $headers `
                      -Method Post `
                      -Body "{}" `
                      -TimeoutSec 180 | Out-Null
    Write-Host "  OK - Optimizer triggered" -ForegroundColor Green
} catch {
    Write-Host "  FEJL: $_" -ForegroundColor Red
    throw
}

# 4. Vent 15 sekunder
Write-Host "`nVenter 15 sekunder på HA optimization..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# 5. Hent HA sensor
Write-Host "Henter HA sensor state..." -ForegroundColor Yellow
$haSensor = Invoke-RestMethod -Uri "https://home.andsbjerg.dk/api/states/sensor.energy_plan" `
                               -Headers $headers `
                               -Method Get

Write-Host "  OK - generated_at: $($haSensor.attributes.generated_at)" -ForegroundColor Green

# 6. Load lokal plan
$localPlan = Get-Content $latestLocal.FullName -Raw | ConvertFrom-Json

# 7. Sammenlign EV SoC
Write-Host "`nSammenligner EV SoC..." -ForegroundColor Yellow

# Extract local SoC
$localMap = @{}
foreach ($slot in $localPlan.plan) {
    if ($slot.timestamp -and ($null -ne $slot.ev_soc_pct)) {
        $dt = [DateTime]::Parse($slot.timestamp).ToUniversalTime()
        $hour = $dt.ToString("yyyy-MM-dd HH:00")
        $localMap[$hour] = [double]$slot.ev_soc_pct
    }
}

# Extract HA SoC
$fields = $haSensor.attributes.plan_fields
$tsIdx = [array]::IndexOf($fields, 'timestamp')
$socIdx = [array]::IndexOf($fields, 'ev_soc_pct')

$haMap = @{}
foreach ($slot in $haSensor.attributes.plan) {
    $ts = $slot[$tsIdx]
    $soc = $slot[$socIdx]
    $dt = [DateTime]::Parse($ts).ToUniversalTime()
    $hour = $dt.ToString("yyyy-MM-dd HH:00")
    $haMap[$hour] = [double]$soc
}

# Compare
$overlap = 0
$mismatches = @()
foreach ($hour in ($localMap.Keys | Where-Object { $haMap.ContainsKey($_) } | Sort-Object)) {
    $overlap++
    $l = $localMap[$hour]
    $h = $haMap[$hour]
    $diff = $l - $h
    
    if ([Math]::Abs($diff) -gt 0.01) {
        $mismatches += [PSCustomObject]@{
            Hour = $hour
            Local = $l
            HA = $h
            Diff = $diff
        }
    }
}

Write-Host "  Overlap: $overlap timer" -ForegroundColor Cyan
Write-Host "  Mismatches: $($mismatches.Count)" -ForegroundColor $(if ($mismatches.Count -eq 0) { "Green" } else { "Red" })

if ($mismatches.Count -gt 0) {
    Write-Host "`n  Første 10 mismatches:" -ForegroundColor Yellow
    $mismatches | Select-Object -First 10 | ForEach-Object {
        Write-Host ("    {0} | local={1:F2} | HA={2:F2} | diff={3:+F2}" -f $_.Hour, $_.Local, $_.HA, $_.Diff)
    }
}

# 8. Tjek consumption kolonner
Write-Host "`nTjekker consumption kolonner..." -ForegroundColor Yellow

# Local
$localHasCons = $null -ne ($localPlan.plan[0].PSObject.Properties | Where-Object { $_.Name -eq 'ev_driving_consumption_kwh' })
if ($localHasCons) {
    $localConsSum = ($localPlan.plan | ForEach-Object { $_.ev_driving_consumption_kwh } | Measure-Object -Sum).Sum
    Write-Host "  Local: ✓ ev_driving_consumption_kwh findes (total $($localConsSum.ToString('F2')) kWh)" -ForegroundColor Green
} else {
    Write-Host "  Local: ✗ ev_driving_consumption_kwh MANGLER!" -ForegroundColor Red
}

# HA
$haHasCons = $fields -contains 'ev_driving_consumption_kwh'
if ($haHasCons) {
    $consIdx = [array]::IndexOf($fields, 'ev_driving_consumption_kwh')
    $haConsSum = ($haSensor.attributes.plan | ForEach-Object { $_[$consIdx] } | Measure-Object -Sum).Sum
    Write-Host "  HA:    ✓ ev_driving_consumption_kwh findes (total $($haConsSum.ToString('F2')) kWh)" -ForegroundColor Green
} else {
    Write-Host "  HA:    ✗ ev_driving_consumption_kwh MANGLER!" -ForegroundColor Red
}

# 9. Resultat
Write-Host "`n$("=" * 80)"
if ($mismatches.Count -eq 0 -and $localHasCons -and $haHasCons) {
    Write-Host "STATUS: ✓ SUCCESS - Fuld paritet opnået!" -ForegroundColor Green
    Write-Host "$("=" * 80)"
    exit 0
} else {
    Write-Host "STATUS: ✗ MISMATCH - Paritet IKKE opnået!" -ForegroundColor Red
    Write-Host "$("=" * 80)"
    exit 1
}

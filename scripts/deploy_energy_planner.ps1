#!/usr/bin/env pwsh
# Deploy energy_planner from dev to Home Assistant
# Usage: .\deploy_energy_planner.ps1

param(
    [switch]$Restart = $false
)

function Start-Countdown {
    param([int]$Seconds)
    for ($i = $Seconds; $i -gt 0; $i--) {
        $remaining = New-TimeSpan -Seconds $i
        Write-Host -NoNewline ("`r      Venter: {0:mm\:ss} tilbage...   " -f $remaining)
        Start-Sleep -Seconds 1
    }
    Write-Host "`r      OK - Ventetid afsluttet.          " -ForegroundColor Green
}

Write-Host "Deploying energy_planner from dev to HA..." -ForegroundColor Cyan
Write-Host ""

# Paths
$DevPath = "C:\Compile\Dev\energy_planner\custom_components\energy_planner"
$HAPath = "Z:\custom_components\energy_planner"
$DevScriptPath = "C:\Compile\Dev\energy_planner\scripts"
$HAScriptPath = "Z:\scripts"

if (-not (Test-Path $DevPath)) {
    Write-Error "Source path $DevPath not found!"
    exit 1
}

# 1. Sync all files using Robocopy (more robust than Copy-Item)
Write-Host "[1/3] Syncing components and scripts..." -ForegroundColor Yellow
& robocopy "$DevPath" "$HAPath" /E /Z /R:5 /W:5 /MT:32 /XD __pycache__ .git | Out-Null
& robocopy "$DevScriptPath" "$HAScriptPath" /E /Z /R:5 /W:5 /MT:32 /XD __pycache__ .git | Out-Null

if ($LASTEXITCODE -ge 8) {
    Write-Host "      ERROR - Robocopy failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}
Write-Host "      OK - Files synced" -ForegroundColor Green

# 2. Clear Python cache
Write-Host "[2/3] Clearing Python cache..." -ForegroundColor Yellow
$CacheDirs = Get-ChildItem -Path $HAPath -Filter "__pycache__" -Recurse -Directory -ErrorAction SilentlyContinue
foreach ($CacheDir in $CacheDirs) {
    Remove-Item -Path $CacheDir.FullName -Recurse -Force
}
Write-Host "      OK - Cache cleared" -ForegroundColor Green

# 3. Optional Restart
if ($Restart) {
    Write-Host "[3/3] Genstarter Home Assistant og kører optimering..." -ForegroundColor Yellow
    $token = (Get-Content "Z:\.env" | Select-String "^HA_TOKEN=" | ForEach-Object { ($_ -replace "^HA_TOKEN=", "").Trim() })
    $headers = @{ Authorization = "Bearer $token"; "Content-Type" = "application/json" }
    
    $restartTriggered = $false
    try {
        Invoke-RestMethod -Uri "https://home.andsbjerg.dk/api/services/homeassistant/restart" -Headers $headers -Method Post | Out-Null
        Write-Host "      OK - Genstart-kommando sendt." -ForegroundColor Green
        $restartTriggered = $true
    } catch {
        # 504 Gateway Timeout is common when proxying HA restart
        if ($_.Exception.Message -match "504" -or $_.Exception.Message -match "timeout" -or $_.Exception.Message -match "The connection was closed") {
            Write-Host "      OK - Genstart-kommando sendt (modtog dog timeout/504, hvilket er normalt)." -ForegroundColor Green
            $restartTriggered = $true
        } else {
            Write-Host "      FEJL ved genstart: $_" -ForegroundColor Red
        }
    }

    if ($restartTriggered) {
        Write-Host "      Venter 90 sekunder på at HA starter op..." -ForegroundColor Cyan
        Start-Countdown -Seconds 90
        
        Write-Host "      Trigger run_optimizer service..." -ForegroundColor Yellow
        try {
            Invoke-RestMethod -Uri "https://home.andsbjerg.dk/api/services/energy_planner/run_optimizer" -Headers $headers -Method Post -Body '{}' | Out-Null
            Write-Host "      Success! Tjekker sensor status om 10 sek..." -ForegroundColor Green
            Start-Countdown -Seconds 10
            
            $sensor = Invoke-RestMethod -Uri "https://home.andsbjerg.dk/api/states/sensor.energy_plan" -Headers $headers -Method Get
            Write-Host "      Sensor status: $($sensor.state)" -ForegroundColor Cyan
            $sensor.attributes | ConvertTo-Json -Depth 2 | Write-Host
        } catch {
            Write-Host "      FEJL ved optimering: $_" -ForegroundColor Red
            Write-Host "      Henter logs for at diagnosticere..." -ForegroundColor Yellow
            $log = Invoke-RestMethod -Uri "https://home.andsbjerg.dk/api/hassio/supervisor/logs" -Headers $headers -Method Get
            $log | Out-String | Select-String "energy_planner|Traceback" -Context 0,20 | Select-Object -Last 10 | Write-Host
        }
    }
} else {
    Write-Host "[3/3] Springer genstart over (brug -Restart for at automatisere)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Deployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "   1. If you didn't use -Restart, restart HA manually now."
Write-Host "   2. WAIT 90 SECONDS before checking logs!"
Write-Host "   3. Check logs via supervisor API."
Write-Host "   4. Verify sensor.energy_plan attributes"
Write-Host ""

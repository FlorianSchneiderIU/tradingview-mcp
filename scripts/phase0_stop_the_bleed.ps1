<#
.SYNOPSIS
    Phase 0 of the strategy remediation - "stop the bleed".
    Disables the unvalidated/bleeding strategies, halves per-trade risk, and
    enables the post-TP1 lock-in. Reversible (backs up every file it touches).

.DESCRIPTION
    The bot runs via docker-compose and reads env from bot/.env (env_file).
    Most flags can be set in bot/.env, BUT docker-compose.yml pins
    ENABLE_GGSHOT_227 inline under environment:, which OVERRIDES env_file -
    so GGShot is disabled by editing the compose file instead.

    Idempotent: re-running updates existing keys in place rather than appending
    duplicates.

.PARAMETER Restart
    After applying changes, recreate the mm-bot container so it picks them up
    (docker compose up -d mm-bot).

.PARAMETER RestrictWolfe
    Instead of fully disabling Wolfe, keep it on but point it at the rigorously
    validated 3-symbol config (LINK/LTC/SOL). Off by default - Phase 0 disables
    Wolfe entirely.

.EXAMPLE
    ./scripts/phase0_stop_the_bleed.ps1
    ./scripts/phase0_stop_the_bleed.ps1 -Restart
#>
[CmdletBinding()]
param(
    [switch]$Restart,
    [switch]$RestrictWolfe
)

$ErrorActionPreference = 'Stop'

# Repo root = parent of this script's directory.
$RepoRoot    = Split-Path -Parent $PSScriptRoot
$EnvFile     = Join-Path $RepoRoot 'bot/.env'
$ComposeFile = Join-Path $RepoRoot 'docker-compose.yml'
$Stamp       = Get-Date -Format 'yyyyMMdd-HHmmss'

# Phase 0 key/value overrides applied to bot/.env.
$EnvSettings = [ordered]@{
    'ENABLE_WOLFE_WAVE'    = 'false'   # worst bleeder: 78 pct SL, -714 usd
    'ENABLE_WOLFE_WAVE_V2' = 'false'
    'ENABLE_SESSION_ORB'   = 'false'   # 54 pct SL
    'ENABLE_TURTLE_SOUP'   = 'false'   # on-by-default + unvalidated
    'ALLOW_MM_WITHOUT_DT'  = 'false'   # disable MM symbols lacking a fresh model
    'NOTIONAL_PCT'         = '0.005'   # halve per-trade risk during remediation
    'BREAKEVEN_LOCKIN_R'   = '0.3'     # lock-in floor on the runner after TP1
}

if ($RestrictWolfe) {
    # If keeping Wolfe on, run only the validated 3 and gate it.
    $EnvSettings['ENABLE_WOLFE_WAVE']      = 'true'
    $EnvSettings['WOLFE_WAVE_CONFIG_PATH'] = '/app/configs/wolfe_wave_universe_4y_oos1y_stage40_configs.json'
    $EnvSettings['WOLFE_WAVE_SYMBOLS']     = 'LINKUSDT,LTCUSDT,SOLUSDT'
}

function Write-Utf8NoBom([string]$Path, [string]$Content) {
    # Avoid the UTF-16/BOM that Set-Content emits in Windows PowerShell 5.1 -
    # a BOM on the first line can break naive .env parsers.
    [System.IO.File]::WriteAllText($Path, $Content, (New-Object System.Text.UTF8Encoding($false)))
}

# --- 1. bot/.env -------------------------------------------------------------
if (-not (Test-Path $EnvFile)) {
    Write-Warning "bot/.env not found - creating a new one (no API credentials in it!)."
    Write-Utf8NoBom $EnvFile ""
} else {
    $bak = "$EnvFile.bak-$Stamp"
    Copy-Item $EnvFile $bak
    Write-Host "Backed up bot/.env -> $bak"
}

# Read existing lines (preserve everything else, including secrets).
$lines = @(Get-Content -LiteralPath $EnvFile -Encoding UTF8)

foreach ($key in $EnvSettings.Keys) {
    $val     = $EnvSettings[$key]
    $newLine = "$key=$val"
    $pattern = '^\s*' + [regex]::Escape($key) + '\s*='
    $found   = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $pattern) {
            if ($lines[$i] -ne $newLine) {
                Write-Host ("  .env  {0,-24} {1} -> {2}" -f $key, $lines[$i], $newLine)
            } else {
                Write-Host ("  .env  {0,-24} already {1}" -f $key, $val)
            }
            $lines[$i] = $newLine
            $found = $true
            break
        }
    }
    if (-not $found) {
        Write-Host ("  .env  {0,-24} + {1}" -f $key, $newLine)
        $lines += $newLine
    }
}

Write-Utf8NoBom $EnvFile (($lines -join "`n") + "`n")
Write-Host "Wrote bot/.env"

# --- 2. docker-compose.yml (GGShot is pinned inline, overrides env_file) ------
if (-not (Test-Path $ComposeFile)) {
    throw "docker-compose.yml not found at $ComposeFile"
}
$composeText = Get-Content -LiteralPath $ComposeFile -Raw -Encoding UTF8
if ($composeText -match 'ENABLE_GGSHOT_227:\s*"true"') {
    $bak = "$ComposeFile.bak-$Stamp"
    Copy-Item $ComposeFile $bak
    Write-Host "Backed up docker-compose.yml -> $bak"
    $composeText = $composeText -replace 'ENABLE_GGSHOT_227:\s*"true"', 'ENABLE_GGSHOT_227: "false"'
    Write-Utf8NoBom $ComposeFile $composeText
    Write-Host '  compose  ENABLE_GGSHOT_227: "true" -> "false"'
} elseif ($composeText -match 'ENABLE_GGSHOT_227:\s*"false"') {
    Write-Host '  compose  ENABLE_GGSHOT_227 already "false"'
} else {
    Write-Warning '  compose  ENABLE_GGSHOT_227 line not found - set ENABLE_GGSHOT_227=false in bot/.env manually if needed.'
}

# --- 3. Restart (optional) ----------------------------------------------------
if ($Restart) {
    Write-Host "`nRecreating mm-bot to apply changes ..."
    Push-Location $RepoRoot
    try {
        docker compose up -d mm-bot
    } finally {
        Pop-Location
    }
    Write-Host "Done. Tail logs with:  docker compose logs -f mm-bot"
} else {
    Write-Host "`nChanges written. Apply them with:"
    Write-Host "    docker compose up -d mm-bot"
    Write-Host "(or re-run this script with -Restart)"
}

Write-Host "`nVerify: watch bot/logs/trade_ledger.jsonl for NO new entries from"
Write-Host "wolfe_wave / wolfe_wave_v2 / session_orb_judas_fvg / ggshot_227 / turtle_soup."

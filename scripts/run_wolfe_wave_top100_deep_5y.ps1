param(
    [int]$Limit = 100,
    [int]$Workers = 4,
    [int]$MaxRandomConfigs = 48,
    [int]$RefineSeeds = 8,
    [int]$RefineSamplesPerSeed = 48,
    [string]$RunName = "",
    [switch]$Refresh,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "wolfe_wave_top100_deep5y_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

$OutputDir = Join-Path "scripts" $RunName
$CacheDir = "scripts\data_wolfe_top100_deep5y"
$TemplateCandidates = "scripts\wolfe_wave_universe_feeaware_consolidated_20260601\candidate_retest.csv"
$LiveConfig = "bot\configs\wolfe_wave_configs.json"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

$ExtraSymbols = @()
if (Test-Path -LiteralPath $LiveConfig) {
    $ExtraSymbols = (Get-Content -LiteralPath $LiveConfig -Raw | ConvertFrom-Json).PSObject.Properties.Name | Sort-Object
}

$PythonArgs = @(
    "scripts\run_wolfe_wave_top100_lowpass.py",
    "--limit", "$Limit",
    "--days", "1825",
    "--min-history-days", "365",
    "--min-bars", "75000",
    "--max-configs", "$MaxRandomConfigs",
    "--template-candidates", $TemplateCandidates,
    "--workers", "$Workers",
    "--pattern-tfs", "5m,15m,1h,4h",
    "--regime-filters", "none,high_vol,low_vol,trend_aligned,mean_reversion",
    "--min-train", "20",
    "--min-validation", "8",
    "--min-oos-net-r", "1.0",
    "--min-all-net-r", "8.0",
    "--lowpass-radius", "0.45",
    "--lowpass-min-neighbors", "9",
    "--lowpass-outlier-penalty", "0.65",
    "--refine",
    "--refine-min-score", "0",
    "--refine-seeds", "$RefineSeeds",
    "--refine-samples-per-seed", "$RefineSamplesPerSeed",
    "--refine-neighbor-width", "1",
    "--cache-dir", $CacheDir,
    "--output-dir", $OutputDir
)

if ($Refresh) {
    $PythonArgs += "--refresh"
}

if ($Resume) {
    $PythonArgs += "--resume"
}

if ($ExtraSymbols.Count -gt 0) {
    $PythonArgs += "--symbols"
    $PythonArgs += $ExtraSymbols
}

$CommandPath = Join-Path $OutputDir "command.txt"
$LogPath = Join-Path $OutputDir "run.log"
$ErrPath = Join-Path $OutputDir "run.err"

@(
    "Started: $(Get-Date -Format o)",
    "Repo: $RepoRoot",
    "OutputDir: $OutputDir",
    "CacheDir: $CacheDir",
    "Limit: $Limit",
    "Workers: $Workers",
    "MaxRandomConfigs: $MaxRandomConfigs",
    "RefineSeeds: $RefineSeeds",
    "RefineSamplesPerSeed: $RefineSamplesPerSeed",
    "Extra live symbols appended: $($ExtraSymbols -join ',')",
    "",
    "python $($PythonArgs -join ' ')"
) | Set-Content -LiteralPath $CommandPath -Encoding utf8

Write-Host "Starting Wolfe 5y deep sweep"
Write-Host "Output: $OutputDir"
Write-Host "Log:    $LogPath"
Write-Host "Cache:  $CacheDir"
Write-Host ""
Write-Host "Use this to continue the same run later:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_wolfe_wave_top100_deep_5y.ps1 -RunName $RunName -Resume"
Write-Host ""

& python @PythonArgs 2> $ErrPath | Tee-Object -FilePath $LogPath
$ExitCode = $LASTEXITCODE

"Finished: $(Get-Date -Format o); exit=$ExitCode" | Add-Content -LiteralPath $CommandPath -Encoding utf8
if ($ExitCode -ne 0) {
    Write-Error "Wolfe deep sweep failed with exit code $ExitCode. See $ErrPath"
}

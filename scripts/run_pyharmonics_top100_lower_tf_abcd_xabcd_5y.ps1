param(
    [int]$Limit = 100,
    [int]$Workers = 2,
    [int]$MaxConfigs = 120,
    [string]$RunName = "",
    [string[]]$Symbols = @(),
    [switch]$Refresh,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "pyharmonics_top100_lower_tf_abcd_xabcd_5y_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

$OutputDir = Join-Path "scripts" $RunName
$CacheDir = "scripts\data_pyharmonics_top100_lower_tf_abcd_xabcd_5y"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

$PythonArgs = @(
    "scripts\run_pyharmonics_top100_lowpass.py",
    "--limit", "$Limit",
    "--days", "1825",
    "--exec-tf", "5m",
    "--min-history-days", "365",
    "--min-bars", "80000",
    "--validation-days", "365",
    "--oos-days", "365",
    "--workers", "$Workers",
    "--max-configs", "$MaxConfigs",
    "--pattern-tfs", "5m,15m",
    "--families", "ABCD,XABCD",
    "--pattern-modes", "formed,forming",
    "--peak-spacings", "10,16",
    "--fib-tolerances", "0.03",
    "--forming-percents", "0.85",
    "--collapse-formed-forming-percents",
    "--pattern-lookback-bars", "1440",
    "--pattern-step-bars", "720",
    "--search-limit-to", "6",
    "--confirm-bars", "3,6,10,16",
    "--entry-window-bars", "6,12,24,48",
    "--prz-buffers", "0.05,0.10,0.20",
    "--candle-filters", "none,any_reversal,engulfing,pinbar,reclaim,strong_close,outside_reversal",
    "--stop-buffers", "0.2,0.5,0.8",
    "--rrs", "1.5,2.0,2.5",
    "--max-hold-bars", "36,72,144,288",
    "--trend-filters", "none,counter_ema",
    "--max-fee-to-price-risk", "0.18",
    "--min-entry-risk-pct", "0.0015",
    "--lowpass-radius", "1.80",
    "--lowpass-min-neighbors", "7",
    "--min-validation", "30",
    "--min-lowpass-score", "0.50",
    "--min-oos-net-r", "2.0",
    "--min-validation-net-r", "1.0",
    "--min-all-net-r", "10.0",
    "--min-avg-r", "0.02",
    "--max-all-dd-r", "18.0",
    "--progress-every-configs", "20",
    "--event-progress-every-chunks", "100",
    "--persist-event-cache",
    "--cache-dir", $CacheDir,
    "--output-dir", $OutputDir
)

if ($Refresh) {
    $PythonArgs += "--refresh"
}

if ($Resume) {
    $PythonArgs += "--resume"
}

if ($Symbols.Count -gt 0) {
    $PythonArgs += "--only-symbols"
    $PythonArgs += "--symbols"
    $PythonArgs += $Symbols
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
    "MaxConfigs: $MaxConfigs",
    "",
    "python $($PythonArgs -join ' ')"
) | Set-Content -LiteralPath $CommandPath -Encoding utf8

Write-Host "Starting pyharmonics 5y top-100 lower-TF ABCD/XABCD low-pass sweep"
Write-Host "Output: $OutputDir"
Write-Host "Log:    $LogPath"
Write-Host "Cache:  $CacheDir"
Write-Host ""
Write-Host "Use this to continue the same run later:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_pyharmonics_top100_lower_tf_abcd_xabcd_5y.ps1 -RunName $RunName -Resume"
Write-Host ""

& python @PythonArgs 2> $ErrPath | Tee-Object -FilePath $LogPath
$ExitCode = $LASTEXITCODE

"Finished: $(Get-Date -Format o); exit=$ExitCode" | Add-Content -LiteralPath $CommandPath -Encoding utf8
if ($ExitCode -ne 0) {
    Write-Error "pyharmonics lower-TF sweep failed with exit code $ExitCode. See $ErrPath"
}

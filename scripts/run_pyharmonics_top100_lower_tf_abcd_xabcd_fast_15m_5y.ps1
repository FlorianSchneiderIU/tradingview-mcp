param(
    [int]$Limit = 100,
    [int]$Workers = 4,
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
    $RunName = "pyharmonics_top100_fast_15m_abcd_xabcd_5y_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

$OutputDir = Join-Path "scripts" $RunName
$CacheDir = "scripts\data_pyharmonics_top100_fast_15m_abcd_xabcd_5y"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

$PythonArgs = @(
    "scripts\run_pyharmonics_top100_lowpass.py",
    "--limit", "$Limit",
    "--days", "1825",
    "--exec-tf", "15m",
    "--min-history-days", "365",
    "--min-bars", "25000",
    "--validation-days", "365",
    "--oos-days", "365",
    "--workers", "$Workers",
    "--max-configs", "$MaxConfigs",
    "--pattern-tfs", "15m",
    "--families", "ABCD,XABCD",
    "--pattern-modes", "formed,forming",
    "--peak-spacings", "10,16",
    "--fib-tolerances", "0.03",
    "--forming-percents", "0.85",
    "--collapse-formed-forming-percents",
    "--pattern-lookback-bars", "800",
    "--pattern-step-bars", "800",
    "--search-limit-to", "6",
    "--confirm-bars", "2,4,8,12",
    "--entry-window-bars", "4,8,16,32",
    "--prz-buffers", "0.05,0.10,0.20",
    "--candle-filters", "none,any_reversal,engulfing,pinbar,reclaim,strong_close,outside_reversal",
    "--stop-buffers", "0.2,0.5,0.8",
    "--rrs", "1.5,2.0,2.5",
    "--max-hold-bars", "24,48,96,192",
    "--trend-filters", "none,counter_ema",
    "--max-fee-to-price-risk", "0.18",
    "--min-entry-risk-pct", "0.0015",
    "--lowpass-radius", "1.80",
    "--lowpass-min-neighbors", "7",
    "--min-validation", "20",
    "--min-lowpass-score", "0.25",
    "--min-oos-net-r", "1.5",
    "--min-validation-net-r", "0.5",
    "--min-all-net-r", "8.0",
    "--min-avg-r", "0.02",
    "--max-all-dd-r", "16.0",
    "--progress-every-configs", "10",
    "--event-progress-every-chunks", "50",
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

Write-Host "Starting pyharmonics fast 15m top-100 ABCD/XABCD low-pass sweep"
Write-Host "Output: $OutputDir"
Write-Host "Log:    $LogPath"
Write-Host "Cache:  $CacheDir"
Write-Host ""
Write-Host "Use this to continue the same run later:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_pyharmonics_top100_lower_tf_abcd_xabcd_fast_15m_5y.ps1 -RunName $RunName -Resume"
Write-Host ""

& python @PythonArgs 2> $ErrPath | Tee-Object -FilePath $LogPath
$ExitCode = $LASTEXITCODE

"Finished: $(Get-Date -Format o); exit=$ExitCode" | Add-Content -LiteralPath $CommandPath -Encoding utf8
if ($ExitCode -ne 0) {
    Write-Error "pyharmonics fast 15m sweep failed with exit code $ExitCode. See $ErrPath"
}

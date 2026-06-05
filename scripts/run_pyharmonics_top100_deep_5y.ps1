param(
    [int]$Limit = 100,
    [int]$Workers = 4,
    [int]$MaxConfigs = 360,
    [string]$RunName = "",
    [switch]$Refresh,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "pyharmonics_top100_deep5y_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

$OutputDir = Join-Path "scripts" $RunName
$CacheDir = "scripts\data_pyharmonics_top100_deep5y"

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
    "--pattern-tfs", "1h,4h",
    "--families", "ABC,ABCD,XABCD",
    "--pattern-modes", "formed,forming",
    "--peak-spacings", "10,20,28",
    "--fib-tolerances", "0.02,0.03,0.05",
    "--forming-percents", "0.70,0.80,0.90",
    "--pattern-lookback-bars", "800",
    "--pattern-step-bars", "200",
    "--search-limit-to", "8",
    "--confirm-bars", "10,20,30",
    "--entry-window-bars", "24,48",
    "--prz-buffers", "0.10,0.25",
    "--candle-filters", "none,any_reversal,engulfing,pinbar,reclaim,strong_close",
    "--stop-buffers", "0.2,0.5,0.8",
    "--rrs", "1.25,1.5,2.0",
    "--max-hold-bars", "48,96,192",
    "--trend-filters", "none,with_ema,counter_ema",
    "--lowpass-radius", "1.60",
    "--lowpass-min-neighbors", "7",
    "--min-validation", "12",
    "--min-lowpass-score", "0.0",
    "--min-oos-net-r", "1.0",
    "--min-validation-net-r", "0.0",
    "--min-all-net-r", "5.0",
    "--min-avg-r", "0.0",
    "--cache-dir", $CacheDir,
    "--output-dir", $OutputDir
)

if ($Refresh) {
    $PythonArgs += "--refresh"
}

if ($Resume) {
    $PythonArgs += "--resume"
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

Write-Host "Starting pyharmonics 5y top-100 low-pass sweep"
Write-Host "Output: $OutputDir"
Write-Host "Log:    $LogPath"
Write-Host "Cache:  $CacheDir"
Write-Host ""
Write-Host "Use this to continue the same run later:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_pyharmonics_top100_deep_5y.ps1 -RunName $RunName -Resume"
Write-Host ""

& python @PythonArgs 2> $ErrPath | Tee-Object -FilePath $LogPath
$ExitCode = $LASTEXITCODE

"Finished: $(Get-Date -Format o); exit=$ExitCode" | Add-Content -LiteralPath $CommandPath -Encoding utf8
if ($ExitCode -ne 0) {
    Write-Error "pyharmonics deep sweep failed with exit code $ExitCode. See $ErrPath"
}

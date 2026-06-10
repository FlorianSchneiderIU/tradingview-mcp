param(
    [int]$Limit = 100,
    [int]$Workers = 4,
    [int]$MaxRandomConfigs = 48,
    [int]$RefineSeeds = 8,
    [int]$RefineSamplesPerSeed = 48,
    [string]$RunName = "",
    [switch]$Refresh,
    [switch]$Resume,
    [switch]$EnableV2Grid,
    [string]$PythonExe = "",
    [switch]$CheckEnvOnly
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Test-WolfePythonEnv {
    param([string]$Candidate)
    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return $false
    }
    $Probe = "import pandas, numpy, requests"
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Candidate -c $Probe 1>$null 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
}

function Add-PythonCandidate {
    param(
        [string[]]$Current,
        [string]$Candidate
    )
    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return $Current
    }
    if ($Current -contains $Candidate) {
        return $Current
    }
    return @($Current + $Candidate)
}

$PythonCandidates = @()
if (![string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonCandidates = Add-PythonCandidate $PythonCandidates $PythonExe
}
else {
    $PythonCandidates = Add-PythonCandidate $PythonCandidates $env:WOLFE_PYTHON
    $PythonCandidates = Add-PythonCandidate $PythonCandidates (Join-Path $RepoRoot ".venv\Scripts\python.exe")
    foreach ($Candidate in @(where.exe python 2>$null)) {
        $PythonCandidates = Add-PythonCandidate $PythonCandidates $Candidate
    }
    foreach ($Line in @(py -0p 2>$null)) {
        if ($Line -match "([A-Za-z]:\\.*python(?:\.exe)?)\s*$") {
            $PythonCandidates = Add-PythonCandidate $PythonCandidates $Matches[1]
        }
    }
}

$SelectedPython = $null
foreach ($Candidate in $PythonCandidates) {
    if (Test-WolfePythonEnv $Candidate) {
        $SelectedPython = $Candidate
        break
    }
}

if ([string]::IsNullOrWhiteSpace($SelectedPython)) {
    $Tried = if ($PythonCandidates.Count -gt 0) { $PythonCandidates -join "`n  " } else { "(none)" }
    throw "No Python interpreter with pandas, numpy, and requests was found. Tried:`n  $Tried`nPass -PythonExe C:\path\to\python.exe or install dependencies into the repo .venv."
}

if ($CheckEnvOnly) {
    Write-Host "Wolfe Python environment OK: $SelectedPython"
    & $SelectedPython -c "import sys, pandas, numpy, requests; print(sys.executable); print('pandas', pandas.__version__); print('numpy', numpy.__version__); print('requests', requests.__version__)"
    exit $LASTEXITCODE
}

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $Prefix = if ($EnableV2Grid) { "wolfe_wave_v2_top100_deep5y" } else { "wolfe_wave_top100_deep5y" }
    $RunName = "{0}_{1}" -f $Prefix, (Get-Date -Format "yyyyMMdd_HHmmss")
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

if ($EnableV2Grid) {
    $PythonArgs += "--enable-v2-grid"
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
    "EnableV2Grid: $EnableV2Grid",
    "PythonExe: $SelectedPython",
    "Extra live symbols appended: $($ExtraSymbols -join ',')",
    "",
    "$SelectedPython $($PythonArgs -join ' ')"
) | Set-Content -LiteralPath $CommandPath -Encoding utf8

Write-Host "Starting Wolfe 5y deep sweep"
Write-Host "Output: $OutputDir"
Write-Host "Log:    $LogPath"
Write-Host "Cache:  $CacheDir"
Write-Host "Python: $SelectedPython"
Write-Host ""
Write-Host "Use this to continue the same run later:"
$ResumeV2 = if ($EnableV2Grid) { " -EnableV2Grid" } else { "" }
$ResumePython = " -PythonExe `"$SelectedPython`""
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_wolfe_wave_top100_deep_5y.ps1 -RunName $RunName -Resume$ResumeV2$ResumePython"
Write-Host ""

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $SelectedPython @PythonArgs 2>&1 | Tee-Object -FilePath $LogPath
    $PythonExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
}

if ($PythonExitCode -ne 0) {
    @(
        "Python exited with code $PythonExitCode.",
        "See full combined output in $LogPath.",
        "",
        "Last log lines:",
        (Get-Content $LogPath -Tail 120)
    ) | Set-Content -Path $ErrPath -Encoding UTF8
    throw "Wolfe sweep failed with exit code $PythonExitCode. See $LogPath"
}

Set-Content -Path $ErrPath -Value "" -Encoding UTF8
$ExitCode = $PythonExitCode

"Finished: $(Get-Date -Format o); exit=$ExitCode" | Add-Content -LiteralPath $CommandPath -Encoding utf8
if ($ExitCode -ne 0) {
    Write-Error "Wolfe deep sweep failed with exit code $ExitCode. See $ErrPath"
}

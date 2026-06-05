param(
    [int]$Limit = 100,
    [int]$Workers = 4,
    [string]$RunName = "",
    [string[]]$Symbols = @(),
    [string]$DataDir = "scripts\data_pyharmonics_top100_fast_15m_abcd_xabcd_5y",
    [string]$TopSymbolsPath = "scripts\pyharmonics_top100_fast_15m_abcd_xabcd_5y_20260603_100738\top_symbols.csv",
    [string]$SelectorFeatureSets = "no_symbol",
    [string]$Python = "",
    [switch]$Resume,
    [switch]$SkipGenerate,
    [switch]$SkipSelector,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "pyharmonics_abc_overnight_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

$OutputDir = Join-Path "scripts" $RunName
$ChunkDir = Join-Path $OutputDir "chunks"
$EventCacheSymbolDir = Join-Path $OutputDir "per_symbol"
$MergedDataset = Join-Path $OutputDir "abc_dataset.csv"
$SelectorPrefix = Join-Path $OutputDir "abc_selector"
$SymbolsPath = Join-Path $OutputDir "symbols.txt"
$CommandPath = Join-Path $OutputDir "command.txt"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $ChunkDir | Out-Null
New-Item -ItemType Directory -Force -Path $EventCacheSymbolDir | Out-Null

function Normalize-Symbol {
    param([string]$Raw)
    $Text = ($Raw -as [string]).Trim().ToUpperInvariant()
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ""
    }
    if (-not $Text.EndsWith("USDT")) {
        $Text = "$($Text)USDT"
    }
    return $Text
}

function Get-DataPath {
    param([string]$Symbol)
    return (Join-Path $DataDir ("{0}_15m_bybit.csv" -f $Symbol.ToLowerInvariant()))
}

function Test-ResearchPython {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $false
    }
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    try {
        & $Path -c "import numpy, pandas, sklearn, pyharmonics" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Resolve-ResearchPython {
    $Candidates = New-Object System.Collections.Generic.List[string]
    foreach ($Candidate in @(
        $Python,
        $env:PYHARMONICS_PYTHON,
        $env:PYTHON,
        "C:\Python314\python.exe",
        "C:\Users\flori\AppData\Local\Python\pythoncore-3.14-64\python.exe"
    )) {
        if (-not [string]::IsNullOrWhiteSpace($Candidate)) {
            $Candidates.Add($Candidate)
        }
    }

    foreach ($Candidate in (where.exe python 2>$null)) {
        if (-not [string]::IsNullOrWhiteSpace($Candidate)) {
            $Candidates.Add($Candidate)
        }
    }

    $SeenPython = @{}
    foreach ($Candidate in $Candidates) {
        $Resolved = $Candidate
        try {
            $Resolved = (Resolve-Path -LiteralPath $Candidate -ErrorAction Stop).Path
        } catch {
            $Resolved = $Candidate
        }
        if ($SeenPython.ContainsKey($Resolved)) {
            continue
        }
        $SeenPython[$Resolved] = $true
        if (Test-ResearchPython $Resolved) {
            return $Resolved
        }
    }

    throw "Could not find a Python interpreter with numpy, pandas, sklearn, and pyharmonics. Pass -Python C:\Python314\python.exe or set PYHARMONICS_PYTHON."
}

$SelectedSymbols = New-Object System.Collections.Generic.List[string]
$Seen = @{}

if ($Symbols.Count -gt 0) {
    foreach ($Raw in $Symbols) {
        foreach ($Chunk in ($Raw -split ",")) {
            $Symbol = Normalize-Symbol $Chunk
            if ($Symbol -and -not $Seen.ContainsKey($Symbol)) {
                $Seen[$Symbol] = $true
                $SelectedSymbols.Add($Symbol)
            }
        }
    }
} else {
    if (-not (Test-Path -LiteralPath $TopSymbolsPath)) {
        throw "Missing top-symbol source: $TopSymbolsPath"
    }
    foreach ($Row in (Import-Csv -LiteralPath $TopSymbolsPath)) {
        $Symbol = Normalize-Symbol $Row.symbol
        if (-not $Symbol -or $Seen.ContainsKey($Symbol)) {
            continue
        }
        if (-not (Test-Path -LiteralPath (Get-DataPath $Symbol))) {
            continue
        }
        $Seen[$Symbol] = $true
        $SelectedSymbols.Add($Symbol)
        if ($Limit -gt 0 -and $SelectedSymbols.Count -ge $Limit) {
            break
        }
    }
}

if ($SelectedSymbols.Count -eq 0) {
    throw "No symbols selected. Check -Symbols, -Limit, -TopSymbolsPath, and -DataDir."
}

if ($Workers -lt 1) {
    $Workers = 1
}
$WorkerCount = [Math]::Min($Workers, $SelectedSymbols.Count)

$SelectedSymbols | Set-Content -LiteralPath $SymbolsPath -Encoding utf8

$Header = @(
    "Started: $(Get-Date -Format o)",
    "Repo: $RepoRoot",
    "OutputDir: $OutputDir",
    "ChunkDir: $ChunkDir",
    "EventCacheSymbolDir: $EventCacheSymbolDir",
    "MergedDataset: $MergedDataset",
    "SelectorPrefix: $SelectorPrefix",
    "DataDir: $DataDir",
    "TopSymbolsPath: $TopSymbolsPath",
    "Limit: $Limit",
    "Workers: $Workers",
    "Selected: $($SelectedSymbols.Count)",
    "SelectorFeatureSets: $SelectorFeatureSets",
    "RequestedPython: $Python",
    "Resume: $Resume",
    ""
)
$Header | Set-Content -LiteralPath $CommandPath -Encoding utf8

Write-Host "Prepared clean ABC pyharmonics overnight job"
Write-Host "Output:  $OutputDir"
Write-Host "Symbols: $($SelectedSymbols.Count) -> $SymbolsPath"
Write-Host "Cache:   $EventCacheSymbolDir"
Write-Host ""
Write-Host "Resume with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\run_pyharmonics_abc_overnight.ps1 -RunName $RunName -Resume"
Write-Host ""

$PythonExe = Resolve-ResearchPython
Write-Host "Python:  $PythonExe"
Add-Content -LiteralPath $CommandPath -Encoding utf8 -Value "Python: $PythonExe"

if (-not $SkipGenerate) {
    $Chunks = New-Object object[] $WorkerCount
    for ($Index = 0; $Index -lt $WorkerCount; $Index++) {
        $Chunks[$Index] = @()
    }
    for ($Index = 0; $Index -lt $SelectedSymbols.Count; $Index++) {
        $Bucket = $Index % $WorkerCount
        $Chunks[$Bucket] += $SelectedSymbols[$Index]
    }

    $Processes = @()
    for ($Index = 0; $Index -lt $WorkerCount; $Index++) {
        $ChunkSymbols = @($Chunks[$Index])
        if ($ChunkSymbols.Count -eq 0) {
            continue
        }
        $ChunkName = "chunk_{0:D2}" -f ($Index + 1)
        $ChunkPrefix = Join-Path $ChunkDir $ChunkName
        $ChunkDataset = "$($ChunkPrefix)_dataset.csv"
        $ChunkLog = "$($ChunkPrefix).log"
        $ChunkErr = "$($ChunkPrefix).err"
        $ChunkCommand = "$($ChunkPrefix).command.txt"

        if ($Resume -and (Test-Path -LiteralPath $ChunkDataset)) {
            Write-Host "Skipping $ChunkName; dataset exists: $ChunkDataset"
            continue
        }

        Remove-Item -LiteralPath $ChunkLog, $ChunkErr -ErrorAction SilentlyContinue

        $GeneratorArgs = @(
            "scripts\train_pyharmonics_survivor_ml.py",
            "--dataset-source", "generated",
            "--output-prefix", $ChunkPrefix,
            "--symbols", ($ChunkSymbols -join ","),
            "--data-dir", $DataDir,
            "--families", "ABC",
            "--event-cache-family", "ABC",
            "--event-cache-symbol-dir", $EventCacheSymbolDir,
            "--write-event-cache-symbol-dir", $EventCacheSymbolDir,
            "--pattern-step-bars", "800",
            "--search-limit-to", "6",
            "--peak-spacings", "10,16",
            "--confirm-bars", "4",
            "--entry-window-bars", "4",
            "--entry-modes", "next_open,trigger_break,trigger_close_break",
            "--time-filters", "all,eu_us",
            "--stop-buffers", "0.8,1.1",
            "--min-quality-scores", "0,80",
            "--rrs", "2.0,2.5",
            "--breakeven-triggers", "0",
            "--models", "hgb",
            "--feature-sets", "no_symbol",
            "--skip-symbol-holdout",
            "--min-train-rows", "20",
            "--min-val-rows", "10",
            "--min-val-trades", "2",
            "--min-holdout-test-rows", "1",
            "--progress-every", "24",
            "--event-progress-every", "50"
        )

        @(
            "Started: $(Get-Date -Format o)",
            "Symbols: $($ChunkSymbols -join ',')",
            "",
            "$PythonExe $($GeneratorArgs -join ' ')"
        ) | Set-Content -LiteralPath $ChunkCommand -Encoding utf8

        if ($DryRun) {
            Write-Host "[dry-run] $ChunkName -> $ChunkCommand"
            continue
        }

        Write-Host "Starting $ChunkName symbols=$($ChunkSymbols.Count) log=$ChunkLog"
        $Process = Start-Process -FilePath $PythonExe -ArgumentList $GeneratorArgs -WorkingDirectory $RepoRoot -RedirectStandardOutput $ChunkLog -RedirectStandardError $ChunkErr -WindowStyle Hidden -PassThru
        $Processes += [pscustomobject]@{
            Name = $ChunkName
            Process = $Process
            Log = $ChunkLog
            Err = $ChunkErr
            Dataset = $ChunkDataset
        }
    }

    if (-not $DryRun) {
        while ($true) {
            foreach ($Item in $Processes) {
                $Item.Process.Refresh()
            }
            $Running = $Processes | Where-Object { -not $_.Process.HasExited }
            if ($Running.Count -eq 0) {
                break
            }
            Start-Sleep -Seconds 60
            foreach ($Item in $Processes) {
                $Item.Process.Refresh()
            }
            $Running = $Processes | Where-Object { -not $_.Process.HasExited }
            $Done = $Processes.Count - $Running.Count
            $Status = $Running | ForEach-Object { "$($_.Name):cpu=$([Math]::Round($_.Process.TotalProcessorTime.TotalMinutes, 1))m" }
            Write-Host ("Progress: {0}/{1} chunks done; running {2}" -f $Done, $Processes.Count, ($Status -join " "))
        }

        $Failed = @()
        foreach ($Item in $Processes) {
            $Item.Process.WaitForExit()
            $Item.Process.Refresh()
            $ExitCode = $Item.Process.ExitCode
            $ExitPath = Join-Path $ChunkDir "$($Item.Name).exit.txt"
            "Finished: $(Get-Date -Format o); exit=$ExitCode" | Set-Content -LiteralPath $ExitPath -Encoding utf8
            $DatasetReady = (Test-Path -LiteralPath $Item.Dataset) -and ((Get-Item -LiteralPath $Item.Dataset).Length -gt 0)
            if ((-not $DatasetReady) -or (($null -ne $ExitCode) -and ($ExitCode -ne 0))) {
                $Failed += $Item
            }
        }
        if ($Failed.Count -gt 0) {
            foreach ($Item in $Failed) {
                Write-Host "Failed $($Item.Name), see $($Item.Err)"
            }
            throw "$($Failed.Count) ABC generator chunks failed."
        }
    }
}

if ($DryRun) {
    Write-Host "Dry run complete. Commands were written under $ChunkDir."
    exit 0
}

$DatasetFiles = Get-ChildItem -LiteralPath $ChunkDir -Filter "chunk_*_dataset.csv" | Sort-Object Name
if ($DatasetFiles.Count -eq 0) {
    throw "No chunk datasets found under $ChunkDir"
}

Remove-Item -LiteralPath $MergedDataset -ErrorAction SilentlyContinue
$First = $true
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$Writer = [System.IO.StreamWriter]::new($MergedDataset, $false, $Utf8NoBom)
try {
    foreach ($File in $DatasetFiles) {
        $Reader = [System.IO.StreamReader]::new($File.FullName)
        try {
            $LineNumber = 0
            while ($true) {
                $Line = $Reader.ReadLine()
                if ($null -eq $Line) {
                    break
                }
                if ((-not $First) -and $LineNumber -eq 0) {
                    $LineNumber += 1
                    continue
                }
                $Writer.WriteLine($Line)
                $LineNumber += 1
            }
        } finally {
            $Reader.Close()
        }
        $First = $false
    }
} finally {
    $Writer.Close()
}
Write-Host "Merged $($DatasetFiles.Count) chunk datasets -> $MergedDataset"

if (-not $SkipSelector) {
    $SelectorLog = "$($SelectorPrefix).log"
    $SelectorErr = "$($SelectorPrefix).err"
    $SelectorCommand = "$($SelectorPrefix).command.txt"
    Remove-Item -LiteralPath $SelectorLog, $SelectorErr -ErrorAction SilentlyContinue

    $SelectorArgs = @(
        "scripts\train_pyharmonics_action_selector.py",
        "--dataset", $MergedDataset,
        "--symbols", ($SelectedSymbols -join ","),
        "--families", "ABC",
        "--entry-modes", "next_open,trigger_break,trigger_close_break",
        "--time-filters", "all,eu_us",
        "--peak-spacings", "10,16",
        "--rrs", "2.0,2.5",
        "--stop-buffers", "0.8,1.1",
        "--min-quality-scores", "0,80",
        "--breakeven-triggers", "0",
        "--models", "hgb",
        "--feature-sets", $SelectorFeatureSets,
        "--skip-symbol-holdout",
        "--output-prefix", $SelectorPrefix
    )

    @(
        "Started: $(Get-Date -Format o)",
        "",
        "$PythonExe $($SelectorArgs -join ' ')"
    ) | Set-Content -LiteralPath $SelectorCommand -Encoding utf8

    Write-Host "Starting ABC action selector"
    Write-Host "Selector log: $SelectorLog"
    $SelectorProcess = Start-Process -FilePath $PythonExe -ArgumentList $SelectorArgs -WorkingDirectory $RepoRoot -RedirectStandardOutput $SelectorLog -RedirectStandardError $SelectorErr -WindowStyle Hidden -PassThru
    while (-not $SelectorProcess.HasExited) {
        Start-Sleep -Seconds 60
        $SelectorProcess.Refresh()
        $SelectorCpuMinutes = [Math]::Round($SelectorProcess.TotalProcessorTime.TotalMinutes, 1)
        Write-Host ("Selector running: cpu={0}m log={1}" -f $SelectorCpuMinutes, $SelectorLog)
    }
    $SelectorProcess.WaitForExit()
    $SelectorProcess.Refresh()
    $SelectorExit = $SelectorProcess.ExitCode
    $SelectorSummary = "$($SelectorPrefix)_pooled_summary.csv"
    $SelectorConfig = "$($SelectorPrefix)_config.json"
    $SelectorSummaryReady = (Test-Path -LiteralPath $SelectorSummary) -and ((Get-Item -LiteralPath $SelectorSummary).Length -gt 0)
    $SelectorConfigReady = (Test-Path -LiteralPath $SelectorConfig) -and ((Get-Item -LiteralPath $SelectorConfig).Length -gt 0)
    $SelectorOutputReady = $SelectorSummaryReady -and $SelectorConfigReady
    if (($null -eq $SelectorExit) -and $SelectorOutputReady) {
        $SelectorExit = 0
    }
    Write-Host "Selector finished with exit code $SelectorExit"
    "Finished: $(Get-Date -Format o); exit=$SelectorExit" | Add-Content -LiteralPath $SelectorCommand -Encoding utf8
    if ($SelectorExit -ne 0) {
        throw "ABC selector failed with exit code $SelectorExit. See $SelectorErr"
    }
}

"Finished: $(Get-Date -Format o)" | Add-Content -LiteralPath $CommandPath -Encoding utf8
Write-Host "ABC overnight job finished."

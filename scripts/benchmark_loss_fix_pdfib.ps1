param(
  [string]$StudyId = "gjHA10",
  [string]$Timeframe = "60",
  [string]$EthSymbol = "BINANCE:ETHUSDT",
  [string]$SolSymbol = "BINANCE:SOLUSDT",
  [string]$StudyName = "Crypto Donchian Regime Strategy MajDiv"
)

function Set-Inputs([hashtable]$inputs) {
  $env:TV_INPUTS = ($inputs | ConvertTo-Json -Compress)
  $env:TV_STUDY_ID = $StudyId
  @'
import * as indCore from './src/core/indicators.js';
const inputs = JSON.parse(process.env.TV_INPUTS);
console.log(JSON.stringify(await indCore.setInputs({ entity_id: process.env.TV_STUDY_ID, inputs })));
process.exit(0);
'@ | node --input-type=module - | Out-Null
}

function Get-InputMap() {
  $raw = node src/cli/index.js indicator get $StudyId | ConvertFrom-Json
  $map = @{}
  foreach ($input in @($raw.inputs)) {
    if ($null -ne $input.id) {
      $map[$input.id] = $input.value
    }
  }
  return $map
}

function Wait-Inputs([hashtable]$expectedInputs, [int]$maxAttempts = 8) {
  for ($attempt = 0; $attempt -lt $maxAttempts; $attempt++) {
    $actual = Get-InputMap
    $allMatch = $true
    foreach ($key in $expectedInputs.Keys) {
      $expectedValue = $expectedInputs[$key]
      $actualValue = $actual[$key]
      if (("$expectedValue") -ne ("$actualValue")) {
        $allMatch = $false
        break
      }
    }
    if ($allMatch) { return $true }
    Start-Sleep -Milliseconds 1200
  }
  return $false
}

function Get-Snapshot() {
  $env:TV_STUDY_NAME = $StudyName
  $raw = @'
import { evaluate } from './src/connection.js';
const studyName = JSON.stringify(process.env.TV_STUDY_NAME);
const result = await evaluate(`
  (function() {
    try {
      var targetStudy = ${studyName};
      var chart = window.TradingViewApi._activeChartWidgetWV.value()._chartWidget;
      var sources = chart.model().model().dataSources();
      var strat = null;
      for (var i = 0; i < sources.length; i++) {
        var s = sources[i];
        var meta = null;
        try { meta = s.metaInfo ? s.metaInfo() : null; } catch (e) {}
        var name = meta ? (meta.description || meta.shortDescription || '') : '';
        if (name === targetStudy) { strat = s; break; }
      }
      if (!strat) return { error: 'Strategy not found: ' + targetStudy };
      var rd = strat.reportData ? (typeof strat.reportData === 'function' ? strat.reportData() : strat.reportData) : null;
      if (rd && typeof rd.value === 'function') rd = rd.value();
      if (!rd || !rd.performance || !rd.performance.all) return { error: 'Strategy report unavailable' };
      var all = rd.performance.all;
      var trades = (rd.trades || []).map(function(t) {
        return {
          side: t.e && t.e.c ? t.e.c : '',
          entryTime: t.e && t.e.tm ? t.e.tm : null,
          exitTime: t.x && t.x.tm ? t.x.tm : null,
          profit: t.tp && t.tp.v ? t.tp.v : 0,
          runUp: t.rn && t.rn.v ? t.rn.v : 0,
          drawdown: t.dd && t.dd.v ? t.dd.v : 0
        };
      });
      return {
        metrics: {
          wr: all.percentProfitable != null ? +(all.percentProfitable * 100).toFixed(2) : null,
          pf: all.profitFactor != null ? +(+all.profitFactor).toFixed(3) : null,
          np: all.netProfit != null ? +(+all.netProfit).toFixed(2) : null,
          trades: all.totalTrades != null ? +all.totalTrades : 0
        },
        trades: trades
      };
    } catch (e) {
      return { error: e.message };
    }
  })()
`);
console.log(JSON.stringify(result));
process.exit(0);
'@ | node --input-type=module -
  return $raw | ConvertFrom-Json
}

function Get-Median([double[]]$values) {
  if (-not $values -or $values.Count -eq 0) { return $null }
  $sorted = $values | Sort-Object
  $count = $sorted.Count
  if ($count % 2 -eq 1) {
    return [double]$sorted[[int][math]::Floor($count / 2)]
  }
  $upper = [int]($count / 2)
  $lower = $upper - 1
  return ([double]$sorted[$lower] + [double]$sorted[$upper]) / 2.0
}

function Merge-Inputs([hashtable]$baseInputs, [hashtable]$overrides) {
  $merged = @{}
  foreach ($key in $baseInputs.Keys) { $merged[$key] = $baseInputs[$key] }
  foreach ($key in $overrides.Keys) { $merged[$key] = $overrides[$key] }
  return $merged
}

function New-BaseInputs() {
  @{
    in_32 = "Auto Regime"
    in_38 = 0.20
    in_39 = 0.35
    in_40 = 0.60
    in_41 = 2.0
    in_42 = 58.0
    in_43 = $false
    in_44 = $true
    in_45 = 3
    in_46 = $false
    in_47 = 1.0
    in_60 = $true
    in_61 = 24
    in_62 = 0.95
    in_63 = 0.98
    in_64 = $true
    in_65 = "Reject Only"
    in_66 = "60"
    in_67 = "240"
    in_68 = 2
    in_69 = 2
    in_70 = 0.50
    in_71 = 0.50
    in_72 = $true
    in_73 = 0.21
    in_74 = 0.40
    in_75 = 0.60
    in_76 = 0.79
    in_77 = $true
    in_78 = 2
    in_79 = 2
    in_80 = 4
    in_81 = $false
    in_82 = $false
    in_95 = $false
    in_107 = $false
    in_113 = $false
    in_120 = $false
    in_126 = $false
    in_131 = $false
    in_137 = $false
    in_138 = $false
    in_139 = 0.10
    in_140 = 0.10
    in_141 = $true
    in_142 = "Both"
    in_143 = "Both"
    in_144 = 2.25
    in_145 = 1.75
    in_146 = 97
    in_147 = 3
    in_148 = $false
    in_149 = "Breakout Only"
    in_150 = 3
    in_151 = 0.75
    in_152 = $false
    in_153 = 0.0
  }
}

function Get-SummaryRow([string]$symbol, [string]$configName, $snapshot) {
  $trades = @($snapshot.trades)
  $losses = @($trades | Where-Object { [double]$_.profit -lt 0 })
  $longTrades = @($trades | Where-Object { $_.side -like "Long*" })
  $shortTrades = @($trades | Where-Object { $_.side -like "Short*" })

  $tradeRows = foreach($trade in $trades) {
    $profit = [double]$trade.profit
    $runUp = [double]$trade.runUp
    $lossAbs = if ($profit -lt 0) { [math]::Abs($profit) } else { 0.0 }
    $hoursHeld = if ($trade.entryTime -and $trade.exitTime) { ([DateTimeOffset]::FromUnixTimeMilliseconds([int64]$trade.exitTime).UtcDateTime - [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$trade.entryTime).UtcDateTime).TotalHours } else { 0.0 }
    $feRatio = if ($lossAbs -gt 0) { $runUp / $lossAbs } else { $null }

    [pscustomobject]@{
      Side = if ($trade.side -like "Long*") { "Long" } else { "Short" }
      Profit = $profit
      HoursHeld = [double]$hoursHeld
      FeToLoss = $feRatio
      IsLoss = $profit -lt 0
      ImmediateFail = ($profit -lt 0 -and $lossAbs -gt 0 -and $runUp / $lossAbs -lt 0.25)
      PromisingReverse = ($profit -lt 0 -and $lossAbs -gt 0 -and $runUp / $lossAbs -ge 1.0)
      Fast24h = ($profit -lt 0 -and $hoursHeld -le 24)
      GaveBackHalf = ($profit -lt 0 -and $lossAbs -gt 0 -and $runUp -ge ($lossAbs * 0.5))
    }
  }

  $lossRows = @($tradeRows | Where-Object IsLoss)

  [pscustomobject]@{
    Symbol = $symbol
    Config = $configName
    WR = [double]$snapshot.metrics.wr
    PF = [double]$snapshot.metrics.pf
    NP = [double]$snapshot.metrics.np
    Trades = [int]$snapshot.metrics.trades
    LossRate = if ($trades.Count -gt 0) { [math]::Round(($lossRows.Count / $trades.Count) * 100, 2) } else { 0 }
    LongLossRate = if ($longTrades.Count -gt 0) { [math]::Round((@($longTrades | Where-Object { [double]$_.profit -lt 0 }).Count / $longTrades.Count) * 100, 2) } else { $null }
    ShortLossRate = if ($shortTrades.Count -gt 0) { [math]::Round((@($shortTrades | Where-Object { [double]$_.profit -lt 0 }).Count / $shortTrades.Count) * 100, 2) } else { $null }
    MedianLossHours = if ($lossRows.Count -gt 0) { [math]::Round((Get-Median ([double[]]($lossRows | ForEach-Object { $_.HoursHeld }))), 2) } else { 0 }
    ImmediateFailPct = if ($lossRows.Count -gt 0) { [math]::Round((@($lossRows | Where-Object ImmediateFail).Count / $lossRows.Count) * 100, 2) } else { 0 }
    PromisingReversePct = if ($lossRows.Count -gt 0) { [math]::Round((@($lossRows | Where-Object PromisingReverse).Count / $lossRows.Count) * 100, 2) } else { 0 }
    Fast24hPct = if ($lossRows.Count -gt 0) { [math]::Round((@($lossRows | Where-Object Fast24h).Count / $lossRows.Count) * 100, 2) } else { 0 }
    GiveBackHalfPct = if ($lossRows.Count -gt 0) { [math]::Round((@($lossRows | Where-Object GaveBackHalf).Count / $lossRows.Count) * 100, 2) } else { 0 }
  }
}

function Invoke-ConfigRun([string]$symbol, [string]$configName, [hashtable]$inputs) {
  node src/cli/index.js symbol $symbol | Out-Null
  Start-Sleep -Milliseconds 1800

  Set-Inputs $inputs
  if (-not (Wait-Inputs $inputs)) {
    throw "Inputs did not settle for $symbol / $configName"
  }

  Start-Sleep -Milliseconds 2400
  $snapshot = Get-Snapshot
  if ($snapshot.error) {
    throw "Snapshot failed for ${symbol} / ${configName}: $($snapshot.error)"
  }

  return Get-SummaryRow $symbol $configName $snapshot
}

function Get-JointRows($configs, [hashtable]$baseInputs) {
  $jointRows = @()
  $perSymbolRows = @()

  foreach ($config in $configs) {
    $inputs = Merge-Inputs $baseInputs $config.Overrides

    $ethRow = Invoke-ConfigRun $EthSymbol $config.Name $inputs
    Write-Host ("ETH | {0} | WR {1}% | PF {2} | NP {3} | Trades {4}" -f $config.Name, $ethRow.WR, $ethRow.PF, $ethRow.NP, $ethRow.Trades)
    $perSymbolRows += $ethRow

    $solRow = Invoke-ConfigRun $SolSymbol $config.Name $inputs
    Write-Host ("SOL | {0} | WR {1}% | PF {2} | NP {3} | Trades {4}" -f $config.Name, $solRow.WR, $solRow.PF, $solRow.NP, $solRow.Trades)
    $perSymbolRows += $solRow

    $jointRows += [pscustomobject]@{
      Config = $config.Name
      Overrides = $config.Overrides
      EthWR = $ethRow.WR
      EthPF = $ethRow.PF
      EthNP = $ethRow.NP
      EthTrades = $ethRow.Trades
      EthImmediateFailPct = $ethRow.ImmediateFailPct
      EthPromisingReversePct = $ethRow.PromisingReversePct
      SolWR = $solRow.WR
      SolPF = $solRow.PF
      SolNP = $solRow.NP
      SolTrades = $solRow.Trades
      SolImmediateFailPct = $solRow.ImmediateFailPct
      SolPromisingReversePct = $solRow.PromisingReversePct
      MinPF = [math]::Round([math]::Min([double]$ethRow.PF, [double]$solRow.PF), 3)
      AvgPF = [math]::Round((([double]$ethRow.PF + [double]$solRow.PF) / 2.0), 3)
      TotalNP = [math]::Round(([double]$ethRow.NP + [double]$solRow.NP), 2)
      TotalTrades = [int]$ethRow.Trades + [int]$solRow.Trades
    }
  }

  return [pscustomobject]@{
    Joint = $jointRows
    PerSymbol = $perSymbolRows
  }
}

function Select-RobustConfig($jointRows, [int]$ethMinTrades, [int]$solMinTrades) {
  $eligible = @($jointRows | Where-Object { $_.EthTrades -ge $ethMinTrades -and $_.SolTrades -ge $solMinTrades })
  if ($eligible.Count -eq 0) { $eligible = @($jointRows) }
  return $eligible |
    Sort-Object @{Expression = "MinPF"; Descending = $true}, @{Expression = "AvgPF"; Descending = $true}, @{Expression = "TotalNP"; Descending = $true}, @{Expression = "TotalTrades"; Descending = $true} |
    Select-Object -First 1
}

$baseInputs = New-BaseInputs

$lossFixConfigs = @(
  @{ Name = "baseline"; Overrides = @{} },
  @{ Name = "excess_010"; Overrides = @{ in_138 = $true; in_139 = 0.10; in_140 = 0.10; in_141 = $false } },
  @{ Name = "excess_short015"; Overrides = @{ in_138 = $true; in_139 = 0.10; in_140 = 0.15; in_141 = $false } },
  @{ Name = "excess_015"; Overrides = @{ in_138 = $true; in_139 = 0.15; in_140 = 0.15; in_141 = $false } },
  @{ Name = "stretch_only"; Overrides = @{ in_138 = $true; in_139 = 0.0; in_140 = 0.0; in_141 = $true; in_142 = "Both"; in_143 = "Both"; in_144 = 2.25; in_145 = 1.75; in_146 = 97; in_147 = 3 } },
  @{ Name = "stretch_short_tight"; Overrides = @{ in_138 = $true; in_139 = 0.0; in_140 = 0.0; in_141 = $true; in_142 = "Both"; in_143 = "Both"; in_144 = 2.25; in_145 = 1.50; in_146 = 97; in_147 = 5 } },
  @{ Name = "quality_combo"; Overrides = @{ in_138 = $true; in_139 = 0.10; in_140 = 0.15; in_141 = $true; in_142 = "Both"; in_143 = "Both"; in_144 = 2.25; in_145 = 1.75; in_146 = 97; in_147 = 3 } },
  @{ Name = "early_mfe_075"; Overrides = @{ in_148 = $true; in_149 = "Breakout Only"; in_150 = 3; in_151 = 0.75; in_152 = $false } },
  @{ Name = "early_mfe_100"; Overrides = @{ in_148 = $true; in_149 = "Breakout Only"; in_150 = 3; in_151 = 1.00; in_152 = $false } },
  @{ Name = "quality_plus_early"; Overrides = @{ in_138 = $true; in_139 = 0.10; in_140 = 0.15; in_141 = $true; in_142 = "Both"; in_143 = "Both"; in_144 = 2.25; in_145 = 1.75; in_146 = 97; in_147 = 3; in_148 = $true; in_149 = "Breakout Only"; in_150 = 3; in_151 = 1.00; in_152 = $false } },
  @{ Name = "quality_early_adverse"; Overrides = @{ in_138 = $true; in_139 = 0.10; in_140 = 0.15; in_141 = $true; in_142 = "Both"; in_143 = "Both"; in_144 = 2.25; in_145 = 1.75; in_146 = 97; in_147 = 3; in_148 = $true; in_149 = "Breakout Only"; in_150 = 3; in_151 = 1.00; in_152 = $true; in_153 = 0.0 } }
)

node src/cli/index.js timeframe $Timeframe | Out-Null
Start-Sleep -Milliseconds 1200

Write-Host ""
Write-Host "=== Phase 1: Loss-fix sweep ==="
$phase1 = Get-JointRows $lossFixConfigs $baseInputs

$lossJoint = @($phase1.Joint)
$lossBaseline = $lossJoint | Where-Object { $_.Config -eq "baseline" } | Select-Object -First 1
$ethMinTrades = [math]::Max(20, [math]::Floor($lossBaseline.EthTrades * 0.60))
$solMinTrades = [math]::Max(20, [math]::Floor($lossBaseline.SolTrades * 0.60))
$bestLossFix = Select-RobustConfig $lossJoint $ethMinTrades $solMinTrades

Write-Host ""
Write-Host "Loss-fix joint ranking:"
$lossJoint |
  Sort-Object @{Expression = "MinPF"; Descending = $true}, @{Expression = "AvgPF"; Descending = $true}, @{Expression = "TotalNP"; Descending = $true} |
  Format-Table Config, EthPF, EthNP, EthTrades, SolPF, SolNP, SolTrades, MinPF, AvgPF, TotalNP -AutoSize

Write-Host ""
Write-Host ("Selected robust loss-fix base: {0} (trade floors ETH {1}, SOL {2})" -f $bestLossFix.Config, $ethMinTrades, $solMinTrades)

$selectedLossFixConfig = $lossFixConfigs | Where-Object { $_.Name -eq $bestLossFix.Config } | Select-Object -First 1
$lossFixBaseInputs = Merge-Inputs $baseInputs $selectedLossFixConfig.Overrides

$pdConfigs = @(
  @{ Name = "pd_current"; Overrides = @{} },
  @{ Name = "pd_reject_strict_240D"; Overrides = @{ in_64 = $true; in_65 = "Reject Only"; in_66 = "240"; in_67 = "D"; in_70 = 0.40; in_71 = 0.60; in_73 = 0.10; in_74 = 0.32; in_75 = 0.68; in_76 = 0.90; in_77 = $true; in_78 = 2; in_79 = 2; in_80 = 4 } },
  @{ Name = "pd_breakout_mild_60_240"; Overrides = @{ in_64 = $true; in_65 = "Breakout Only"; in_66 = "60"; in_67 = "240"; in_70 = 0.45; in_71 = 0.55; in_73 = 0.15; in_74 = 0.35; in_75 = 0.65; in_76 = 0.85; in_77 = $true; in_78 = 2; in_79 = 1; in_80 = 3 } },
  @{ Name = "pd_breakout_strict_60_240"; Overrides = @{ in_64 = $true; in_65 = "Breakout Only"; in_66 = "60"; in_67 = "240"; in_70 = 0.40; in_71 = 0.60; in_73 = 0.10; in_74 = 0.30; in_75 = 0.70; in_76 = 0.90; in_77 = $true; in_78 = 2; in_79 = 2; in_80 = 4 } },
  @{ Name = "pd_breakout_mild_60_D"; Overrides = @{ in_64 = $true; in_65 = "Breakout Only"; in_66 = "60"; in_67 = "D"; in_70 = 0.45; in_71 = 0.55; in_73 = 0.15; in_74 = 0.35; in_75 = 0.65; in_76 = 0.85; in_77 = $true; in_78 = 2; in_79 = 1; in_80 = 3 } },
  @{ Name = "pd_breakout_strict_60_D"; Overrides = @{ in_64 = $true; in_65 = "Breakout Only"; in_66 = "60"; in_67 = "D"; in_70 = 0.40; in_71 = 0.60; in_73 = 0.10; in_74 = 0.30; in_75 = 0.70; in_76 = 0.90; in_77 = $true; in_78 = 2; in_79 = 2; in_80 = 4 } },
  @{ Name = "pd_breakout_mild_240_D"; Overrides = @{ in_64 = $true; in_65 = "Breakout Only"; in_66 = "240"; in_67 = "D"; in_70 = 0.45; in_71 = 0.55; in_73 = 0.15; in_74 = 0.35; in_75 = 0.65; in_76 = 0.85; in_77 = $true; in_78 = 2; in_79 = 1; in_80 = 3 } },
  @{ Name = "pd_breakout_strict_240_D"; Overrides = @{ in_64 = $true; in_65 = "Breakout Only"; in_66 = "240"; in_67 = "D"; in_70 = 0.40; in_71 = 0.60; in_73 = 0.10; in_74 = 0.30; in_75 = 0.70; in_76 = 0.90; in_77 = $true; in_78 = 2; in_79 = 2; in_80 = 4 } },
  @{ Name = "pd_breakout_price_only_240_D"; Overrides = @{ in_64 = $true; in_65 = "Breakout Only"; in_66 = "240"; in_67 = "D"; in_70 = 0.42; in_71 = 0.58; in_73 = 0.12; in_74 = 0.32; in_75 = 0.68; in_76 = 0.88; in_77 = $false; in_79 = 1; in_80 = 3 } },
  @{ Name = "pd_all_mild_240_D"; Overrides = @{ in_64 = $true; in_65 = "All Signals"; in_66 = "240"; in_67 = "D"; in_70 = 0.45; in_71 = 0.55; in_73 = 0.15; in_74 = 0.35; in_75 = 0.65; in_76 = 0.85; in_77 = $true; in_78 = 2; in_79 = 1; in_80 = 3 } }
)

Write-Host ""
Write-Host "=== Phase 2: PD/Fib extremeness sweep ==="
$phase2 = Get-JointRows $pdConfigs $lossFixBaseInputs
$pdJoint = @($phase2.Joint)
$bestPd = Select-RobustConfig $pdJoint $ethMinTrades $solMinTrades

Write-Host ""
Write-Host "PD/Fib joint ranking:"
$pdJoint |
  Sort-Object @{Expression = "MinPF"; Descending = $true}, @{Expression = "AvgPF"; Descending = $true}, @{Expression = "TotalNP"; Descending = $true} |
  Format-Table Config, EthPF, EthNP, EthTrades, SolPF, SolNP, SolTrades, MinPF, AvgPF, TotalNP -AutoSize

Write-Host ""
Write-Host ("Selected robust PD/Fib profile: {0}" -f $bestPd.Config)

$selectedPdConfig = $pdConfigs | Where-Object { $_.Name -eq $bestPd.Config } | Select-Object -First 1
$finalInputs = Merge-Inputs $lossFixBaseInputs $selectedPdConfig.Overrides

node src/cli/index.js symbol $EthSymbol | Out-Null
Start-Sleep -Milliseconds 1800
Set-Inputs $finalInputs
if (-not (Wait-Inputs $finalInputs)) {
  throw "Final ETH inputs did not settle"
}
Start-Sleep -Milliseconds 2400
$finalSnapshot = Get-Snapshot
if ($finalSnapshot.error) {
  throw "Final ETH snapshot failed: $($finalSnapshot.error)"
}

Write-Host ""
Write-Host "Final ETH snapshot:"
([pscustomobject]@{
  Config = "$($bestLossFix.Config) + $($bestPd.Config)"
  WR = $finalSnapshot.metrics.wr
  PF = $finalSnapshot.metrics.pf
  NP = $finalSnapshot.metrics.np
  Trades = $finalSnapshot.metrics.trades
} | Format-Table -AutoSize | Out-String)

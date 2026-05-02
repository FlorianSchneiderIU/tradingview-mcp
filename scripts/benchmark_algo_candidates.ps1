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

$baseInputs = New-BaseInputs
$configs = @(
  @{ Name = "baseline"; Overrides = @{} },
  @{ Name = "exhaust_70_30"; Overrides = @{ in_120 = $true; in_121 = "Breakout Only"; in_122 = 22; in_123 = 70; in_124 = 30; in_125 = 4 } },
  @{ Name = "exhaust_80_20"; Overrides = @{ in_120 = $true; in_121 = "Breakout Only"; in_122 = 22; in_123 = 80; in_124 = 20; in_125 = 4 } },
  @{ Name = "vwap_1p5"; Overrides = @{ in_126 = $true; in_127 = "Breakout Only"; in_128 = 1.5; in_129 = 1.5 } },
  @{ Name = "vwap_2p0"; Overrides = @{ in_126 = $true; in_127 = "Breakout Only"; in_128 = 2.0; in_129 = 2.0 } },
  @{ Name = "bbpct_95_5"; Overrides = @{ in_131 = $true; in_132 = "Breakout Only"; in_133 = 20; in_134 = 2.0; in_135 = 95; in_136 = 5 } },
  @{ Name = "bbpct_90_10"; Overrides = @{ in_131 = $true; in_132 = "Breakout Only"; in_133 = 20; in_134 = 2.0; in_135 = 90; in_136 = 10 } },
  @{ Name = "exhaust_vwap"; Overrides = @{ in_120 = $true; in_121 = "Breakout Only"; in_122 = 22; in_123 = 70; in_124 = 30; in_125 = 4; in_126 = $true; in_127 = "Breakout Only"; in_128 = 2.0; in_129 = 2.0 } },
  @{ Name = "vwap_bbpct"; Overrides = @{ in_126 = $true; in_127 = "Breakout Only"; in_128 = 2.0; in_129 = 2.0; in_131 = $true; in_132 = "Breakout Only"; in_133 = 20; in_134 = 2.0; in_135 = 95; in_136 = 5 } },
  @{ Name = "all_three"; Overrides = @{ in_120 = $true; in_121 = "Breakout Only"; in_122 = 22; in_123 = 70; in_124 = 30; in_125 = 4; in_126 = $true; in_127 = "Breakout Only"; in_128 = 2.0; in_129 = 2.0; in_131 = $true; in_132 = "Breakout Only"; in_133 = 20; in_134 = 2.0; in_135 = 95; in_136 = 5 } }
)

node src/cli/index.js timeframe $Timeframe | Out-Null
Start-Sleep -Milliseconds 1200

node src/cli/index.js symbol $EthSymbol | Out-Null
Start-Sleep -Milliseconds 1800

$ethResults = @()
foreach($config in $configs) {
  $inputs = Merge-Inputs $baseInputs $config.Overrides
  Set-Inputs $inputs
  if (-not (Wait-Inputs $inputs)) { throw "ETH inputs did not settle for $($config.Name)" }
  Start-Sleep -Milliseconds 2200
  $snapshot = Get-Snapshot
  if ($snapshot.error) { throw "ETH snapshot failed for $($config.Name): $($snapshot.error)" }
  $row = Get-SummaryRow $EthSymbol $config.Name $snapshot
  $ethResults += $row
  Write-Host ("ETH | {0} | WR {1}% | PF {2} | NP {3} | Trades {4} | ImmediateFail {5}% | PromisingReverse {6}%" -f $row.Config, $row.WR, $row.PF, $row.NP, $row.Trades, $row.ImmediateFailPct, $row.PromisingReversePct)
}

$baseline = $ethResults | Where-Object { $_.Config -eq "baseline" } | Select-Object -First 1
$minTradeThreshold = [math]::Max(20, [math]::Floor($baseline.Trades * 0.60))
$eligible = $ethResults | Where-Object { $_.Trades -ge $minTradeThreshold }
$bestEth = $eligible | Sort-Object @{Expression = "PF"; Descending = $true}, @{Expression = "NP"; Descending = $true}, @{Expression = "WR"; Descending = $true}, @{Expression = "Trades"; Descending = $true} | Select-Object -First 1

Write-Host ""
Write-Host "ETH ranking:"
$ethResults |
  Sort-Object @{Expression = "PF"; Descending = $true}, @{Expression = "NP"; Descending = $true}, @{Expression = "Trades"; Descending = $true} |
  Format-Table Config, WR, PF, NP, Trades, LossRate, ImmediateFailPct, PromisingReversePct, Fast24hPct, GiveBackHalfPct -AutoSize

Write-Host ""
Write-Host ("Selected ETH pattern: {0} (minimum trade threshold {1})" -f $bestEth.Config, $minTradeThreshold)

node src/cli/index.js symbol $SolSymbol | Out-Null
Start-Sleep -Milliseconds 1800

$solResults = @()
foreach($configName in @("baseline", $bestEth.Config)) {
  $config = $configs | Where-Object { $_.Name -eq $configName } | Select-Object -First 1
  $inputs = Merge-Inputs $baseInputs $config.Overrides
  Set-Inputs $inputs
  if (-not (Wait-Inputs $inputs)) { throw "SOL inputs did not settle for $($configName)" }
  Start-Sleep -Milliseconds 2200
  $snapshot = Get-Snapshot
  if ($snapshot.error) { throw "SOL snapshot failed for $($configName): $($snapshot.error)" }
  $row = Get-SummaryRow $SolSymbol $configName $snapshot
  $solResults += $row
  Write-Host ("SOL | {0} | WR {1}% | PF {2} | NP {3} | Trades {4} | ImmediateFail {5}% | PromisingReverse {6}%" -f $row.Config, $row.WR, $row.PF, $row.NP, $row.Trades, $row.ImmediateFailPct, $row.PromisingReversePct)
}

Write-Host ""
Write-Host "SOL out-of-sample comparison:"
$solResults | Format-Table Config, WR, PF, NP, Trades, LossRate, ImmediateFailPct, PromisingReversePct, Fast24hPct, GiveBackHalfPct -AutoSize

# Restore chart to ETH baseline.
node src/cli/index.js symbol $EthSymbol | Out-Null
Start-Sleep -Milliseconds 1800
Set-Inputs $baseInputs
Wait-Inputs $baseInputs | Out-Null
Start-Sleep -Milliseconds 1500

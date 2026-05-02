param(
  [string]$StudyId = "ZHUZNe",
  [string]$Timeframe = "60",
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
      return {
        wr: all.percentProfitable != null ? +(all.percentProfitable * 100).toFixed(2) : null,
        pf: all.profitFactor != null ? +(+all.profitFactor).toFixed(3) : null,
        np: all.netProfit != null ? +(+all.netProfit).toFixed(2) : null,
        trades: all.totalTrades != null ? +all.totalTrades : 0,
        avgWin: all.avgWinningTrade != null ? +(+all.avgWinningTrade).toFixed(2) : null,
        avgLoss: all.avgLosingTrade != null ? +(+all.avgLosingTrade).toFixed(2) : null
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

$baseInputs = @{
  in_24 = ""
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
  in_48 = "Immediate"
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
  in_148 = $false
  in_154 = $false
}

$benchmarkInputs = $baseInputs.Clone()
$benchmarkInputs["in_24"] = "BINANCE:ETHBTC"

$symbols = @(
  "OANDA:XAUUSD",
  "OANDA:SPX500USD",
  "OANDA:NAS100USD"
)

node src/cli/index.js timeframe $Timeframe | Out-Null
Start-Sleep -Milliseconds 1200

Set-Inputs $benchmarkInputs
if (-not (Wait-Inputs $benchmarkInputs)) {
  throw "Benchmark inputs did not settle"
}

$results = @()
foreach ($symbol in $symbols) {
  node src/cli/index.js symbol $symbol | Out-Null
  Start-Sleep -Milliseconds 4500
  $snapshot = Get-Snapshot
  if ($snapshot.error) {
    throw "Snapshot failed for ${symbol}: $($snapshot.error)"
  }
  $results += [pscustomobject]@{
    Symbol = $symbol
    WR = $snapshot.wr
    PF = $snapshot.pf
    NP = $snapshot.np
    Trades = $snapshot.trades
    AvgWin = $snapshot.avgWin
    AvgLoss = $snapshot.avgLoss
    AvgWinLoss = if ($snapshot.avgLoss -and $snapshot.avgLoss -ne 0) { [math]::Round([math]::Abs($snapshot.avgWin / $snapshot.avgLoss), 3) } else { $null }
  }
  Write-Host ("{0} | WR {1}% | PF {2} | NP {3} | Trades {4}" -f $symbol, $snapshot.wr, $snapshot.pf, $snapshot.np, $snapshot.trades)
}

node src/cli/index.js symbol BINANCE:ETHUSDT | Out-Null
Start-Sleep -Milliseconds 1800
Set-Inputs $baseInputs
Wait-Inputs $baseInputs | Out-Null

Write-Host ""
$results | Format-Table Symbol, WR, PF, NP, Trades, AvgWin, AvgLoss, AvgWinLoss -AutoSize

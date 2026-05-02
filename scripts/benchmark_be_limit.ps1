param(
  [string]$StudyId = "ZHUZNe",
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
    in_155 = "Breakout Only"
    in_156 = 1.0
    in_157 = 0.0
    in_158 = $false
    in_159 = 2.0
    in_160 = 0.5
    in_161 = 0.35
    in_162 = 3
  }
}

function Get-SummaryRow([string]$symbol, [string]$configName, $snapshot) {
  $trades = @($snapshot.trades)
  $losses = @($trades | Where-Object { [double]$_.profit -lt 0 })
  $wins = @($trades | Where-Object { [double]$_.profit -gt 0 })
  [pscustomobject]@{
    Symbol = $symbol
    Config = $configName
    WR = [double]$snapshot.metrics.wr
    PF = [double]$snapshot.metrics.pf
    NP = [double]$snapshot.metrics.np
    Trades = [int]$snapshot.metrics.trades
    AvgLossRunup = if ($losses.Count -gt 0) { [math]::Round((($losses | Measure-Object -Property runUp -Average).Average), 2) } else { 0 }
    AvgWinRunup = if ($wins.Count -gt 0) { [math]::Round((($wins | Measure-Object -Property runUp -Average).Average), 2) } else { 0 }
  }
}

function Invoke-ConfigRun([string]$symbol, [string]$configName, [hashtable]$inputs) {
  node src/cli/index.js symbol $symbol | Out-Null
  Start-Sleep -Milliseconds 1800
  Set-Inputs $inputs
  if (-not (Wait-Inputs $inputs)) { throw "Inputs did not settle for ${symbol} / ${configName}" }
  Start-Sleep -Milliseconds 2400
  $snapshot = Get-Snapshot
  if ($snapshot.error) { throw "Snapshot failed for ${symbol} / ${configName}: $($snapshot.error)" }
  return Get-SummaryRow $symbol $configName $snapshot
}

function Get-JointRows($configs, [hashtable]$baseInputs) {
  $jointRows = @()
  foreach ($config in $configs) {
    $inputs = Merge-Inputs $baseInputs $config.Overrides
    $ethRow = Invoke-ConfigRun $EthSymbol $config.Name $inputs
    Write-Host ("ETH | {0} | WR {1}% | PF {2} | NP {3} | Trades {4}" -f $config.Name, $ethRow.WR, $ethRow.PF, $ethRow.NP, $ethRow.Trades)
    $solRow = Invoke-ConfigRun $SolSymbol $config.Name $inputs
    Write-Host ("SOL | {0} | WR {1}% | PF {2} | NP {3} | Trades {4}" -f $config.Name, $solRow.WR, $solRow.PF, $solRow.NP, $solRow.Trades)
    $jointRows += [pscustomobject]@{
      Config = $config.Name
      Overrides = $config.Overrides
      EthWR = $ethRow.WR
      EthPF = $ethRow.PF
      EthNP = $ethRow.NP
      EthTrades = $ethRow.Trades
      SolWR = $solRow.WR
      SolPF = $solRow.PF
      SolNP = $solRow.NP
      SolTrades = $solRow.Trades
      MinPF = [math]::Round([math]::Min([double]$ethRow.PF, [double]$solRow.PF), 3)
      AvgPF = [math]::Round((([double]$ethRow.PF + [double]$solRow.PF) / 2.0), 3)
      TotalNP = [math]::Round(([double]$ethRow.NP + [double]$solRow.NP), 2)
      TotalTrades = [int]$ethRow.Trades + [int]$solRow.Trades
    }
  }
  return $jointRows
}

function Select-RobustConfig($jointRows, [int]$ethMinTrades, [int]$solMinTrades) {
  $eligible = @($jointRows | Where-Object { $_.EthTrades -ge $ethMinTrades -and $_.SolTrades -ge $solMinTrades })
  if ($eligible.Count -eq 0) { $eligible = @($jointRows) }
  return $eligible |
    Sort-Object @{Expression = "MinPF"; Descending = $true}, @{Expression = "AvgPF"; Descending = $true}, @{Expression = "TotalNP"; Descending = $true}, @{Expression = "TotalTrades"; Descending = $true} |
    Select-Object -First 1
}

$baseInputs = New-BaseInputs
$configs = @(
  @{ Name = "baseline"; Overrides = @{} },
  @{ Name = "be_1r"; Overrides = @{ in_154 = $true; in_156 = 1.0; in_157 = 0.0; in_158 = $false } },
  @{ Name = "be_1p5r"; Overrides = @{ in_154 = $true; in_156 = 1.5; in_157 = 0.0; in_158 = $false } },
  @{ Name = "be_offset_025"; Overrides = @{ in_154 = $true; in_156 = 1.0; in_157 = 0.25; in_158 = $false } },
  @{ Name = "lock_1_2_05"; Overrides = @{ in_154 = $true; in_156 = 1.0; in_157 = 0.0; in_158 = $true; in_159 = 2.0; in_160 = 0.50 } },
  @{ Name = "lock_1_1p5_025"; Overrides = @{ in_154 = $true; in_156 = 1.0; in_157 = 0.0; in_158 = $true; in_159 = 1.5; in_160 = 0.25 } },
  @{ Name = "limit_025"; Overrides = @{ in_48 = "Pullback Limit"; in_161 = 0.25; in_162 = 3 } },
  @{ Name = "limit_035"; Overrides = @{ in_48 = "Pullback Limit"; in_161 = 0.35; in_162 = 3 } },
  @{ Name = "limit_050"; Overrides = @{ in_48 = "Pullback Limit"; in_161 = 0.50; in_162 = 3 } },
  @{ Name = "be_limit_025"; Overrides = @{ in_154 = $true; in_156 = 1.0; in_157 = 0.0; in_158 = $true; in_159 = 1.5; in_160 = 0.25; in_48 = "Pullback Limit"; in_161 = 0.25; in_162 = 3 } },
  @{ Name = "be_limit_035"; Overrides = @{ in_154 = $true; in_156 = 1.0; in_157 = 0.0; in_158 = $true; in_159 = 1.5; in_160 = 0.25; in_48 = "Pullback Limit"; in_161 = 0.35; in_162 = 3 } }
)

node src/cli/index.js timeframe $Timeframe | Out-Null
Start-Sleep -Milliseconds 1200

$jointRows = Get-JointRows $configs $baseInputs
$baseline = $jointRows | Where-Object { $_.Config -eq "baseline" } | Select-Object -First 1
$ethMinTrades = [math]::Max(20, [math]::Floor($baseline.EthTrades * 0.60))
$solMinTrades = [math]::Max(20, [math]::Floor($baseline.SolTrades * 0.60))
$best = Select-RobustConfig $jointRows $ethMinTrades $solMinTrades

Write-Host ""
Write-Host "Joint ranking:"
$jointRows |
  Sort-Object @{Expression = "MinPF"; Descending = $true}, @{Expression = "AvgPF"; Descending = $true}, @{Expression = "TotalNP"; Descending = $true} |
  Format-Table Config, EthPF, EthNP, EthTrades, SolPF, SolNP, SolTrades, MinPF, AvgPF, TotalNP -AutoSize

Write-Host ""
Write-Host ("Selected robust config: {0} (trade floors ETH {1}, SOL {2})" -f $best.Config, $ethMinTrades, $solMinTrades)

$selectedConfig = $configs | Where-Object { $_.Name -eq $best.Config } | Select-Object -First 1
$finalInputs = Merge-Inputs $baseInputs $selectedConfig.Overrides

node src/cli/index.js symbol $EthSymbol | Out-Null
Start-Sleep -Milliseconds 1800
Set-Inputs $finalInputs
if (-not (Wait-Inputs $finalInputs)) { throw "Final ETH inputs did not settle" }
Start-Sleep -Milliseconds 2400
$finalSnapshot = Get-Snapshot
if ($finalSnapshot.error) { throw "Final ETH snapshot failed: $($finalSnapshot.error)" }

Write-Host ""
Write-Host "Final ETH snapshot:"
([pscustomobject]@{
  Config = $best.Config
  WR = $finalSnapshot.metrics.wr
  PF = $finalSnapshot.metrics.pf
  NP = $finalSnapshot.metrics.np
  Trades = $finalSnapshot.metrics.trades
} | Format-Table -AutoSize | Out-String)

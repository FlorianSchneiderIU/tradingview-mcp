param(
  [string[]]$Symbols = @("BINANCE:ETHUSDT", "BINANCE:BTCUSDT", "BINANCE:SOLUSDT", "BINANCE:BNBUSDT", "BINANCE:XRPUSDT", "BINANCE:LINKUSDT"),
  [string]$Timeframe = "60",
  [string]$StudyId = "cOAOfw"
)

function Set-Inputs([hashtable]$inputs) {
  $env:TV_INPUTS = ($inputs | ConvertTo-Json -Compress)
  $env:TV_STUDY_ID = $StudyId
  @'
import * as indCore from './src/core/indicators.js';
const inputs = JSON.parse(process.env.TV_INPUTS);
await indCore.setInputs({ entity_id: process.env.TV_STUDY_ID, inputs });
process.exit(0);
'@ | node --input-type=module - | Out-Null
}

function Get-Metrics() {
  $j = node src/cli/index.js data strategy | ConvertFrom-Json
  $all = $j.metrics.performance.all
  $long = $j.metrics.performance.long
  $short = $j.metrics.performance.short
  [pscustomobject]@{
    WR = [math]::Round([double]$all.percentProfitable * 100, 2)
    PF = [math]::Round([double]$all.profitFactor, 3)
    NP = [math]::Round([double]$all.netProfit, 2)
    Trades = [int]$all.totalTrades
    LongPF = if ($null -ne $long) { [math]::Round([double]$long.profitFactor, 3) } else { $null }
    LongNP = if ($null -ne $long) { [math]::Round([double]$long.netProfit, 2) } else { $null }
    ShortPF = if ($null -ne $short) { [math]::Round([double]$short.profitFactor, 3) } else { $null }
    ShortNP = if ($null -ne $short) { [math]::Round([double]$short.netProfit, 2) } else { $null }
  }
}

$profiles = @(
  @{
    Name = "baseline"
    Inputs = @{
      in_32 = "Off"
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
    }
  },
  @{
    Name = "combine"
    Inputs = @{
      in_32 = "Combine"
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
    }
  },
  @{
    Name = "auto_regime"
    Inputs = @{
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
    }
  }
)

$results = @()

node src/cli/index.js timeframe $Timeframe | Out-Null
Start-Sleep -Milliseconds 1200

foreach($symbol in $Symbols) {
  node src/cli/index.js symbol $symbol | Out-Null
  Start-Sleep -Milliseconds 1800

  foreach($profile in $profiles) {
    Set-Inputs $profile.Inputs
    Start-Sleep -Milliseconds 1200
    $m = Get-Metrics
    $row = [pscustomobject]@{
      Symbol = $symbol
      Profile = $profile.Name
      WR = $m.WR
      PF = $m.PF
      NP = $m.NP
      Trades = $m.Trades
      LongPF = $m.LongPF
      LongNP = $m.LongNP
      ShortPF = $m.ShortPF
      ShortNP = $m.ShortNP
    }
    $results += $row
    Write-Host ("{0} | {1} | WR {2}% | PF {3} | NP {4} | Trades {5}" -f $symbol, $profile.Name, $m.WR, $m.PF, $m.NP, $m.Trades)
  }
}

Write-Host ""
Write-Host "Best profile per symbol:"
$results |
  Group-Object Symbol |
  ForEach-Object {
    $_.Group | Sort-Object @{Expression = "PF"; Descending = $true}, @{Expression = "NP"; Descending = $true} | Select-Object -First 1
  } |
  Format-Table -AutoSize

Write-Host ""
Write-Host "Basket summary:"
$results |
  Group-Object Profile |
  ForEach-Object {
    $group = $_.Group
    [pscustomobject]@{
      Profile = $_.Name
      ProfitableSymbols = ($group | Where-Object { $_.NP -gt 0 }).Count
      AvgPF = [math]::Round((($group | Measure-Object PF -Average).Average), 3)
      TotalNP = [math]::Round((($group | Measure-Object NP -Sum).Sum), 2)
      AvgTrades = [math]::Round((($group | Measure-Object Trades -Average).Average), 2)
    }
  } |
  Sort-Object @{Expression = "AvgPF"; Descending = $true}, @{Expression = "TotalNP"; Descending = $true} |
  Format-Table -AutoSize

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
  [pscustomobject]@{
    WR = [math]::Round([double]$all.percentProfitable * 100, 2)
    PF = [math]::Round([double]$all.profitFactor, 3)
    NP = [math]::Round([double]$all.netProfit, 2)
    Trades = [int]$all.totalTrades
  }
}

function New-AutoInputs([int]$entropyLen, [double]$breakoutMax, [double]$rejectMin) {
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
    in_61 = $entropyLen
    in_62 = $breakoutMax
    in_63 = $rejectMin
  }
}

$configs = @(
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
  @{ Name = "auto_e16_b093_r097"; Inputs = (New-AutoInputs 16 0.93 0.97) },
  @{ Name = "auto_e16_b095_r098"; Inputs = (New-AutoInputs 16 0.95 0.98) },
  @{ Name = "auto_e16_b097_r099"; Inputs = (New-AutoInputs 16 0.97 0.99) },
  @{ Name = "auto_e24_b093_r097"; Inputs = (New-AutoInputs 24 0.93 0.97) },
  @{ Name = "auto_e24_b095_r098"; Inputs = (New-AutoInputs 24 0.95 0.98) },
  @{ Name = "auto_e24_b097_r099"; Inputs = (New-AutoInputs 24 0.97 0.99) },
  @{ Name = "auto_e32_b093_r097"; Inputs = (New-AutoInputs 32 0.93 0.97) },
  @{ Name = "auto_e32_b095_r098"; Inputs = (New-AutoInputs 32 0.95 0.98) },
  @{ Name = "auto_e32_b097_r099"; Inputs = (New-AutoInputs 32 0.97 0.99) },
  @{ Name = "auto_e40_b093_r097"; Inputs = (New-AutoInputs 40 0.93 0.97) },
  @{ Name = "auto_e40_b095_r098"; Inputs = (New-AutoInputs 40 0.95 0.98) },
  @{ Name = "auto_e40_b097_r099"; Inputs = (New-AutoInputs 40 0.97 0.99) }
)

$results = @()

node src/cli/index.js timeframe $Timeframe | Out-Null
Start-Sleep -Milliseconds 1200

foreach($symbol in $Symbols) {
  node src/cli/index.js symbol $symbol | Out-Null
  Start-Sleep -Milliseconds 1800

  foreach($config in $configs) {
    Set-Inputs $config.Inputs
    Start-Sleep -Milliseconds 1500
    $m = Get-Metrics
    $row = [pscustomobject]@{
      Symbol = $symbol
      Config = $config.Name
      WR = $m.WR
      PF = $m.PF
      NP = $m.NP
      Trades = $m.Trades
    }
    $results += $row
    Write-Host ("{0} | {1} | WR {2}% | PF {3} | NP {4} | Trades {5}" -f $symbol, $config.Name, $m.WR, $m.PF, $m.NP, $m.Trades)
  }
}

Write-Host ""
Write-Host "Basket summary:"
$summary = $results |
  Group-Object Config |
  ForEach-Object {
    $group = $_.Group
    [pscustomobject]@{
      Config = $_.Name
      ProfitableSymbols = ($group | Where-Object { $_.NP -gt 0 }).Count
      AvgPF = [math]::Round((($group | Measure-Object PF -Average).Average), 3)
      TotalNP = [math]::Round((($group | Measure-Object NP -Sum).Sum), 2)
      AvgTrades = [math]::Round((($group | Measure-Object Trades -Average).Average), 2)
    }
  } |
  Sort-Object @{Expression = "ProfitableSymbols"; Descending = $true}, @{Expression = "AvgPF"; Descending = $true}, @{Expression = "TotalNP"; Descending = $true}

$summary | Format-Table -AutoSize

Write-Host ""
Write-Host "Top configs:"
$summary | Select-Object -First 5 | Format-Table -AutoSize


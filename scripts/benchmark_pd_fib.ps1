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
    in_64 = $false
    in_65 = "All Signals"
    in_66 = "240"
    in_67 = "D"
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
    in_79 = 1
    in_80 = 2
    in_81 = $false
  }
}

function New-Inputs(
  [bool]$usePdFib,
  [string]$applyTo,
  [string]$tf1,
  [string]$tf2,
  [bool]$useFibPrice,
  [bool]$useFibTime,
  [int]$minTfAlign,
  [int]$minTotal,
  [double]$discountMax = 0.50,
  [double]$premiumMin = 0.50,
  [double]$longFibMin = 0.21,
  [double]$longFibMax = 0.40,
  [double]$shortFibMin = 0.60,
  [double]$shortFibMax = 0.79,
  [int]$fibTolerance = 2
) {
  $inputs = New-BaseInputs
  $inputs.in_64 = $usePdFib
  $inputs.in_65 = $applyTo
  $inputs.in_66 = $tf1
  $inputs.in_67 = $tf2
  $inputs.in_70 = $discountMax
  $inputs.in_71 = $premiumMin
  $inputs.in_72 = $useFibPrice
  $inputs.in_73 = $longFibMin
  $inputs.in_74 = $longFibMax
  $inputs.in_75 = $shortFibMin
  $inputs.in_76 = $shortFibMax
  $inputs.in_77 = $useFibTime
  $inputs.in_78 = $fibTolerance
  $inputs.in_79 = $minTfAlign
  $inputs.in_80 = $minTotal
  return $inputs
}

$configs = @(
  @{
    Name = "auto_base"
    Inputs = (New-BaseInputs)
  },
  @{
    Name = "pd_all_4h_d_loose"
    Inputs = (New-Inputs $true "All Signals" "240" "D" $false $false 1 1)
  },
  @{
    Name = "pd_breakout_4h_d"
    Inputs = (New-Inputs $true "Breakout Only" "240" "D" $true $false 1 2)
  },
  @{
    Name = "pd_reject_4h_d"
    Inputs = (New-Inputs $true "Reject Only" "240" "D" $true $false 1 2)
  },
  @{
    Name = "pd_reject_time_4h_d"
    Inputs = (New-Inputs $true "Reject Only" "240" "D" $true $true 1 2)
  },
  @{
    Name = "pd_reject_1h_4h"
    Inputs = (New-Inputs $true "Reject Only" "60" "240" $true $false 1 2)
  },
  @{
    Name = "pd_reject_time_1h_4h_strict"
    Inputs = (New-Inputs $true "Reject Only" "60" "240" $true $true 2 4)
  }
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
Write-Host "Best config per symbol:"
$results |
  Group-Object Symbol |
  ForEach-Object {
    $_.Group | Sort-Object @{Expression = "PF"; Descending = $true}, @{Expression = "NP"; Descending = $true} | Select-Object -First 1
  } |
  Format-Table -AutoSize

Write-Host ""
Write-Host "Basket summary:"
$results |
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
  Sort-Object @{Expression = "ProfitableSymbols"; Descending = $true}, @{Expression = "AvgPF"; Descending = $true}, @{Expression = "TotalNP"; Descending = $true} |
  Format-Table -AutoSize

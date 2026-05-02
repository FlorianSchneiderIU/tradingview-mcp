param(
  [string[]]$Symbols = @("BINANCE:ETHUSDT", "BINANCE:BTCUSDT", "BINANCE:SOLUSDT", "BINANCE:BNBUSDT"),
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

function Get-Strategy() {
  node src/cli/index.js data strategy | ConvertFrom-Json
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

function Get-Session([datetime]$utcTime) {
  $hour = $utcTime.Hour
  if ($hour -lt 8) { return "Asia" }
  if ($hour -lt 16) { return "Europe" }
  return "US"
}

function Get-LossBucket([double]$profit, [double]$runUp) {
  if ($profit -ge 0) { return "Win" }
  $lossAbs = [math]::Abs($profit)
  if ($lossAbs -le 0) { return "Flat" }
  $ratio = $runUp / $lossAbs
  if ($ratio -lt 0.25) { return "Immediate Fail" }
  if ($ratio -lt 1.0) { return "Shallow Follow-Through" }
  return "Promising Then Reversed"
}

$profileInputs = @{
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
}

Set-Inputs $profileInputs
node src/cli/index.js timeframe $Timeframe | Out-Null
Start-Sleep -Milliseconds 1500

$allTrades = @()
$symbolSummaries = @()

foreach($symbol in $Symbols) {
  node src/cli/index.js symbol $symbol | Out-Null
  Start-Sleep -Milliseconds 2200

  $strategy = Get-Strategy
  $metrics = $strategy.metrics
  $closedTrades = @($metrics.trades)

  $tradeRows = foreach($trade in $closedTrades) {
    $entryTime = [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$trade.e.tm).UtcDateTime
    $exitTime = [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$trade.x.tm).UtcDateTime
    $profit = [double]$trade.tp.v
    $runUp = if ($null -ne $trade.rn) { [double]$trade.rn.v } else { 0.0 }
    $drawdown = if ($null -ne $trade.dd) { [double]$trade.dd.v } else { 0.0 }
    $lossAbs = if ($profit -lt 0) { [math]::Abs($profit) } else { 0.0 }
    $feRatio = if ($lossAbs -gt 0) { $runUp / $lossAbs } else { $null }
    $hoursHeld = ($exitTime - $entryTime).TotalHours
    $side = if ($trade.e.c -like "Long*") { "Long" } else { "Short" }

    [pscustomobject]@{
      Symbol = $symbol
      Side = $side
      EntryTimeUtc = $entryTime
      ExitTimeUtc = $exitTime
      EntryHourUtc = $entryTime.Hour
      Session = Get-Session $entryTime
      Weekday = $entryTime.DayOfWeek.ToString()
      Profit = [math]::Round($profit, 4)
      RunUp = [math]::Round($runUp, 4)
      Drawdown = [math]::Round($drawdown, 4)
      HoursHeld = [math]::Round($hoursHeld, 2)
      IsLoss = $profit -lt 0
      LossBucket = Get-LossBucket $profit $runUp
      FeToLossRatio = if ($null -ne $feRatio) { [math]::Round($feRatio, 3) } else { $null }
      GaveBackAtLeastHalfLoss = ($profit -lt 0 -and $runUp -ge ($lossAbs * 0.5))
      GaveBackFullLoss = ($profit -lt 0 -and $runUp -ge $lossAbs)
      FastStop6h = ($profit -lt 0 -and $hoursHeld -le 6)
      FastStop24h = ($profit -lt 0 -and $hoursHeld -le 24)
    }
  }

  $allTrades += $tradeRows

  $wins = @($tradeRows | Where-Object { -not $_.IsLoss })
  $losses = @($tradeRows | Where-Object { $_.IsLoss })
  $longTrades = @($tradeRows | Where-Object { $_.Side -eq "Long" })
  $shortTrades = @($tradeRows | Where-Object { $_.Side -eq "Short" })
  $longLossRate = if ($longTrades.Count -gt 0) { [math]::Round((@($longTrades | Where-Object IsLoss).Count / $longTrades.Count) * 100, 2) } else { $null }
  $shortLossRate = if ($shortTrades.Count -gt 0) { [math]::Round((@($shortTrades | Where-Object IsLoss).Count / $shortTrades.Count) * 100, 2) } else { $null }

  $symbolSummaries += [pscustomobject]@{
    Symbol = $symbol
    Trades = $tradeRows.Count
    WR = [math]::Round([double]$metrics.performance.all.percentProfitable * 100, 2)
    PF = [math]::Round([double]$metrics.performance.all.profitFactor, 3)
    NP = [math]::Round([double]$metrics.performance.all.netProfit, 2)
    LongLossRate = $longLossRate
    ShortLossRate = $shortLossRate
    LossMedianHours = [math]::Round((Get-Median ([double[]]($losses | ForEach-Object { $_.HoursHeld }))), 2)
    LossMedianFeRatio = [math]::Round((Get-Median ([double[]]($losses | Where-Object { $null -ne $_.FeToLossRatio } | ForEach-Object { $_.FeToLossRatio }))), 3)
    LossesGivingBackHalf = if ($losses.Count -gt 0) { [math]::Round((@($losses | Where-Object GaveBackAtLeastHalfLoss).Count / $losses.Count) * 100, 2) } else { 0 }
    LossesGivingBackFull = if ($losses.Count -gt 0) { [math]::Round((@($losses | Where-Object GaveBackFullLoss).Count / $losses.Count) * 100, 2) } else { 0 }
  }
}

$lossTrades = @($allTrades | Where-Object IsLoss)
$winTrades = @($allTrades | Where-Object { -not $_.IsLoss })

Write-Host ""
Write-Host "Per-symbol summary:"
$symbolSummaries | Format-Table -AutoSize

Write-Host ""
Write-Host "Loss anatomy across majors:"
[pscustomobject]@{
  TotalTrades = $allTrades.Count
  TotalLosses = $lossTrades.Count
  LossRate = if ($allTrades.Count -gt 0) { [math]::Round(($lossTrades.Count / $allTrades.Count) * 100, 2) } else { 0 }
  MedianLossHours = [math]::Round((Get-Median ([double[]]($lossTrades | ForEach-Object { $_.HoursHeld }))), 2)
  AvgLossHours = [math]::Round((($lossTrades | Measure-Object HoursHeld -Average).Average), 2)
  MedianFeToLoss = [math]::Round((Get-Median ([double[]]($lossTrades | Where-Object { $null -ne $_.FeToLossRatio } | ForEach-Object { $_.FeToLossRatio }))), 3)
  LossesFast6hPct = [math]::Round((@($lossTrades | Where-Object FastStop6h).Count / $lossTrades.Count) * 100, 2)
  LossesFast24hPct = [math]::Round((@($lossTrades | Where-Object FastStop24h).Count / $lossTrades.Count) * 100, 2)
  LossesGiveBackHalfPct = [math]::Round((@($lossTrades | Where-Object GaveBackAtLeastHalfLoss).Count / $lossTrades.Count) * 100, 2)
  LossesGiveBackFullPct = [math]::Round((@($lossTrades | Where-Object GaveBackFullLoss).Count / $lossTrades.Count) * 100, 2)
  LongLossPctOfAllLosses = [math]::Round((@($lossTrades | Where-Object { $_.Side -eq "Long" }).Count / $lossTrades.Count) * 100, 2)
  ShortLossPctOfAllLosses = [math]::Round((@($lossTrades | Where-Object { $_.Side -eq "Short" }).Count / $lossTrades.Count) * 100, 2)
} | Format-List

Write-Host ""
Write-Host "Loss buckets:"
$lossTrades |
  Group-Object LossBucket |
  ForEach-Object {
    [pscustomobject]@{
      Bucket = $_.Name
      Count = $_.Count
      Pct = [math]::Round(($_.Count / $lossTrades.Count) * 100, 2)
    }
  } |
  Sort-Object @{Expression = "Count"; Descending = $true} |
  Format-Table -AutoSize

Write-Host ""
Write-Host "Loss rate by session:"
$allTrades |
  Group-Object Session |
  ForEach-Object {
    $group = $_.Group
    [pscustomobject]@{
      Session = $_.Name
      Trades = $group.Count
      Losses = @($group | Where-Object IsLoss).Count
      LossRate = [math]::Round((@($group | Where-Object IsLoss).Count / $group.Count) * 100, 2)
      MedianLossHours = [math]::Round((Get-Median ([double[]](@($group | Where-Object IsLoss | ForEach-Object { $_.HoursHeld })))), 2)
    }
  } |
  Sort-Object @{Expression = "LossRate"; Descending = $true}, @{Expression = "Trades"; Descending = $true} |
  Format-Table -AutoSize

Write-Host ""
Write-Host "Loss rate by weekday:"
$allTrades |
  Group-Object Weekday |
  ForEach-Object {
    $group = $_.Group
    [pscustomobject]@{
      Weekday = $_.Name
      Trades = $group.Count
      Losses = @($group | Where-Object IsLoss).Count
      LossRate = [math]::Round((@($group | Where-Object IsLoss).Count / $group.Count) * 100, 2)
    }
  } |
  Sort-Object @{Expression = "LossRate"; Descending = $true}, @{Expression = "Trades"; Descending = $true} |
  Format-Table -AutoSize

Write-Host ""
Write-Host "Symbol / side breakdown:"
$allTrades |
  Group-Object Symbol, Side |
  ForEach-Object {
    $group = $_.Group
    $losses = @($group | Where-Object IsLoss)
    [pscustomobject]@{
      Symbol = $group[0].Symbol
      Side = $group[0].Side
      Trades = $group.Count
      Losses = $losses.Count
      LossRate = [math]::Round(($losses.Count / $group.Count) * 100, 2)
      MedianLossHours = [math]::Round((Get-Median ([double[]]($losses | ForEach-Object { $_.HoursHeld }))), 2)
      MedianFeToLoss = [math]::Round((Get-Median ([double[]]($losses | Where-Object { $null -ne $_.FeToLossRatio } | ForEach-Object { $_.FeToLossRatio }))), 3)
      PromisingReversalsPct = if ($losses.Count -gt 0) { [math]::Round((@($losses | Where-Object { $_.LossBucket -eq "Promising Then Reversed" }).Count / $losses.Count) * 100, 2) } else { 0 }
    }
  } |
  Sort-Object @{Expression = "LossRate"; Descending = $true}, @{Expression = "Trades"; Descending = $true} |
  Format-Table -AutoSize

Write-Host ""
Write-Host "Top symbols by loss count:"
$lossTrades |
  Group-Object Symbol |
  ForEach-Object {
    [pscustomobject]@{
      Symbol = $_.Name
      Losses = $_.Count
      MedianLossHours = [math]::Round((Get-Median ([double[]]($_.Group | ForEach-Object { $_.HoursHeld }))), 2)
      MedianFeToLoss = [math]::Round((Get-Median ([double[]]($_.Group | Where-Object { $null -ne $_.FeToLossRatio } | ForEach-Object { $_.FeToLossRatio }))), 3)
      PromisingReversalsPct = [math]::Round((@($_.Group | Where-Object { $_.LossBucket -eq "Promising Then Reversed" }).Count / $_.Count) * 100, 2)
    }
  } |
  Sort-Object @{Expression = "Losses"; Descending = $true} |
  Format-Table -AutoSize

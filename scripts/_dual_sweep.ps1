$orig = Get-Content -Raw scripts/current.pine

$stro_vals = @(50, 55, 60, 65, 70)
$candle_vals = @(0.3, 0.5, 0.7)

$results = @()
foreach($stro in $stro_vals){
  foreach($candle in $candle_vals){
    $code = $orig
    $code = [regex]::Replace($code, 'i_stro_threshold = input\.float\([0-9.]+,', "i_stro_threshold = input.float($stro,")
    $code = [regex]::Replace($code, 'i_candle_strength = input\.float\([0-9.]+,', "i_candle_strength = input.float($candle,")
    
    Set-Content -Path scripts/current.pine -Value $code -NoNewline
    node scripts/pine_push.js 2>&1 | Out-Null
    if($LASTEXITCODE -ne 0){ continue }
    
    $j = node src/cli/index.js data strategy | ConvertFrom-Json
    $all = $j.metrics.performance.all
    $wr = [math]::Round([double]$all.percentProfitable * 100, 2)
    $pf = [math]::Round([double]$all.profitFactor, 3)
    $np = [math]::Round([double]$all.netProfit, 3)
    $trades = [int]$all.totalTrades
    
    $results += [pscustomobject]@{StroThreshold=$stro; CandleStr=$candle; WR=$wr; PF=$pf; NP=$np; Trades=$trades}
  }
}

Write-Host "Top 10 by WR*Trades (quality+frequency):"
$results | Sort-Object @{E={$_.WR * $_.Trades};Descending=$true} | Select-Object -First 10 | Format-Table -Auto

Write-Host "`nTop 5 by WR only (quality focus):"
$results | Sort-Object WR -Descending | Select-Object -First 5 | Format-Table -Auto

$orig = Get-Content -Raw scripts/current.pine

$variants = @(30, 40, 50, 60, 65, 70)

$results = @()
foreach($threshold in $variants){
  $code = $orig -replace "stro_ok_long = not i_use_stro or stro_k < \d+\.0", "stro_ok_long = not i_use_stro or stro_k < $threshold.0"
  
  Set-Content -Path scripts/current.pine -Value $code -NoNewline
  node scripts/pine_push.js 2>&1 | Out-Null
  if($LASTEXITCODE -ne 0){ continue }
  
  $j = node src/cli/index.js data strategy | ConvertFrom-Json
  $all = $j.metrics.performance.all
  $wr = [math]::Round([double]$all.percentProfitable * 100, 2)
  $pf = [math]::Round([double]$all.profitFactor, 3)
  $np = [math]::Round([double]$all.netProfit, 3)
  $trades = [int]$all.totalTrades
  
  $results += [pscustomobject]@{StroThreshold=$threshold; WR=$wr; PF=$pf; NP=$np; Trades=$trades; TPD=[math]::Round($trades/94,2)}
}

$results | Sort-Object WR -Descending | Sort-Object Trades -Descending | Format-Table -Auto

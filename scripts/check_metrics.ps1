$j = node src/cli/index.js data strategy | ConvertFrom-Json
$all = $j.metrics.performance.all
$wr = [math]::Round([double]$all.percentProfitable * 100, 2)
$pf = [math]::Round([double]$all.profitFactor, 3)
$np = [math]::Round([double]$all.netProfit, 3)
$trades = [int]$all.totalTrades
Write-Host "WR: $wr% | PF: $pf | NP: $np | Trades: $trades"

$orig = Get-Content -Raw scripts/current.pine

$configs = @(
  @{name='best_balance'; stro=65; candle=0.3; rr=2.4},
  @{name='tight_rr_1.5'; stro=65; candle=0.3; rr=1.5},
  @{name='tight_rr_1.8'; stro=65; candle=0.3; rr=1.8},
  @{name='very_tight_rr_1.2'; stro=70; candle=0.2; rr=1.2},
  @{name='tight_sl_sl0.5'; stro=65; candle=0.3; rr=2.0; sl_atr=0.5}
)

$results = @()
foreach($c in $configs){
  $code = $orig
  $code = [regex]::Replace($code, 'i_stro_threshold = input\.float\([0-9.]+,', "i_stro_threshold = input.float($($c.stro),")
  $code = [regex]::Replace($code, 'i_candle_strength = input\.float\([0-9.]+,', "i_candle_strength = input.float($($c.candle),")
  $code = [regex]::Replace($code, 'i_rr\s*=\s*input\.float\([0-9.]+,', "i_rr = input.float($($c.rr),")
  if($c.sl_atr){
    $code = [regex]::Replace($code, 'i_sl_atr = input\.float\([0-9.]+,', "i_sl_atr = input.float($($c.sl_atr),")
  }
  
  Set-Content -Path scripts/current.pine -Value $code -NoNewline
  node scripts/pine_push.js 2>&1 | Out-Null
  if($LASTEXITCODE -ne 0){ continue }
  
  $j = node src/cli/index.js data strategy | ConvertFrom-Json
  $all = $j.metrics.performance.all
  $wr = [math]::Round([double]$all.percentProfitable * 100, 2)
  $pf = [math]::Round([double]$all.profitFactor, 3)
  $np = [math]::Round([double]$all.netProfit, 3)
  $trades = [int]$all.totalTrades
  
  $results += [pscustomobject]@{Config=$c.name; WR=$wr; PF=$pf; NP=$np; Trades=$trades}
}

$results | Sort-Object WR -Descending | Format-Table -Auto

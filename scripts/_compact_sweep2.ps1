$orig = Get-Content -Raw scripts/current.pine

$configs = @(
  @{name='ultra_tight_rr1.0'; stro=70; candle=0.2; rr=1.0; sl=0.5},
  @{name='ultra_tight_rr1.0_sl0.4'; stro=70; candle=0.2; rr=1.0; sl=0.4},
  @{name='aggressive_rr1.1'; stro=75; candle=0.2; rr=1.1; sl=0.4},
  @{name='very_tight_rr1.2_sl0.4'; stro=70; candle=0.2; rr=1.2; sl=0.4},
  @{name='tight_ish_rr1.3_sl0.5'; stro=70; candle=0.2; rr=1.3; sl=0.5}
)

$results = @()
foreach($c in $configs){
  $code = $orig
  $code = [regex]::Replace($code, 'i_stro_threshold = input\.float\([0-9.]+,', "i_stro_threshold = input.float($($c.stro),")
  $code = [regex]::Replace($code, 'i_candle_strength = input\.float\([0-9.]+,', "i_candle_strength = input.float($($c.candle),")
  $code = [regex]::Replace($code, 'i_rr\s*=\s*input\.float\([0-9.]+,', "i_rr = input.float($($c.rr),")
  $code = [regex]::Replace($code, 'i_sl_atr = input\.float\([0-9.]+,', "i_sl_atr = input.float($($c.sl),")
  
  Set-Content -Path scripts/current.pine -Value $code -NoNewline
  node scripts/pine_push.js 2>&1 | Out-Null
  if($LASTEXITCODE -ne 0){ Write-Host "Compile error for $($c.name)"; continue }
  
  $j = node src/cli/index.js data strategy | ConvertFrom-Json
  $all = $j.metrics.performance.all
  $wr = [math]::Round([double]$all.percentProfitable * 100, 2)
  $pf = [math]::Round([double]$all.profitFactor, 3)
  $np = [math]::Round([double]$all.netProfit, 3)
  $trades = [int]$all.totalTrades
  
  $results += [pscustomobject]@{Config=$c.name; WR=$wr; PF=$pf; NP=$np; Trades=$trades}
}

$results | Sort-Object WR -Descending | Format-Table -Auto
Write-Host "TARGET: 70% WR + 1:2 RR minimum + 30+ trades"

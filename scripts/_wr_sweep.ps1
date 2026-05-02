$path = "scripts/current.pine"
$orig = Get-Content -Raw $path

# Variants focused on higher WR without killing trade count
$variants = @(
  @{name='current'; touch='0.20'; rsi='58'; cd='5'; min_fvg='0.20'; touch_fvg='14'},
  @{name='strict_wr_1'; touch='0.15'; rsi='55'; cd='7'; min_fvg='0.25'; touch_fvg='12'},
  @{name='strict_wr_2'; touch='0.10'; rsi='52'; cd='8'; min_fvg='0.30'; touch_fvg='10'},
  @{name='balanced_wr'; touch='0.18'; rsi='56'; cd='6'; min_fvg='0.22'; touch_fvg='13'},
  @{name='aggressive_wr'; touch='0.12'; rsi='54'; cd='6'; min_fvg='0.28'; touch_fvg='11'}
)

$results = @()
foreach($v in $variants){
  $code = $orig
  $code = [regex]::Replace($code, 'i_fvg_life = input\.int\([0-9]+,', "i_fvg_life = input.int($($v.touch_fvg),")
  $code = [regex]::Replace($code, 'i_touch_margin = input\.float\([0-9.]+,', "i_touch_margin = input.float($($v.touch),")
  $code = [regex]::Replace($code, 'i_min_fvg_atr = input\.float\([0-9.]+,', "i_min_fvg_atr = input.float($($v.min_fvg),")
  $code = [regex]::Replace($code, 'i_rsi_max_long = input\.int\([0-9]+,', "i_rsi_max_long = input.int($($v.rsi),")
  $code = [regex]::Replace($code, 'i_cooldown_bars = input\.int\([0-9]+,', "i_cooldown_bars = input.int($($v.cd),")
  
  Set-Content -Path $path -Value $code -NoNewline
  node scripts/pine_push.js 2>&1 | Out-Null
  if($LASTEXITCODE -ne 0){ continue }

  $j = node src/cli/index.js data strategy | ConvertFrom-Json
  $all = $j.metrics.performance.all
  $from = [DateTimeOffset]::FromUnixTimeMilliseconds($j.metrics.settings.dateRange.trade.from).UtcDateTime
  $to = [DateTimeOffset]::FromUnixTimeMilliseconds($j.metrics.settings.dateRange.trade.to).UtcDateTime
  $days = [Math]::Max(1, ($to - $from).TotalDays)
  $wr = [math]::Round([double]$all.percentProfitable*100,2)
  $pf = [math]::Round([double]$all.profitFactor,3)
  $np = [math]::Round([double]$all.netProfit,3)
  $trades = [int]$all.totalTrades
  $tpd = [math]::Round([double]$trades / $days,2)

  $results += [pscustomobject]@{
    Variant=$v.name; WinRate=$wr; ProfitFactor=$pf; NetProfit=$np; TotalTrades=$trades; TradesPerDay=$tpd
  }
}

$results | Sort-Object WinRate -Descending | ConvertTo-Json -Depth 3

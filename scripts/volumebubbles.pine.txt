// This script is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
// https://creativecommons.org/licenses/by-nc-sa/4.0/
// © QuantAlgo

//@version=6
indicator("Volume Bubbles [QuantAlgo]", overlay=true, max_labels_count=500)

import TradingView/ta/12 as taLib

//              ╔════════════════════════════════╗              //
//              ║      USER-DEFINED SETTINGS     ║              //
//              ╚════════════════════════════════╝              //

var string detection_settings   = '════════ Detection Method ════════'
var string delta_settings       = '════════ Volume Delta Settings ════════'
var string sensitivity_settings = '════════ Cluster Sensitivity ════════'
var string consensus_settings   = '════════ Lookback & Consensus ════════'
var string visual_settings      = '════════ Visualization Settings ════════'

tooltip_detect_method   = 'Determines what data drives cluster detection.\n\nVolume Only: Uses raw bar volume. Simplest and most reliable, works on all symbols. Detects any elevated volume regardless of buy/sell breakdown.\n\nDelta Only: Uses net buy-sell pressure from lower timeframe scan. Triggers only when directional volume (not just total volume) is elevated. Requires LTF data availability.\n\nVolume + Delta: Both volume AND delta must independently exceed their thresholds. Strictest mode, filters out high-volume bars with balanced buy/sell (no clear direction). Best for high-conviction signals.\n\nVolume OR Delta: Either elevated volume OR elevated delta triggers a cluster. Most inclusive mode, catches both pure volume events and strong directional flow even on normal-volume bars.'
tooltip_classify        = 'Determines how each cluster is labeled as Buy, Sell, or Mixed.\n\nCandle Direction: Uses the candle color (green = buy, red = sell). Simple and always available, but does not account for intra-bar volume flow.\n\nDelta Direction: Uses the sign of net volume delta (positive = buy, negative = sell). More accurate reflection of actual buying/selling pressure, but requires LTF data. Falls back to candle direction if delta is unavailable.\n\nBoth: Both candle direction AND delta must point the same way. If they disagree, the cluster is labeled Mixed. Highest confidence classification but produces more Mixed labels.'
tooltip_small_pct       = 'Percentile threshold for identifying Small volume clusters. A value of 75 means a bar must exceed 75% of recent volume history (top 25%) to qualify as a Small cluster. Lower values produce more frequent signals, higher values make Small clusters rarer. This is the first tier of detection: bars that show notably elevated volume but not extreme. Ideal starting point for spotting early accumulation or distribution activity.'
tooltip_medium_pct      = 'Percentile threshold for identifying Medium volume clusters. A value of 90 means a bar must exceed 90% of recent volume history (top 10%) to qualify as a Medium cluster. These represent significant volume events that often coincide with institutional activity, key support/resistance tests, or momentum acceleration. Medium clusters appear less frequently than Small ones and typically signal stronger conviction behind the move.'
tooltip_big_pct         = 'Percentile threshold for identifying Big volume clusters. A value of 97 means a bar must exceed 97% of recent volume history (top 3%) to qualify as a Big cluster. These are rare, high-impact events typically associated with climax moves, major breakouts/breakdowns, capitulation, or large institutional block trades. When a Big cluster appears, it often marks a significant turning point or trend acceleration.'
tooltip_consensus       = 'Controls how many lookback windows must agree that volume is elevated before a cluster is confirmed.\n\nAny Window: Only one window needs to confirm. Most sensitive, produces the most signals. Good for catching early moves but may generate more noise.\n\nMajority (2 of 3): At least two of the three windows must agree. Balanced approach that filters out noise while remaining responsive. Recommended for most users.\n\nAll Windows (strictest): All three windows must confirm. Most conservative, only triggers on volume that is elevated relative to short, medium, AND long-term history. Fewest signals but highest confidence.'
tooltip_short_window    = 'Short-term lookback window for percentile calculation. Captures recent volume context and adapts quickly to changing market conditions. A value of 20 means the indicator compares current volume against the last 20 bars. Shorter windows react faster to regime changes (e.g., earnings season, news events) but may be less stable. Works best for detecting sudden volume shifts.'
tooltip_mid_window      = 'Medium-term lookback window for percentile calculation. Provides a balanced perspective between recent activity and broader context. A value of 50 means the indicator compares current volume against the last 50 bars. This window smooths out day-to-day fluctuations while still adapting to medium-term changes in market activity. Serves as the anchor for the consensus system.'
tooltip_long_window     = 'Long-term lookback window for percentile calculation. Establishes a broad baseline of typical volume behavior. A value of 100 means the indicator compares current volume against the last 100 bars. This window is resistant to short-term noise and identifies volume that is truly exceptional relative to extended history. Helps avoid false signals during temporary volume spikes.'
tooltip_custom_tf       = 'Override the automatically detected lower timeframe for delta calculation. Enable this if you want precise control over the granularity of buy/sell volume estimation. Higher timeframes provide more historical data but less precision in delta calculation. Lower timeframes give more accurate delta but may have limited history.\n\nCommon choices: 1S for sub-minute charts, 1 for hourly charts, 5 for 4H charts, 15 for daily charts.'
tooltip_custom_tf_value = 'The specific lower timeframe to use when custom override is enabled. This timeframe must be lower than your chart timeframe for meaningful delta calculation. If the specified timeframe data is unavailable, the indicator falls back to candle direction for buy/sell classification.'
tooltip_show_small      = 'Toggle visibility of Small volume clusters. Small clusters represent the lowest tier of elevated volume (default: top 25%). Disabling this reduces visual clutter and focuses attention on more significant Medium and Big events. Useful when you only want to see high-impact volume activity.'
tooltip_show_medium     = 'Toggle visibility of Medium volume clusters. Medium clusters represent the middle tier of elevated volume (default: top 10%). These often mark meaningful institutional activity or momentum shifts. Disable to focus exclusively on the rarest Big cluster events.'
tooltip_show_big        = 'Toggle visibility of Big volume clusters. Big clusters represent the highest tier of elevated volume (default: top 3%). These are rare, high-impact events that often mark major turning points. It is recommended to always keep these enabled as they represent the most significant volume activity.'
tooltip_color_preset    = 'Pre-configured color schemes optimized for different chart themes and visual preferences. Each preset maintains strong contrast for clear bullish vs bearish identification.\n\nClassic: Traditional green/red for familiar directional reading.\nAqua: Cool blue/warm orange for dark backgrounds.\nCosmic: Cyan/purple for a modern, high-contrast look.\nCyber: Orange/teal for an inverted warm/cool palette.\nNeon: Yellow/magenta for maximum visibility.\nCustom: Use your own color choices below.'
tooltip_buy_color       = 'Color for Buy clusters (positive delta or bullish candle). Represents periods where buying pressure dominates, indicating accumulation and potential upward price movement. Applied to bubble fills and label text for bullish-classified clusters. Only used when Color Preset is set to Custom.'
tooltip_sell_color      = 'Color for Sell clusters (negative delta or bearish candle). Represents periods where selling pressure dominates, indicating distribution and potential downward price movement. Applied to bubble fills and label text for bearish-classified clusters. Only used when Color Preset is set to Custom.'
tooltip_mix_color       = 'Color for Mixed clusters (when buy/sell classification is inconclusive). Appears when using "Both" classification mode and candle direction conflicts with delta direction. Indicates elevated volume without clear directional bias. Only used when Color Preset is set to Custom.'
tooltip_show_labels     = 'Display text inside each volume bubble showing the selected metric. When enabled, bubbles contain numeric information (volume, delta, ratio, or combinations). When disabled, only the colored circles appear. Disabling can reduce visual clutter on busy charts while still showing cluster locations and sizes. Hover over any bubble to see full details in the tooltip regardless of this setting.'
tooltip_label_content   = 'Choose what text appears inside each bubble.\n\nVolume: Raw bar volume formatted as K/M/B.\nDelta: Net buy-sell volume delta. Falls back to volume if delta is unavailable.\nRatio: Volume as a multiple of the average (e.g., 2.5x means 2.5 times average volume).\nVol + Delta: Both volume and delta on separate lines.\nVol + Ratio: Both volume and ratio on separate lines.\n\nFull details including all metrics, detection method, and window confirmations are always available in the hover tooltip.'

detectMethod = input.string("Volume OR Delta", "Detection Method",
     options = ["Volume Only", "Delta Only", "Volume + Delta", "Volume OR Delta"], group = detection_settings,
     tooltip = tooltip_detect_method)
classifyMode = input.string("Both", "Buy/Sell Classification",
     options = ["Candle Direction", "Delta Direction", "Both"],                    group = detection_settings,
     tooltip = tooltip_classify)

useCustomTF = input.bool(false,    "Use Custom Lower Timeframe", group = delta_settings,                       tooltip = tooltip_custom_tf)
customTF    = input.timeframe("1", "Custom Timeframe",           group = delta_settings, active = useCustomTF, tooltip = tooltip_custom_tf_value)

smallPct  = input.float(75, "Small Cluster Percentile",  step = 1, group = sensitivity_settings, tooltip = tooltip_small_pct)
mediumPct = input.float(90, "Medium Cluster Percentile", step = 1, group = sensitivity_settings, tooltip = tooltip_medium_pct)
bigPct    = input.float(97, "Big Cluster Percentile",    step = 1, group = sensitivity_settings, tooltip = tooltip_big_pct)

consensusMode = input.string("Majority (2 of 3)", "Consensus Mode",
     options  = ["Any Window", "Majority (2 of 3)", "All Windows (strictest)"], group = consensus_settings,
     tooltip  = tooltip_consensus)
shortLen = input.int(20,  "Short Window",  group = consensus_settings, tooltip = tooltip_short_window)
midLen   = input.int(50,  "Medium Window", group = consensus_settings, tooltip = tooltip_mid_window)
longLen  = input.int(100, "Long Window",   group = consensus_settings, tooltip = tooltip_long_window)

showSmall    = input.bool(true,  "Show Small Clusters",  group = visual_settings, tooltip = tooltip_show_small)
showMedium   = input.bool(true,  "Show Medium Clusters", group = visual_settings, tooltip = tooltip_show_medium)
showBig      = input.bool(true,  "Show Big Clusters",    group = visual_settings, tooltip = tooltip_show_big)

colorPreset  = input.string("Custom", "Color Preset",
     options = ["Classic", "Aqua", "Cosmic", "Cyber", "Neon", "Custom"], group = visual_settings, tooltip = tooltip_color_preset)
buyColorInput  = input.color(color.new(#00ffaa, 40), "Buy Color",   group = visual_settings, tooltip = tooltip_buy_color)
sellColorInput = input.color(color.new(#ff0000, 20), "Sell Color",  group = visual_settings, tooltip = tooltip_sell_color)
mixColorInput  = input.color(color.new(#ffd600, 30), "Mixed Color", group = visual_settings, tooltip = tooltip_mix_color)

[presetBuy, presetSell, presetMix] = switch colorPreset
    "Classic" => [color.new(#00ff00, 40), color.new(#ff0000, 20), color.new(#ffd600, 30)]
    "Aqua"    => [color.new(#00bfff, 40), color.new(#ff8c00, 20), color.new(#9932cc, 30)]
    "Cosmic"  => [color.new(#49ffce, 40), color.new(#9932cc, 20), color.new(#ffd600, 30)]
    "Cyber"   => [color.new(#00cccc, 20), color.new(#ff6600, 40), color.new(#ffd600, 30)]
    "Neon"    => [color.new(#ffff00, 40), color.new(#ff00ff, 20), color.new(#9932cc, 30)]
    "Custom"  => [buyColorInput, sellColorInput, mixColorInput]

color buyColor  = presetBuy
color sellColor = presetSell
color mixColor  = presetMix

showLabels   = input.bool(true, "Show Numbers in Bubbles", group = visual_settings, tooltip = tooltip_show_labels)
labelContent = input.string("Volume", "Bubble Text",
     options = ["Volume", "Delta", "Ratio", "Vol + Delta", "Vol + Ratio"], group = visual_settings,
     tooltip = tooltip_label_content)

//              ╔════════════════════════════════╗              //
//              ║    VOLUME DELTA CALCULATION    ║              //
//              ╚════════════════════════════════╝              //

var string lowerTimeframe = switch
    useCustomTF          => customTF
    timeframe.isseconds  => "1S"
    timeframe.isintraday => "1"
    timeframe.isdaily    => "5"
    => "60"

[_, _, _, lastDelta] = taLib.requestVolumeDelta(lowerTimeframe)
bool  deltaAvailable = not na(lastDelta)
float netDelta = nz(lastDelta)
float absDelta = math.abs(netDelta)

//              ╔════════════════════════════════╗              //
//              ║     PERCENTILE THRESHOLDS      ║              //
//              ╚════════════════════════════════╝              //

float vSmallShort = ta.percentile_linear_interpolation(volume, shortLen, smallPct)
float vSmallMid   = ta.percentile_linear_interpolation(volume, midLen,   smallPct)
float vSmallLong  = ta.percentile_linear_interpolation(volume, longLen,  smallPct)
float vMedShort   = ta.percentile_linear_interpolation(volume, shortLen, mediumPct)
float vMedMid     = ta.percentile_linear_interpolation(volume, midLen,   mediumPct)
float vMedLong    = ta.percentile_linear_interpolation(volume, longLen,  mediumPct)
float vBigShort   = ta.percentile_linear_interpolation(volume, shortLen, bigPct)
float vBigMid     = ta.percentile_linear_interpolation(volume, midLen,   bigPct)
float vBigLong    = ta.percentile_linear_interpolation(volume, longLen,  bigPct)

float dSmallShort = ta.percentile_linear_interpolation(absDelta, shortLen, smallPct)
float dSmallMid   = ta.percentile_linear_interpolation(absDelta, midLen,   smallPct)
float dSmallLong  = ta.percentile_linear_interpolation(absDelta, longLen,  smallPct)
float dMedShort   = ta.percentile_linear_interpolation(absDelta, shortLen, mediumPct)
float dMedMid     = ta.percentile_linear_interpolation(absDelta, midLen,   mediumPct)
float dMedLong    = ta.percentile_linear_interpolation(absDelta, longLen,  mediumPct)
float dBigShort   = ta.percentile_linear_interpolation(absDelta, shortLen, bigPct)
float dBigMid     = ta.percentile_linear_interpolation(absDelta, midLen,   bigPct)
float dBigLong    = ta.percentile_linear_interpolation(absDelta, longLen,  bigPct)

//              ╔════════════════════════════════╗              //
//              ║      CONSENSUS DETECTION       ║              //
//              ╚════════════════════════════════╝              //

f_consensus(bool pS, bool pM, bool pL, string mode) =>
    int hits = (pS ? 1 : 0) + (pM ? 1 : 0) + (pL ? 1 : 0)
    switch mode
        "Any Window"               => hits >= 1
        "Majority (2 of 3)"       => hits >= 2
        "All Windows (strictest)" => hits >= 3

bool volBig   = f_consensus(volume >= vBigShort, volume >= vBigMid, volume >= vBigLong, consensusMode)
bool volMed   = not volBig and f_consensus(volume >= vMedShort, volume >= vMedMid, volume >= vMedLong, consensusMode)
bool volSmall = not volBig and not volMed and f_consensus(volume >= vSmallShort, volume >= vSmallMid, volume >= vSmallLong, consensusMode)

bool deltaBig   = f_consensus(absDelta >= dBigShort, absDelta >= dBigMid, absDelta >= dBigLong, consensusMode)
bool deltaMed   = not deltaBig and f_consensus(absDelta >= dMedShort, absDelta >= dMedMid, absDelta >= dMedLong, consensusMode)
bool deltaSmall = not deltaBig and not deltaMed and f_consensus(absDelta >= dSmallShort, absDelta >= dSmallMid, absDelta >= dSmallLong, consensusMode)

//              ╔════════════════════════════════╗              //
//              ║      COMBINED DETECTION        ║              //
//              ╚════════════════════════════════╝              //

f_detect(bool vHit, bool dHit, string method) =>
    switch method
        "Volume Only"      => vHit
        "Delta Only"       => dHit
        "Volume + Delta"   => vHit and dHit
        "Volume OR Delta"  => vHit or dHit

string effectiveMethod = deltaAvailable ? detectMethod : "Volume Only"

bool isBigCluster   = f_detect(volBig, deltaBig, effectiveMethod)
bool isMedCluster   = not isBigCluster and f_detect(volMed or volBig, deltaMed or deltaBig, effectiveMethod)
bool isSmallCluster = not isBigCluster and not isMedCluster and f_detect(volSmall or volMed or volBig, deltaSmall or deltaMed or deltaBig, effectiveMethod)

//              ╔════════════════════════════════╗              //
//              ║     BUY/SELL CLASSIFICATION    ║              //
//              ╚════════════════════════════════╝              //

bool candleBuy  = close >= open
bool candleSell = close < open

bool isBuy = switch classifyMode
    "Candle Direction"          => candleBuy
    "Delta Direction"           => deltaAvailable ? netDelta > 0 : candleBuy
    "Both" => deltaAvailable ? (candleBuy and netDelta > 0) : candleBuy

bool isSell = switch classifyMode
    "Candle Direction"          => candleSell
    "Delta Direction"           => deltaAvailable ? netDelta < 0 : candleSell
    "Both" => deltaAvailable ? (candleSell and netDelta < 0) : candleSell

//              ╔════════════════════════════════╗              //
//              ║           FORMATTING           ║              //
//              ╚════════════════════════════════╝              //

f_formatNum(float v) =>
    float av = math.abs(v)
    string sign = v < 0 ? "-" : ""
    av >= 1000000000 ? sign + str.tostring(av / 1000000000, "#.##") + "B" :
     av >= 1000000   ? sign + str.tostring(av / 1000000, "#.##") + "M" :
     av >= 1000      ? sign + str.tostring(av / 1000, "#.##") + "K" :
     sign + str.tostring(av, "#")

float avgVol   = ta.sma(volume, midLen)
float volRatio = avgVol > 0 ? volume / avgVol : 0.0

bool isSmallVisible = isSmallCluster and showSmall
bool isMedVisible   = isMedCluster and showMedium
bool isBigVisible   = isBigCluster and showBig
bool isAnyVisible   = isSmallVisible or isMedVisible or isBigVisible

float bubbleY     = isAnyVisible ? (isSell ? high : low) : na
color bubbleColor = isBuy ? buyColor : isSell ? sellColor : mixColor

plotshape(isSmallVisible ? bubbleY : na, "", shape.circle, location.absolute, bubbleColor, 0, "", na, size=size.normal)
plotshape(isMedVisible   ? bubbleY : na, "", shape.circle, location.absolute, bubbleColor, 0, "", na, size=size.large)
plotshape(isBigVisible   ? bubbleY : na, "", shape.circle, location.absolute, bubbleColor, 0, "", na, size=size.huge)

//              ╔════════════════════════════════╗              //
//              ║          TEXT OVERLAY          ║              //
//              ╚════════════════════════════════╝              //

if isAnyVisible and showLabels
    string fv = f_formatNum(volume)
    string fd = f_formatNum(netDelta)
    string rx = str.tostring(volRatio, "#.#") + "x"

    string bubbleText = switch labelContent
        "Volume"      => fv
        "Delta"       => deltaAvailable ? fd : fv
        "Ratio"       => rx
        "Vol + Delta" => deltaAvailable ? fv + "\n" + fd : fv
        "Vol + Ratio" => fv + "\n" + rx

    int cLevel      = isBigVisible ? 3 : isMedVisible ? 2 : 1
    string levelTag = cLevel == 3 ? "BIG" : cLevel == 2 ? "MEDIUM" : "SMALL"
    string sideTag  = isBuy ? "BUY" : isSell ? "SELL" : "MIXED"

    bool vpS = cLevel == 3 ? volume >= vBigShort : cLevel == 2 ? volume >= vMedShort : volume >= vSmallShort
    bool vpM = cLevel == 3 ? volume >= vBigMid   : cLevel == 2 ? volume >= vMedMid   : volume >= vSmallMid
    bool vpL = cLevel == 3 ? volume >= vBigLong  : cLevel == 2 ? volume >= vMedLong  : volume >= vSmallLong
    bool dpS = cLevel == 3 ? absDelta >= dBigShort : cLevel == 2 ? absDelta >= dMedShort : absDelta >= dSmallShort
    bool dpM = cLevel == 3 ? absDelta >= dBigMid   : cLevel == 2 ? absDelta >= dMedMid   : absDelta >= dSmallMid
    bool dpL = cLevel == 3 ? absDelta >= dBigLong  : cLevel == 2 ? absDelta >= dMedLong  : absDelta >= dSmallLong

    string vWin = (vpS ? "S" : "·") + " " + (vpM ? "M" : "·") + " " + (vpL ? "L" : "·")
    string dWin = (dpS ? "S" : "·") + " " + (dpM ? "M" : "·") + " " + (dpL ? "L" : "·")

    string deltaNote = deltaAvailable ? "" : "\nNote: Delta unavailable. Using candle direction and volume only."
    string tip = "── " + levelTag + " " + sideTag + " ──" +
         "\nVolume: " + fv +
         "\nDelta: " + (deltaAvailable ? fd : "n/a") +
         "\nRatio: " + str.tostring(volRatio, "#.##") + "x" +
         "\nMethod: " + detectMethod + (deltaAvailable ? "" : " > Vol Only") +
         "\nVol Windows: [" + vWin + "]" +
         "\nDelta Windows: [" + dWin + "]" +
         "\nClassification: " + classifyMode +
         deltaNote +
         "\n─────────────────────" +
         "\n⚠️ Note: Pine Script limits labels to 500." +
         "\nOlder bubbles still appear but without" +
         "\ntext or tooltip data (plotshape has no cap)."

    label.new(bar_index, bubbleY, bubbleText,
         style=label.style_label_center,
         color=color.new(chart.fg_color, 100),
         textcolor=chart.fg_color,
         size=size.small,
         tooltip=tip)

//              ╔════════════════════════════════╗              //
//              ║        NO VOLUME WARNING       ║              //
//              ╚════════════════════════════════╝              //

var float cumVol = 0.0
cumVol += nz(volume)

var table warningBox = na

if barstate.islast and cumVol == 0
    if na(warningBox)
        warningBox := table.new(position.middle_center, 1, 1, bgcolor = #ff0000)
    table.cell(warningBox, 0, 0,
         '\n  ⚠️  NO VOLUME DATA  ⚠️  \n\n  This symbol does not provide volume data.  \n\n  Volume Bubbles requires volume to function.  \n\n  Switch to a symbol on an exchange that provides volume data.  \n',
         text_color = chart.fg_color, text_size = size.large,
         bgcolor = #ff0000)
else
    if not na(warningBox)
        table.delete(warningBox)
        warningBox := na

//              ╔════════════════════════════════╗              //
//              ║             ALERTS             ║              //
//              ╚════════════════════════════════╝              //

alertcondition(isBigVisible,                 "Big Cluster",  "Big volume cluster detected!")
alertcondition(isBigVisible or isMedVisible, "Med+ Cluster", "Medium or Big volume cluster detected!")
alertcondition(isAnyVisible,                 "Any Cluster",  "New volume cluster detected!")

//              ╔════════════════════════════════╗              //
//              ║           CREATED BY           ║              //
//              ╚════════════════════════════════╝              //

// ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗     █████╗ ██╗      ██████╗  ██████╗ 
//██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝    ██╔══██╗██║     ██╔════╝ ██╔═══██╗
//██║   ██║██║   ██║███████║██╔██╗ ██║   ██║       ███████║██║     ██║  ███╗██║   ██║
//██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║       ██╔══██║██║     ██║   ██║██║   ██║
//╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║       ██║  ██║███████╗╚██████╔╝╚██████╔╝
// ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚═╝  ╚═╝╚══════╝ ╚═════╝  ╚═════╝
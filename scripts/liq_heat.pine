// This work is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International  
// https://creativecommons.org/licenses/by-nc-sa/4.0/
// © BigBeluga

//@version=6
indicator("Liquidity Thermal Map [BigBeluga]", overlay = true, max_lines_count = 500, calc_bars_count = 1500)

// ＩＮＰＵＴＳ ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――{

profileType = input.string(
    "Balanced Profile",
    "Profile Depth",
    options = ["Shallow Profile", "Balanced Profile", "Deep Profile"],
    tooltip =
        "Controls how much historical data is used to build the liquidity profile.\n\n" +
        "Shallow Profile  → Short-term structure\n" +
        "Balanced Profile → Medium-term structure\n" +
        "Deep Profile     → Broad structural distribution"
)

period =
     profileType == "Shallow Profile"  ? 100  :
     profileType == "Balanced Profile" ? 300  :
     600


sensitivityMode = input.string(
    "Balanced",
    "Thermal Sensitivity",
    options = ["High Contrast", "Balanced", "Smooth"],
    tooltip =
        "Controls heatmap intensity.\n\n" +
        "High Contrast → Strong color separation\n" +
        "Balanced      → Default behavior\n" +
        "Smooth        → Softer gradient blending"
)

min =
     sensitivityMode == "High Contrast" ? 0.5  :
     sensitivityMode == "Balanced"      ? 0.25 :
     0.0   

pocCol1  = input.color(
    color.rgb(189, 43, 43),
    "Upper Liquidity Color",
    tooltip =
        "Color used for liquidity located above current price.\n" +
        "Typically represents overhead resistance zones."
)

pocCol2  = input.color(
    color.rgb(41, 180, 83),
    "Lower Liquidity Color",
    tooltip =
        "Color used for liquidity located below current price.\n" +
        "Typically represents structural support zones."
)

removeCol = input.bool(
    true,
    "Hide Active Candle Color",
    tooltip =
        "Removes heatmap coloring around the active candle\n" +
        "to improve real-time price visibility."
)

colorNA = #44444400
// }



// ＣＡＬＣＵＬＡＴＩＯＮＳ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――{

H = ta.highest(period)
L = ta.lowest(period)


step   = (H-L)/31

levels  = array.new<float>(31)
volBins = array.new<float>(31, 0.)


gradient(volBins, number)=>
    val  = volBins.get(number)/volBins.max()
    loww = levels.get(number)
    upp  = levels.get(number + 1 > 29 ? 29 : number + 1)

    pocCol =  close <= loww ? pocCol1 : pocCol2
    
    if removeCol
        if (open >= loww and open <= upp) 
             or (close >= loww and close <= upp) 
             or (hl2 >= loww and hl2 <= upp)
             or (hlc3 >= loww and hlc3 <= upp)
            pocCol := colorNA



    volBins.get(number) == volBins.max() ? color.new(pocCol, 20) : color.from_gradient(val, min, 1, colorNA, color.new(pocCol, 50))


for i = 0 to 31 - 1

    ll = L + step * i
    hh = ll + step
    mid = ll + step/2

    levels.set(i, ll)

    for k = 0 to period - 1

        c = close[k]

        if math.abs(c-mid) < step
            volBins.set(i, volBins.get(i) + volume[k])



if barstate.islast
    int binsCount = 31
    int lastIdx   = binsCount - 1
    float buyVol  = 0.0
    float sellVol = 0.0

    var boxes = array.new<box>()
    var labels = array.new<label>()

    for b in boxes
        b.delete()

    boxes.clear()

    for l in labels
        l.delete()

    labels.clear()

    // Find the boundary index where close sits inside the bin ladder
    // (clamped so it never goes out of range)
    int splitIdx = int(math.floor((close - L) / step))
    splitIdx := math.max(0, math.min(lastIdx, splitIdx))


    for i = 0 to lastIdx
        float mid = L + step * i + step/2
        float vol = volBins.get(i)

        if mid < close
            buyVol += vol
        else
            sellVol += vol

    float totalVol = buyVol + sellVol
    float buyPct   = totalVol > 0 ? buyVol  / totalVol : 0
    float sellPct  = totalVol > 0 ? sellVol / totalVol : 0
    float imbalance = sellVol > 0 ? buyVol - sellVol : na

    label buyLabel = label.new(
        bar_index + 40,
        L + (close - L)/2,
        "BUY LIQ\n" +
        str.tostring(buyVol, format.volume) + "\n" +
        str.tostring(buyPct * 100, "#.0") + "%",
        style = label.style_label_left,
        textcolor = pocCol2,
        color = colorNA,
        tooltip =
            "Liquidity Below Price\n\n" +
            "Total Volume Below Current Price:\n" +
            str.tostring(buyVol, format.volume) + "\n\n" +
            "Represents accumulated volume Below Current Price"
    )

    label sellLabel = label.new(
        bar_index + 40,
        close + (H - close)/2,
        "SELL LIQ\n" +
        str.tostring(sellVol, format.volume) + "\n" +
        str.tostring(sellPct * 100, "#.0") + "%",
        style = label.style_label_left,
        textcolor = pocCol1,
        color = colorNA,
        tooltip =
            "Liquidity Above Price\n\n" +
            "Total Volume Above Current Price:\n" +
            str.tostring(sellVol, format.volume) + "\n\n" +
            "Represents accumulated volume Above Current Price"
    )

    label ratioLabel = label.new(
        bar_index + 40,
        H + step,
        "Imbalance: " + str.tostring(imbalance, format.volume),
        style = label.style_label_left,
        textcolor = chart.fg_color,
        color = colorNA,
        tooltip =
            "Liquidity Imbalance\n\n" +
            "Difference Between Below and Above Liquidity:\n" +
            str.tostring(imbalance, format.volume) + "\n\n" +
            "Positive → More liquidity below price\n" +
            "Negative → More liquidity above price."
    )


    labels.push(buyLabel)
    labels.push(sellLabel)
    labels.push(ratioLabel)




    for i = 0 to lastIdx
        float ll  = L + step * i
        float hh  = ll + step
        float mid = ll + step / 2

        vol = volBins.get(i)

        // Your original color rule (based on close vs mid)
        color baseCol = close > mid ? pocCol2 : pocCol1

        // Distance from the split point (fade starts here)
        int dist = math.abs(i - splitIdx)

        // Max distance to ends (so fade scales correctly on both sides)
        int maxDist = math.max(splitIdx, lastIdx - splitIdx)
        float fade  = maxDist > 0 ? (dist / maxDist) : 0.0

        // Transparency: strong near split → more transparent toward ends
        int alpha = int(10 + fade * 85)   // 10 (strong) → 95 (faded)
        color col = color.new(baseCol, alpha)

        b = box.new(
            bar_index + 25, hh,
            bar_index + 35, ll,
            bgcolor = col,
            border_color = na
        )

        if volBins.get(i) == volBins.max()
            b.set_text(">"+ str.tostring(volBins.max(), format.volume) + "<")
       
            b.set_left(bar_index+10)
            b.set_bgcolor(close > mid ? pocCol2 : pocCol1)


        boxes.push(b)
// }



// ＰＬＯＴ ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――{

p0  = plot(levels.get(0), color = colorNA, editable = false)
p1  = plot(levels.get(1), color = colorNA, editable = false)
p2  = plot(levels.get(2), color = colorNA, editable = false)
p3  = plot(levels.get(3), color = colorNA, editable = false)
p4  = plot(levels.get(4), color = colorNA, editable = false)
p5  = plot(levels.get(5), color = colorNA, editable = false)
p6  = plot(levels.get(6), color = colorNA, editable = false)
p7  = plot(levels.get(7), color = colorNA, editable = false)
p8  = plot(levels.get(8), color = colorNA, editable = false)
p9  = plot(levels.get(9), color = colorNA, editable = false)
p10 = plot(levels.get(10), color = colorNA, editable = false)
p11 = plot(levels.get(11), color = colorNA, editable = false)
p12 = plot(levels.get(12), color = colorNA, editable = false)
p13 = plot(levels.get(13), color = colorNA, editable = false)
p14 = plot(levels.get(14), color = colorNA, editable = false)
p15 = plot(levels.get(15), color = colorNA, editable = false)
p16 = plot(levels.get(16), color = colorNA, editable = false)
p17 = plot(levels.get(17), color = colorNA, editable = false)
p18 = plot(levels.get(18), color = colorNA, editable = false)
p19 = plot(levels.get(19), color = colorNA, editable = false)
p20 = plot(levels.get(20), color = colorNA, editable = false)
p21 = plot(levels.get(21), color = colorNA, editable = false)
p22 = plot(levels.get(22), color = colorNA, editable = false)
p23 = plot(levels.get(23), color = colorNA, editable = false)
p24 = plot(levels.get(24), color = colorNA, editable = false)
p25 = plot(levels.get(25), color = colorNA, editable = false)
p26 = plot(levels.get(26), color = colorNA, editable = false)
p27 = plot(levels.get(27), color = colorNA, editable = false)
p28 = plot(levels.get(28), color = colorNA, editable = false)
p29 = plot(levels.get(29), color = colorNA, editable = false)
p30 = plot(levels.get(30), color = colorNA, editable = false)

HH = plot(H, color = colorNA, editable = false)

col0 = gradient(volBins, 0)
col1 = gradient(volBins, 1)
col2 = gradient(volBins, 2)
col3 = gradient(volBins, 3)
col4 = gradient(volBins, 4)
col5 = gradient(volBins, 5)
col6 = gradient(volBins, 6)
col7 = gradient(volBins, 7)
col8 = gradient(volBins, 8)
col9 = gradient(volBins, 9)
col10 = gradient(volBins, 10)
col11 = gradient(volBins, 11)
col12 = gradient(volBins, 12)
col13 = gradient(volBins, 13)
col14 = gradient(volBins, 14)
col15 = gradient(volBins, 15)
col16 = gradient(volBins, 16)
col17 = gradient(volBins, 17)
col18 = gradient(volBins, 18)
col19 = gradient(volBins, 19)
col20 = gradient(volBins, 20)
col21 = gradient(volBins, 21)
col22 = gradient(volBins, 22)
col23 = gradient(volBins, 23)
col24 = gradient(volBins, 24)
col25 = gradient(volBins, 25)
col26 = gradient(volBins, 26)
col27 = gradient(volBins, 27)
col28 = gradient(volBins, 28)
col29 = gradient(volBins, 29)
col30 = gradient(volBins, 30)



fill(p0, p1, col0)
fill(p1, p2, col1)
fill(p2, p3, col2)
fill(p3, p4, col3)
fill(p4, p5, col4)
fill(p5, p6, col5)
fill(p6, p7, col6)
fill(p7, p8, col7)
fill(p8, p9, col8)
fill(p9, p10, col9)
fill(p10, p11, col10)
fill(p11, p12, col11)
fill(p12, p13, col12)
fill(p13, p14, col13)
fill(p14, p15, col14)
fill(p15, p16, col15)
fill(p16, p17, col16)
fill(p17, p18, col17)
fill(p18, p19, col18)
fill(p19, p20, col19)
fill(p20, p21, col20)
fill(p21, p22, col21)
fill(p22, p23, col22)
fill(p23, p24, col23)
fill(p24, p25, col24)
fill(p25, p26, col25)
fill(p26, p27, col26)
fill(p27, p28, col27)
fill(p28, p29, col28)
fill(p29, p30, col29)
fill(p30, HH, col30)



if barstate.islast

    var lines = array.new<line>()

    for l in lines
        l.delete()

    lines.clear()

    for i = 0 to 29
        float level1 = levels.get(i)
        float level2 = levels.get(i + 1)

        line l1 = line.new(
            bar_index, level1,
            bar_index + 10, level1,
            color = colorNA
        )

        line l2 = line.new(
            bar_index, level2,
            bar_index + 10, level2,
            color = colorNA
        )

        lines.push(l1)
        lines.push(l2)

        linefill.new(l1, l2, gradient(volBins, i))

// }

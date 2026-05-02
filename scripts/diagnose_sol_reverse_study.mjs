import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { evaluate } from '../src/connection.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const STUDY_NAME = 'Crypto Donchian Regime Strategy MajDiv';
const TIMEFRAME = '60';
const SOL_SYMBOL = 'BINANCE:SOLUSDT';
const ETH_SYMBOL = 'BINANCE:ETHUSDT';

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function tv(args) {
  const out = execFileSync('node', ['src/cli/index.js', ...args], {
    cwd: repoRoot,
    encoding: 'utf8',
  });
  return JSON.parse(out);
}

function findStudyId() {
  const state = tv(['state']);
  const study = (state.studies || []).find(s => s.name === STUDY_NAME);
  if (!study) throw new Error(`Study not found on chart: ${STUDY_NAME}`);
  return study.id;
}

function getInputMap(studyId) {
  const data = tv(['indicator', 'get', studyId]);
  const map = {};
  for (const input of data.inputs || []) {
    if (typeof input.id === 'string' && /^in_\d+$/.test(input.id)) {
      map[input.id] = input.value;
    }
  }
  return map;
}

async function waitInputs(studyId, expected, maxAttempts = 8) {
  for (let i = 0; i < maxAttempts; i += 1) {
    const actual = getInputMap(studyId);
    const match = Object.entries(expected).every(([key, value]) => String(actual[key]) === String(value));
    if (match) return true;
    await sleep(1200);
  }
  return false;
}

async function getChartDataset(maxBars = 6000) {
  return evaluate(`
    (() => {
      try {
        const targetStudy = ${JSON.stringify(STUDY_NAME)};
        const chart = window.TradingViewApi._activeChartWidgetWV.value()._chartWidget;
        const model = chart.model();
        const sources = model.model().dataSources();
        let strat = null;
        for (const s of sources) {
          let meta = null;
          try { meta = s.metaInfo ? s.metaInfo() : null; } catch (e) {}
          const name = meta ? (meta.description || meta.shortDescription || '') : '';
          if (name === targetStudy) { strat = s; break; }
        }
        if (!strat) return { error: 'Strategy not found: ' + targetStudy };

        const bars = model.mainSeries().bars();
        if (!bars || typeof bars.lastIndex !== 'function') return { error: 'Chart bars unavailable' };

        const firstIndex = bars.firstIndex();
        const lastIndex = bars.lastIndex();
        const start = Math.max(firstIndex, lastIndex - ${maxBars} + 1);
        const resultBars = [];
        for (let i = start; i <= lastIndex; i += 1) {
          const v = bars.valueAt(i);
          if (v) {
            resultBars.push({
              index: i,
              time: v[0],
              open: v[1],
              high: v[2],
              low: v[3],
              close: v[4],
              volume: v[5] || 0
            });
          }
        }

        let rd = strat.reportData ? (typeof strat.reportData === 'function' ? strat.reportData() : strat.reportData) : null;
        if (rd && typeof rd.value === 'function') rd = rd.value();
        if (!rd || !rd.performance || !rd.performance.all) return { error: 'Strategy report unavailable' };

        const all = rd.performance.all;
        const trades = (rd.trades || []).map(t => ({
          side: t.e && t.e.c ? t.e.c : '',
          entryTime: t.e && t.e.tm ? t.e.tm : null,
          entryPrice: t.e && t.e.p ? t.e.p : null,
          exitTime: t.x && t.x.tm ? t.x.tm : null,
          exitPrice: t.x && t.x.p ? t.x.p : null,
          profit: t.tp && t.tp.v ? t.tp.v : 0,
          runUp: t.rn && t.rn.v ? t.rn.v : 0,
          drawdown: t.dd && t.dd.v ? t.dd.v : 0
        }));

        return {
          bars: resultBars,
          trades,
          metrics: {
            wr: all.percentProfitable != null ? +(all.percentProfitable * 100).toFixed(2) : null,
            pf: all.profitFactor != null ? +(+all.profitFactor).toFixed(3) : null,
            np: all.netProfit != null ? +(+all.netProfit).toFixed(2) : null,
            trades: all.totalTrades != null ? +all.totalTrades : 0
          }
        };
      } catch (e) {
        return { error: e.message };
      }
    })()
  `);
}

function toBinanceInterval(tf) {
  const map = {
    '1': '1m',
    '3': '3m',
    '5': '5m',
    '15': '15m',
    '30': '30m',
    '45': '45m',
    '60': '1h',
    '120': '2h',
    '180': '3h',
    '240': '4h',
    '360': '6h',
    '480': '8h',
    '720': '12h',
    D: '1d',
    '1D': '1d',
    W: '1w',
    '1W': '1w',
  };
  return map[String(tf)] || '1h';
}

async function fetchBinanceBars(symbol, interval, startTime, endTime) {
  const plainSymbol = symbol.includes(':') ? symbol.split(':')[1] : symbol;
  const bars = [];
  let cursor = startTime;

  while (cursor < endTime) {
    const url = new URL('https://api.binance.com/api/v3/klines');
    url.searchParams.set('symbol', plainSymbol);
    url.searchParams.set('interval', interval);
    url.searchParams.set('startTime', String(cursor));
    url.searchParams.set('endTime', String(endTime));
    url.searchParams.set('limit', '1000');

    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Binance klines error ${resp.status}`);
    const data = await resp.json();
    if (!Array.isArray(data) || !data.length) break;

    for (const row of data) {
      bars.push({
        time: Number(row[0]),
        open: Number(row[1]),
        high: Number(row[2]),
        low: Number(row[3]),
        close: Number(row[4]),
        volume: Number(row[5]),
      });
    }

    const lastOpenTime = Number(data[data.length - 1][0]);
    if (!Number.isFinite(lastOpenTime) || lastOpenTime <= cursor) break;
    cursor = lastOpenTime + 1;
    if (data.length < 1000) break;
    await sleep(120);
  }

  return bars;
}

function sma(values, length) {
  const out = Array(values.length).fill(null);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    sum += values[i] ?? 0;
    if (i >= length) sum -= values[i - length] ?? 0;
    if (i >= length - 1) out[i] = sum / length;
  }
  return out;
}

function ema(values, length) {
  const out = Array(values.length).fill(null);
  const alpha = 2 / (length + 1);
  let prev = null;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    if (v == null) continue;
    if (prev == null) prev = v;
    else prev = alpha * v + (1 - alpha) * prev;
    out[i] = prev;
  }
  return out;
}

function rma(values, length) {
  const out = Array(values.length).fill(null);
  let prev = null;
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i] ?? 0;
    if (i < length) {
      sum += v;
      if (i === length - 1) {
        prev = sum / length;
        out[i] = prev;
      }
      continue;
    }
    prev = ((prev * (length - 1)) + v) / length;
    out[i] = prev;
  }
  return out;
}

function rollingStd(values, length) {
  const out = Array(values.length).fill(null);
  for (let i = length - 1; i < values.length; i += 1) {
    let sum = 0;
    let sumSq = 0;
    for (let j = i - length + 1; j <= i; j += 1) {
      const v = values[j];
      sum += v;
      sumSq += v * v;
    }
    const mean = sum / length;
    const variance = Math.max(sumSq / length - mean * mean, 0);
    out[i] = Math.sqrt(variance);
  }
  return out;
}

function highest(values, start, end) {
  let max = -Infinity;
  for (let i = start; i <= end; i += 1) {
    if (values[i] > max) max = values[i];
  }
  return Number.isFinite(max) ? max : null;
}

function lowest(values, start, end) {
  let min = Infinity;
  for (let i = start; i <= end; i += 1) {
    if (values[i] < min) min = values[i];
  }
  return Number.isFinite(min) ? min : null;
}

function percentileRank(values, index, length) {
  const start = Math.max(0, index - length);
  if (index - start < Math.max(5, Math.floor(length * 0.6))) return null;
  const sample = [];
  for (let i = start; i < index; i += 1) sample.push(values[i]);
  const current = values[index];
  let count = 0;
  for (const v of sample) if (v <= current) count += 1;
  return (count / sample.length) * 100;
}

function median(values) {
  const filtered = values.filter(v => Number.isFinite(v)).sort((a, b) => a - b);
  if (!filtered.length) return null;
  const mid = Math.floor(filtered.length / 2);
  if (filtered.length % 2) return filtered[mid];
  return (filtered[mid - 1] + filtered[mid]) / 2;
}

function mean(values) {
  const filtered = values.filter(v => Number.isFinite(v));
  if (!filtered.length) return null;
  return filtered.reduce((a, b) => a + b, 0) / filtered.length;
}

function pct(num, den) {
  if (!den) return null;
  return (num / den) * 100;
}

function formatNumber(value, digits = 2) {
  if (value == null || !Number.isFinite(value)) return 'n/a';
  return value.toFixed(digits);
}

function sessionName(date) {
  const hour = date.getUTCHours();
  if (hour < 8) return 'Asia';
  if (hour < 16) return 'Europe';
  return 'US';
}

function addTechnicalSeries(bars) {
  const high = bars.map(b => b.high);
  const low = bars.map(b => b.low);
  const close = bars.map(b => b.close);
  const volume = bars.map(b => b.volume);
  const hlc3 = bars.map(b => (b.high + b.low + b.close) / 3);

  const emaFast = ema(close, 20);
  const emaSlow = ema(close, 50);

  const tr = Array(bars.length).fill(null);
  const plusDm = Array(bars.length).fill(0);
  const minusDm = Array(bars.length).fill(0);
  const trueRange = Array(bars.length).fill(null);
  const upBar = Array(bars.length).fill(0);
  const downBar = Array(bars.length).fill(0);
  const gains = Array(bars.length).fill(0);
  const losses = Array(bars.length).fill(0);

  for (let i = 0; i < bars.length; i += 1) {
    if (i === 0) {
      tr[i] = high[i] - low[i];
      trueRange[i] = tr[i];
      continue;
    }
    const upMove = high[i] - high[i - 1];
    const downMove = low[i - 1] - low[i];
    plusDm[i] = upMove > downMove && upMove > 0 ? upMove : 0;
    minusDm[i] = downMove > upMove && downMove > 0 ? downMove : 0;
    tr[i] = Math.max(high[i] - low[i], Math.max(Math.abs(high[i] - close[i - 1]), Math.abs(low[i] - close[i - 1])));
    trueRange[i] = tr[i];
    upBar[i] = close[i] > close[i - 1] ? 1 : 0;
    downBar[i] = close[i] < close[i - 1] ? 1 : 0;
    const ch = close[i] - close[i - 1];
    gains[i] = Math.max(ch, 0);
    losses[i] = Math.max(-ch, 0);
  }

  const atr = rma(tr, 14);
  const plusDmRma = rma(plusDm, 14);
  const minusDmRma = rma(minusDm, 14);
  const plusDi = Array(bars.length).fill(null);
  const minusDi = Array(bars.length).fill(null);
  const dx = Array(bars.length).fill(null);
  for (let i = 0; i < bars.length; i += 1) {
    if (!atr[i] || atr[i] === 0) continue;
    plusDi[i] = 100 * plusDmRma[i] / atr[i];
    minusDi[i] = 100 * minusDmRma[i] / atr[i];
    const denom = plusDi[i] + minusDi[i];
    dx[i] = denom ? 100 * Math.abs(plusDi[i] - minusDi[i]) / denom : null;
  }
  const adx = rma(dx.map(v => v ?? 0), 14);

  const chop = Array(bars.length).fill(null);
  const chopLen = 18;
  for (let i = chopLen - 1; i < bars.length; i += 1) {
    let trSum = 0;
    let hh = -Infinity;
    let ll = Infinity;
    for (let j = i - chopLen + 1; j <= i; j += 1) {
      trSum += trueRange[j] ?? 0;
      if (high[j] > hh) hh = high[j];
      if (low[j] < ll) ll = low[j];
    }
    const range = Math.max(hh - ll, Number.EPSILON);
    chop[i] = 100 * Math.log10(trSum / range) / Math.log10(chopLen);
  }

  const upProb = sma(upBar, 24);
  const downProb = sma(downBar, 24);
  const entropy = Array(bars.length).fill(null);
  for (let i = 0; i < bars.length; i += 1) {
    if (upProb[i] == null || downProb[i] == null) continue;
    const total = Math.max(upProb[i] + downProb[i], 1e-10);
    const upNorm = upProb[i] / total;
    const downNorm = downProb[i] / total;
    const termUp = upNorm > 0 ? -upNorm * Math.log(upNorm) : 0;
    const termDown = downNorm > 0 ? -downNorm * Math.log(downNorm) : 0;
    entropy[i] = (termUp + termDown) / Math.log(2);
  }

  const gainRma = rma(gains, 22);
  const lossRma = rma(losses, 22);
  const rsi22 = Array(bars.length).fill(null);
  for (let i = 0; i < bars.length; i += 1) {
    if (gainRma[i] == null || lossRma[i] == null) continue;
    if (lossRma[i] === 0) rsi22[i] = 100;
    else if (gainRma[i] === 0) rsi22[i] = 0;
    else {
      const rs = gainRma[i] / lossRma[i];
      rsi22[i] = 100 - (100 / (1 + rs));
    }
  }

  const bbBasis = sma(close, 20);
  const bbStd = rollingStd(close, 20);
  const bbPct = Array(bars.length).fill(null);
  for (let i = 0; i < bars.length; i += 1) {
    if (bbBasis[i] == null || bbStd[i] == null) continue;
    const upper = bbBasis[i] + 2 * bbStd[i];
    const lower = bbBasis[i] - 2 * bbStd[i];
    bbPct[i] = upper > lower ? 100 * (close[i] - lower) / (upper - lower) : 50;
  }

  const sessionVwap = Array(bars.length).fill(null);
  const sessionVwapDev = Array(bars.length).fill(null);
  let curDay = null;
  let cumVol = 0;
  let cumPv = 0;
  let cumPv2 = 0;
  for (let i = 0; i < bars.length; i += 1) {
    const date = new Date(bars[i].time);
    const dayKey = `${date.getUTCFullYear()}-${date.getUTCMonth()}-${date.getUTCDate()}`;
    if (dayKey !== curDay) {
      curDay = dayKey;
      cumVol = 0;
      cumPv = 0;
      cumPv2 = 0;
    }
    const v = Math.max(volume[i] ?? 0, 0);
    const p = hlc3[i];
    cumVol += v;
    cumPv += v * p;
    cumPv2 += v * p * p;
    if (cumVol > 0) {
      const meanValue = cumPv / cumVol;
      const variance = Math.max(cumPv2 / cumVol - meanValue * meanValue, 0);
      const stdev = Math.sqrt(variance);
      sessionVwap[i] = meanValue;
      sessionVwapDev[i] = stdev > Number.EPSILON ? (close[i] - meanValue) / stdev : 0;
    }
  }

  return {
    emaFast,
    emaSlow,
    atr,
    adx,
    chop,
    entropy,
    rsi22,
    bbPct,
    sessionVwap,
    sessionVwapDev,
    close,
    high,
    low,
    volume,
  };
}

function buildTradeFeatures(dataset) {
  const { bars, trades } = dataset;
  const series = addTechnicalSeries(bars);
  const timeToIndex = new Map(bars.map((bar, idx) => [bar.time, idx]));
  const rows = [];
  let skipped = 0;

  for (const trade of trades) {
    const entryIdx = timeToIndex.get(trade.entryTime);
    if (entryIdx == null || entryIdx < 31) {
      skipped += 1;
      continue;
    }

    const signalIdx = entryIdx - 1;
    const signalBar = bars[signalIdx];
    const entryBar = bars[entryIdx];
    const side = trade.side.startsWith('Long') ? 'Long' : 'Short';
    const dir = side === 'Long' ? 1 : -1;
    const atr = series.atr[signalIdx];
    if (!atr || !Number.isFinite(atr)) {
      skipped += 1;
      continue;
    }

    const prevHigh = highest(series.high, signalIdx - 30, signalIdx - 1);
    const prevLow = lowest(series.low, signalIdx - 30, signalIdx - 1);
    const breakoutExcessAtr = side === 'Long'
      ? (signalBar.close - prevHigh - atr * 0.05) / atr
      : (prevLow - signalBar.close - atr * 0.05) / atr;
    const donchianWidthAtr = (prevHigh - prevLow) / atr;
    const range = Math.max(signalBar.high - signalBar.low, Number.EPSILON);
    const bodyFrac = Math.abs(signalBar.close - signalBar.open) / range;
    const closePos = (signalBar.close - signalBar.low) / range;
    const dirClosePos = side === 'Long' ? closePos : 1 - closePos;
    const volPct20 = percentileRank(series.volume, signalIdx, 20);
    const volPct50 = percentileRank(series.volume, signalIdx, 50);
    const volPct100 = percentileRank(series.volume, signalIdx, 100);
    const volumeSma20 = mean(series.volume.slice(Math.max(0, signalIdx - 20), signalIdx));
    const volRatio20 = volumeSma20 ? signalBar.volume / volumeSma20 : null;
    const dirReturn3Atr = signalIdx >= 3 ? dir * (signalBar.close - series.close[signalIdx - 3]) / atr : null;
    const dirReturn6Atr = signalIdx >= 6 ? dir * (signalBar.close - series.close[signalIdx - 6]) / atr : null;
    const dirReturn12Atr = signalIdx >= 12 ? dir * (signalBar.close - series.close[signalIdx - 12]) / atr : null;
    const gapDirAtr = dir * ((trade.entryPrice ?? entryBar.open) - signalBar.close) / atr;
    const dirEmaFastDistAtr = dir * (signalBar.close - series.emaFast[signalIdx]) / atr;
    const dirEmaSlowDistAtr = dir * (signalBar.close - series.emaSlow[signalIdx]) / atr;
    const dirVwapExt = side === 'Long' ? series.sessionVwapDev[signalIdx] : -series.sessionVwapDev[signalIdx];
    const dirBbExt = side === 'Long' ? series.bbPct[signalIdx] : 100 - series.bbPct[signalIdx];
    const signalDate = new Date(signalBar.time);
    const hoursHeld = trade.exitTime && trade.entryTime ? (trade.exitTime - trade.entryTime) / 36e5 : null;

    let maxFav3 = null;
    let close1 = null;
    const last3 = Math.min(entryIdx + 2, bars.length - 1);
    if (entryIdx < bars.length) {
      let best3 = -Infinity;
      for (let i = entryIdx; i <= last3; i += 1) {
        const fav = dir === 1 ? (bars[i].high - trade.entryPrice) / atr : (trade.entryPrice - bars[i].low) / atr;
        if (fav > best3) best3 = fav;
      }
      maxFav3 = best3;
      close1 = dir * (bars[entryIdx].close - trade.entryPrice) / atr;
    }

    rows.push({
      side,
      isLoss: trade.profit < 0,
      profit: trade.profit,
      runUp: trade.runUp,
      drawdown: trade.drawdown,
      hoursHeld,
      session: sessionName(signalDate),
      weekday: signalDate.toLocaleDateString('en-US', { weekday: 'long', timeZone: 'UTC' }),
      adx: series.adx[signalIdx],
      chop: series.chop[signalIdx],
      entropy: series.entropy[signalIdx],
      rsi22: series.rsi22[signalIdx],
      rangeAtr: range / atr,
      bodyFrac,
      dirClosePos,
      volPct20,
      volPct50,
      volPct100,
      volRatio20,
      dirReturn3Atr,
      dirReturn6Atr,
      dirReturn12Atr,
      gapDirAtr,
      dirEmaFastDistAtr,
      dirEmaSlowDistAtr,
      dirVwapExt,
      dirBbExt,
      breakoutExcessAtr,
      donchianWidthAtr,
      close1Atr: close1,
      maxFav3Atr: maxFav3,
    });
  }

  return { rows, skipped };
}

function summarizeGroup(rows) {
  const losses = rows.filter(r => r.isLoss);
  return {
    trades: rows.length,
    lossRate: pct(losses.length, rows.length),
    wr: pct(rows.length - losses.length, rows.length),
    np: rows.reduce((sum, r) => sum + r.profit, 0),
  };
}

function metricContrast(rows, feature) {
  const wins = rows.filter(r => !r.isLoss).map(r => r[feature]);
  const losses = rows.filter(r => r.isLoss).map(r => r[feature]);
  return {
    feature,
    winMedian: median(wins),
    lossMedian: median(losses),
    winMean: mean(wins),
    lossMean: mean(losses),
  };
}

const candidateDefs = [
  { name: 'Shorts', test: r => r.side === 'Short' },
  { name: 'Longs', test: r => r.side === 'Long' },
  { name: 'Asia', test: r => r.session === 'Asia' },
  { name: 'Europe', test: r => r.session === 'Europe' },
  { name: 'US', test: r => r.session === 'US' },
  { name: 'ADX >= 30', test: r => r.adx >= 30 },
  { name: 'ADX < 20', test: r => r.adx < 20 },
  { name: 'CHOP > 50', test: r => r.chop > 50 },
  { name: 'CHOP 45-50', test: r => r.chop >= 45 && r.chop <= 50 },
  { name: 'Entropy > 0.97', test: r => r.entropy > 0.97 },
  { name: 'RSI22 > 70', test: r => r.rsi22 > 70 },
  { name: 'RSI22 < 30', test: r => r.rsi22 < 30 },
  { name: 'Vol pct >= 90', test: r => r.volPct50 >= 90 },
  { name: 'Vol pct < 30', test: r => r.volPct50 < 30 },
  { name: 'Vol ratio < 0.9', test: r => r.volRatio20 < 0.9 },
  { name: 'Vol ratio >= 1.5', test: r => r.volRatio20 >= 1.5 },
  { name: 'Return6 >= 2 ATR', test: r => r.dirReturn6Atr >= 2 },
  { name: 'Return6 < 0.5 ATR', test: r => r.dirReturn6Atr < 0.5 },
  { name: 'Gap chase > 0.15 ATR', test: r => r.gapDirAtr > 0.15 },
  { name: 'EMA fast dist > 1.0 ATR', test: r => r.dirEmaFastDistAtr > 1.0 },
  { name: 'VWAP ext > 1.5', test: r => r.dirVwapExt > 1.5 },
  { name: 'BB ext > 90', test: r => r.dirBbExt > 90 },
  { name: 'Breakout excess <= 0.10', test: r => r.breakoutExcessAtr <= 0.10 },
  { name: 'Breakout excess > 0.60', test: r => r.breakoutExcessAtr > 0.60 },
  { name: 'Donchian width < 4 ATR', test: r => r.donchianWidthAtr < 4 },
  { name: 'Close1 adverse', test: r => r.close1Atr < 0 },
  { name: 'MaxFav3 < 0.5 ATR', test: r => r.maxFav3Atr < 0.5 },
  { name: 'MaxFav3 0.5-1.0 ATR', test: r => r.maxFav3Atr >= 0.5 && r.maxFav3Atr < 1.0 },
  { name: 'MaxFav3 >= 2 ATR', test: r => r.maxFav3Atr >= 2.0 },
  { name: 'Short high-vol extension', test: r => r.side === 'Short' && r.volPct50 >= 90 && r.dirVwapExt > 1.5 },
  { name: 'Short late extension', test: r => r.side === 'Short' && r.adx >= 30 && r.dirVwapExt > 1.5 },
  { name: 'High ext + high BB', test: r => r.dirVwapExt > 1.5 && r.dirBbExt > 90 },
  { name: 'Weak first bar + low MFE', test: r => r.close1Atr < 0 && r.maxFav3Atr < 0.5 },
];

function findNotablePatterns(rows) {
  const baseline = summarizeGroup(rows);
  const patterns = [];
  for (const def of candidateDefs) {
    const subset = rows.filter(def.test);
    if (subset.length < 5) continue;
    const summary = summarizeGroup(subset);
    const lossEdge = (summary.lossRate ?? 0) - (baseline.lossRate ?? 0);
    const wrDrop = (baseline.wr ?? 0) - (summary.wr ?? 0);
    if (lossEdge >= 8 || summary.np < 0) {
      patterns.push({
        name: def.name,
        trades: summary.trades,
        lossRate: summary.lossRate,
        wr: summary.wr,
        np: summary.np,
        lossEdge,
        wrDrop,
        test: def.test,
      });
    }
  }
  return patterns.sort((a, b) => {
    if (b.lossEdge !== a.lossEdge) return b.lossEdge - a.lossEdge;
    if (a.np !== b.np) return a.np - b.np;
    return b.trades - a.trades;
  });
}

async function analyzeSymbol(symbol) {
  tv(['timeframe', TIMEFRAME]);
  await sleep(1200);
  tv(['symbol', symbol]);
  await sleep(1800);
  await sleep(1800);

  const dataset = await getChartDataset();
  if (dataset.error) throw new Error(dataset.error);

  const earliestTrade = Math.min(...dataset.trades.map(t => t.entryTime).filter(Boolean));
  const latestTrade = Math.max(...dataset.trades.map(t => t.entryTime).filter(Boolean));
  const earliestBar = dataset.bars[0]?.time;
  const latestBar = dataset.bars[dataset.bars.length - 1]?.time;
  if (earliestTrade < earliestBar || latestTrade > latestBar) {
    const fetchedBars = await fetchBinanceBars(
      symbol,
      toBinanceInterval(TIMEFRAME),
      earliestTrade - 30 * 24 * 60 * 60 * 1000,
      latestTrade + 7 * 24 * 60 * 60 * 1000,
    );
    if (!fetchedBars.length) throw new Error(`Could not fetch Binance fallback history for ${symbol}.`);
    dataset.bars = fetchedBars;
  }

  const { rows, skipped } = buildTradeFeatures(dataset);
  return { symbol, metrics: dataset.metrics, rows, skipped };
}

function printSummary(label, analysis) {
  const summary = summarizeGroup(analysis.rows);
  console.log(`\n${label}`);
  console.log(`Symbol ${analysis.symbol} | TF ${TIMEFRAME} | trades ${analysis.metrics.trades} | WR ${analysis.metrics.wr}% | PF ${analysis.metrics.pf} | NP ${analysis.metrics.np}`);
  console.log(`Feature rows analyzed ${analysis.rows.length} | skipped ${analysis.skipped} | loss rate ${formatNumber(summary.lossRate, 2)}%`);

  console.log('\nWin vs loss contrasts');
  [
    'adx',
    'chop',
    'entropy',
    'rsi22',
    'volPct50',
    'volRatio20',
    'dirReturn6Atr',
    'gapDirAtr',
    'dirEmaFastDistAtr',
    'dirVwapExt',
    'dirBbExt',
    'breakoutExcessAtr',
    'donchianWidthAtr',
    'close1Atr',
    'maxFav3Atr',
  ].forEach(feature => {
    const c = metricContrast(analysis.rows, feature);
    console.log(`${feature.padEnd(18)} wins median ${formatNumber(c.winMedian, 3)} | losses median ${formatNumber(c.lossMedian, 3)} | wins mean ${formatNumber(c.winMean, 3)} | losses mean ${formatNumber(c.lossMean, 3)}`);
  });
}

function printPatternComparison(solAnalysis, ethAnalysis, patterns) {
  const solBase = summarizeGroup(solAnalysis.rows);
  const ethBase = summarizeGroup(ethAnalysis.rows);

  console.log('\nTop SOL danger patterns and ETH cross-check');
  for (const pattern of patterns.slice(0, 10)) {
    const solRows = solAnalysis.rows.filter(pattern.test);
    const ethRows = ethAnalysis.rows.filter(pattern.test);
    const solSummary = summarizeGroup(solRows);
    const ethSummary = summarizeGroup(ethRows);
    const generalizes = ethRows.length >= 5 && (ethSummary.lossRate ?? 0) >= (ethBase.lossRate ?? 0) + 8;
    console.log(`\n${pattern.name}`);
    console.log(`  SOL trades ${solSummary.trades} | loss ${formatNumber(solSummary.lossRate, 1)}% vs base ${formatNumber(solBase.lossRate, 1)}% | net ${formatNumber(solSummary.np, 2)}${generalizes ? ' | generalizes to ETH' : ' | SOL-specific / weak on ETH'}`);
    if (ethRows.length >= 1) {
      console.log(`  ETH trades ${ethSummary.trades} | loss ${formatNumber(ethSummary.lossRate, 1)}% vs base ${formatNumber(ethBase.lossRate, 1)}% | net ${formatNumber(ethSummary.np, 2)}`);
    } else {
      console.log('  ETH trades 0 | no comparable sample');
    }
  }
}

function printGeneralizedPatterns(solAnalysis, ethAnalysis, patterns) {
  const solBase = summarizeGroup(solAnalysis.rows);
  const ethBase = summarizeGroup(ethAnalysis.rows);
  const generalized = [];
  const solOnly = [];

  for (const pattern of patterns) {
    const solRows = solAnalysis.rows.filter(pattern.test);
    const ethRows = ethAnalysis.rows.filter(pattern.test);
    const solSummary = summarizeGroup(solRows);
    const ethSummary = summarizeGroup(ethRows);
    const solBad = solSummary.trades >= 5 && (solSummary.lossRate ?? 0) >= (solBase.lossRate ?? 0) + 8;
    const ethBad = ethSummary.trades >= 5 && (ethSummary.lossRate ?? 0) >= (ethBase.lossRate ?? 0) + 8;
    if (solBad && ethBad) generalized.push({ name: pattern.name, sol: solSummary, eth: ethSummary });
    else if (solBad) solOnly.push({ name: pattern.name, sol: solSummary, eth: ethSummary });
  }

  console.log('\nGeneralized patterns (SOL -> ETH)');
  if (!generalized.length) console.log('  None cleared the current threshold cleanly.');
  for (const row of generalized.slice(0, 6)) {
    console.log(`  ${row.name.padEnd(24)} SOL loss ${formatNumber(row.sol.lossRate, 1)}% | ETH loss ${formatNumber(row.eth.lossRate, 1)}%`);
  }

  console.log('\nSOL-only or mostly SOL-specific patterns');
  if (!solOnly.length) console.log('  None stood out as strongly SOL-specific.');
  for (const row of solOnly.slice(0, 6)) {
    console.log(`  ${row.name.padEnd(24)} SOL loss ${formatNumber(row.sol.lossRate, 1)}% | ETH loss ${formatNumber(row.eth.lossRate, 1)}%`);
  }
}

async function main() {
  const studyId = findStudyId();
  const activeInputs = getInputMap(studyId);

  console.error('Analyzing current active profile on SOL...');
  const solAnalysis = await analyzeSymbol(SOL_SYMBOL);
  printSummary('SOL reverse study', solAnalysis);

  const patterns = findNotablePatterns(solAnalysis.rows);
  console.log('\nTop SOL pattern candidates');
  patterns.slice(0, 12).forEach(pattern => {
    console.log(`  ${pattern.name.padEnd(24)} trades ${String(pattern.trades).padStart(3)} | loss ${formatNumber(pattern.lossRate, 1)}% | loss edge ${formatNumber(pattern.lossEdge, 1)} | net ${formatNumber(pattern.np, 2)}`);
  });

  console.error('Cross-checking the same active profile on ETH...');
  const ethAnalysis = await analyzeSymbol(ETH_SYMBOL);
  printSummary('ETH cross-check', ethAnalysis);

  printPatternComparison(solAnalysis, ethAnalysis, patterns);
  printGeneralizedPatterns(solAnalysis, ethAnalysis, patterns);

  console.log('\nActive profile snapshot');
  console.log(`  study_id ${studyId}`);
  console.log(`  entry_mode ${activeInputs.in_48}`);
  console.log(`  channel_mode ${activeInputs.in_32}`);
  console.log(`  be_protect ${activeInputs.in_154}`);
  console.log(`  be_trigger_atr ${activeInputs.in_156}`);
  console.log(`  be_offset_atr ${activeInputs.in_157}`);
  console.log(`  profit_lock ${activeInputs.in_158}`);
  console.log(`  limit_pullback_frac ${activeInputs.in_161}`);
}

main()
  .then(() => process.exit(0))
  .catch(err => {
    console.error(err?.stack || err?.message || String(err));
    process.exit(1);
  });

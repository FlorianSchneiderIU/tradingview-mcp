import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import * as indCore from '../src/core/indicators.js';
import { evaluate } from '../src/connection.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const STUDY_NAME = 'Crypto Donchian Regime Strategy MajDiv';
const SYMBOL = 'BINANCE:ETHUSDT';
const TIMEFRAME = '60';

const baseInputs = {
  in_32: 'Auto Regime',
  in_38: 0.20,
  in_39: 0.35,
  in_40: 0.60,
  in_41: 2.0,
  in_42: 58.0,
  in_43: false,
  in_44: true,
  in_45: 3,
  in_46: false,
  in_47: 1.0,
  in_60: true,
  in_61: 24,
  in_62: 0.95,
  in_63: 0.98,
  in_64: true,
  in_65: 'Reject Only',
  in_66: '60',
  in_67: '240',
  in_68: 2,
  in_69: 2,
  in_70: 0.50,
  in_71: 0.50,
  in_72: true,
  in_73: 0.21,
  in_74: 0.40,
  in_75: 0.60,
  in_76: 0.79,
  in_77: true,
  in_78: 2,
  in_79: 2,
  in_80: 4,
  in_81: false,
  in_82: false,
  in_95: false,
  in_107: false,
  in_113: false,
  in_120: false,
  in_126: false,
  in_131: false,
};

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
    map[input.id] = input.value;
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
    'D': '1d',
    '1D': '1d',
    'W': '1w',
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
  const open = bars.map(b => b.open);
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
      const mean = cumPv / cumVol;
      const variance = Math.max(cumPv2 / cumVol - mean * mean, 0);
      const stdev = Math.sqrt(variance);
      sessionVwap[i] = mean;
      sessionVwapDev[i] = stdev > Number.EPSILON ? (close[i] - mean) / stdev : 0;
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
    open,
    volume,
  };
}

function buildTradeFeatures(dataset) {
  const { bars, trades, metrics } = dataset;
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
    let maxAdv3 = null;
    let close3 = null;
    let maxFav6 = null;
    let maxAdv6 = null;
    let close1 = null;
    const last3 = Math.min(entryIdx + 2, bars.length - 1);
    const last6 = Math.min(entryIdx + 5, bars.length - 1);
    if (entryIdx < bars.length) {
      let best3 = -Infinity;
      let worst3 = Infinity;
      let best6 = -Infinity;
      let worst6 = Infinity;
      for (let i = entryIdx; i <= last6; i += 1) {
        const fav = dir === 1 ? (bars[i].high - trade.entryPrice) / atr : (trade.entryPrice - bars[i].low) / atr;
        const adv = dir === 1 ? (trade.entryPrice - bars[i].low) / atr : (bars[i].high - trade.entryPrice) / atr;
        if (i <= last3) {
          if (fav > best3) best3 = fav;
          if (adv < worst3) worst3 = adv;
        }
        if (fav > best6) best6 = fav;
        if (adv < worst6) worst6 = adv;
      }
      maxFav3 = best3;
      maxAdv3 = worst3;
      maxFav6 = best6;
      maxAdv6 = worst6;
      close1 = entryIdx < bars.length ? dir * (bars[entryIdx].close - trade.entryPrice) / atr : null;
      close3 = dir * (bars[last3].close - trade.entryPrice) / atr;
    }

    rows.push({
      side,
      dir,
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
      close3Atr: close3,
      maxFav3Atr: maxFav3,
      maxAdv3Atr: maxAdv3,
      maxFav6Atr: maxFav6,
      maxAdv6Atr: maxAdv6,
    });
  }

  return { rows, metrics, skipped };
}

function summarizeGroup(rows, label) {
  const losses = rows.filter(r => r.isLoss);
  return {
    label,
    trades: rows.length,
    lossRate: pct(losses.length, rows.length),
    wr: pct(rows.length - losses.length, rows.length),
    np: rows.reduce((sum, r) => sum + r.profit, 0),
  };
}

function printMetricContrast(rows, feature) {
  const wins = rows.filter(r => !r.isLoss).map(r => r[feature]);
  const losses = rows.filter(r => r.isLoss).map(r => r[feature]);
  console.log(
    `${feature.padEnd(18)} wins median ${formatNumber(median(wins), 3)} | losses median ${formatNumber(median(losses), 3)} | wins mean ${formatNumber(mean(wins), 3)} | losses mean ${formatNumber(mean(losses), 3)}`
  );
}

function printBucketAnalysis(rows, feature, buckets) {
  console.log(`\n${feature}`);
  for (const bucket of buckets) {
    const subset = rows.filter(r => bucket.test(r[feature]));
    if (!subset.length) continue;
    const summary = summarizeGroup(subset, bucket.name);
    console.log(`  ${bucket.name.padEnd(16)} trades ${String(summary.trades).padStart(2)} | loss ${formatNumber(summary.lossRate, 1)}% | net ${formatNumber(summary.np, 2)} | WR ${formatNumber(summary.wr, 1)}%`);
  }
}

function printCategoricalAnalysis(rows, feature) {
  console.log(`\n${feature}`);
  const groups = new Map();
  for (const row of rows) {
    const key = row[feature];
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  for (const [key, subset] of [...groups.entries()].sort((a, b) => b[1].length - a[1].length)) {
    const summary = summarizeGroup(subset, key);
    console.log(`  ${String(key).padEnd(16)} trades ${String(summary.trades).padStart(2)} | loss ${formatNumber(summary.lossRate, 1)}% | net ${formatNumber(summary.np, 2)} | WR ${formatNumber(summary.wr, 1)}%`);
  }
}

function printComboAnalysis(rows) {
  const combos = [
    {
      name: 'High ext + high BB',
      test: r => r.dirVwapExt > 1.5 && r.dirBbExt > 90,
    },
    {
      name: 'High vol + extension',
      test: r => r.volPct50 >= 90 && r.dirBbExt >= 110,
    },
    {
      name: 'Short late extension',
      test: r => r.side === 'Short' && r.adx >= 30 && r.dirVwapExt > 1.5,
    },
    {
      name: 'Short high-vol extension',
      test: r => r.side === 'Short' && r.volPct50 >= 90 && r.dirVwapExt > 1.5,
    },
    {
      name: 'High ext + gap chase',
      test: r => r.dirVwapExt > 1.0 && r.gapDirAtr > 0.15,
    },
    {
      name: 'Low vol + breakout far',
      test: r => r.volPct50 < 30 && r.breakoutExcessAtr > 0.25,
    },
    {
      name: 'High chop + extension',
      test: r => r.chop > 50 && r.dirBbExt > 90,
    },
    {
      name: 'Weak first bar',
      test: r => r.close1Atr < 0,
    },
    {
      name: 'No 3-bar follow-through',
      test: r => r.maxFav3Atr < 0.5,
    },
  ];
  console.log('\nCombination checks');
  for (const combo of combos) {
    const subset = rows.filter(combo.test);
    if (subset.length < 5) continue;
    const summary = summarizeGroup(subset, combo.name);
    console.log(`  ${combo.name.padEnd(22)} trades ${String(summary.trades).padStart(2)} | loss ${formatNumber(summary.lossRate, 1)}% | net ${formatNumber(summary.np, 2)} | WR ${formatNumber(summary.wr, 1)}%`);
  }
}

async function main() {
  console.error('Setting ETH baseline context...');
  tv(['timeframe', TIMEFRAME]);
  await sleep(1200);
  tv(['symbol', SYMBOL]);
  await sleep(1800);

  const studyId = findStudyId();
  await indCore.setInputs({ entity_id: studyId, inputs: baseInputs });
  const settled = await waitInputs(studyId, baseInputs);
  if (!settled) throw new Error('Baseline inputs did not settle on the live study.');
  await sleep(2200);

  console.error('Fetching bars and trades...');
  const dataset = await getChartDataset();
  if (dataset.error) throw new Error(dataset.error);

  const earliestTrade = Math.min(...dataset.trades.map(t => t.entryTime).filter(Boolean));
  const latestTrade = Math.max(...dataset.trades.map(t => t.entryTime).filter(Boolean));
  const earliestBar = dataset.bars[0]?.time;
  const latestBar = dataset.bars[dataset.bars.length - 1]?.time;
  if (earliestTrade < earliestBar || latestTrade > latestBar) {
    console.error('Chart history is too shallow, fetching Binance bars for the full trade window...');
    const fetchedBars = await fetchBinanceBars(
      SYMBOL,
      toBinanceInterval(TIMEFRAME),
      earliestTrade - 30 * 24 * 60 * 60 * 1000,
      latestTrade + 7 * 24 * 60 * 60 * 1000,
    );
    if (!fetchedBars.length) throw new Error('Could not fetch Binance fallback history.');
    dataset.bars = fetchedBars;
  }

  console.error(`Fetched ${dataset.bars.length} bars and ${dataset.trades.length} trades.`);
  const { rows, metrics, skipped } = buildTradeFeatures(dataset);
  console.error(`Built ${rows.length} feature rows.`);
  const losses = rows.filter(r => r.isLoss);
  const wins = rows.filter(r => !r.isLoss);

  console.log('\nETH baseline loss study');
  console.log(`Symbol ${SYMBOL} | TF ${TIMEFRAME} | trades ${metrics.trades} | WR ${metrics.wr}% | PF ${metrics.pf} | NP ${metrics.np}`);
  console.log(`Feature rows analyzed ${rows.length} | skipped ${skipped}`);
  console.log(`Losses ${losses.length} | Wins ${wins.length} | Loss rate ${formatNumber(pct(losses.length, rows.length), 2)}%`);

  console.log('\nWin vs loss contrasts');
  [
    'adx',
    'chop',
    'entropy',
    'rsi22',
    'rangeAtr',
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
    'close3Atr',
  ].forEach(feature => printMetricContrast(rows, feature));

  printCategoricalAnalysis(rows, 'side');
  printCategoricalAnalysis(rows, 'session');
  printCategoricalAnalysis(rows, 'weekday');

  printBucketAnalysis(rows, 'adx', [
    { name: '<20', test: v => v != null && v < 20 },
    { name: '20-25', test: v => v != null && v >= 20 && v < 25 },
    { name: '25-30', test: v => v != null && v >= 25 && v < 30 },
    { name: '>=30', test: v => v != null && v >= 30 },
  ]);

  printBucketAnalysis(rows, 'chop', [
    { name: '<45', test: v => v != null && v < 45 },
    { name: '45-50', test: v => v != null && v >= 45 && v < 50 },
    { name: '50-56', test: v => v != null && v >= 50 && v <= 56 },
    { name: '>56', test: v => v != null && v > 56 },
  ]);

  printBucketAnalysis(rows, 'volPct50', [
    { name: '<30 pct', test: v => v != null && v < 30 },
    { name: '30-70 pct', test: v => v != null && v >= 30 && v < 70 },
    { name: '70-90 pct', test: v => v != null && v >= 70 && v < 90 },
    { name: '>=90 pct', test: v => v != null && v >= 90 },
  ]);

  printBucketAnalysis(rows, 'dirVwapExt', [
    { name: '<=0.5', test: v => v != null && v <= 0.5 },
    { name: '0.5-1.0', test: v => v != null && v > 0.5 && v <= 1.0 },
    { name: '1.0-2.0', test: v => v != null && v > 1.0 && v <= 2.0 },
    { name: '>2.0', test: v => v != null && v > 2.0 },
  ]);

  printBucketAnalysis(rows, 'dirBbExt', [
    { name: '<70', test: v => v != null && v < 70 },
    { name: '70-85', test: v => v != null && v >= 70 && v < 85 },
    { name: '85-95', test: v => v != null && v >= 85 && v < 95 },
    { name: '>=95', test: v => v != null && v >= 95 },
  ]);

  printBucketAnalysis(rows, 'breakoutExcessAtr', [
    { name: '<=0.10', test: v => v != null && v <= 0.10 },
    { name: '0.10-0.30', test: v => v != null && v > 0.10 && v <= 0.30 },
    { name: '0.30-0.60', test: v => v != null && v > 0.30 && v <= 0.60 },
    { name: '>0.60', test: v => v != null && v > 0.60 },
  ]);

  printBucketAnalysis(rows, 'gapDirAtr', [
    { name: '<=0', test: v => v != null && v <= 0 },
    { name: '0-0.15', test: v => v != null && v > 0 && v <= 0.15 },
    { name: '0.15-0.35', test: v => v != null && v > 0.15 && v <= 0.35 },
    { name: '>0.35', test: v => v != null && v > 0.35 },
  ]);

  printBucketAnalysis(rows, 'close1Atr', [
    { name: '<0', test: v => v != null && v < 0 },
    { name: '0-0.5', test: v => v != null && v >= 0 && v < 0.5 },
    { name: '0.5-1.0', test: v => v != null && v >= 0.5 && v < 1.0 },
    { name: '>=1.0', test: v => v != null && v >= 1.0 },
  ]);

  printBucketAnalysis(rows, 'maxFav3Atr', [
    { name: '<0.5', test: v => v != null && v < 0.5 },
    { name: '0.5-1.0', test: v => v != null && v >= 0.5 && v < 1.0 },
    { name: '1.0-2.0', test: v => v != null && v >= 1.0 && v < 2.0 },
    { name: '>=2.0', test: v => v != null && v >= 2.0 },
  ]);

  printComboAnalysis(rows);

  const worstTrades = losses
    .slice()
    .sort((a, b) => a.profit - b.profit)
    .slice(0, 5)
    .map(r => ({
      side: r.side,
      profit: formatNumber(r.profit, 2),
      session: r.session,
      adx: formatNumber(r.adx, 1),
      chop: formatNumber(r.chop, 1),
      volPct50: formatNumber(r.volPct50, 1),
      dirVwapExt: formatNumber(r.dirVwapExt, 2),
      dirBbExt: formatNumber(r.dirBbExt, 1),
      breakoutExcessAtr: formatNumber(r.breakoutExcessAtr, 2),
      close1Atr: formatNumber(r.close1Atr, 2),
      maxFav3Atr: formatNumber(r.maxFav3Atr, 2),
    }));

  console.log('\nWorst loss snapshots');
  for (const trade of worstTrades) {
    console.log(`  ${trade.side.padEnd(5)} profit ${trade.profit.padStart(8)} | ${trade.session.padEnd(6)} | ADX ${trade.adx.padStart(5)} | CHOP ${trade.chop.padStart(5)} | vol% ${trade.volPct50.padStart(5)} | VWAPext ${trade.dirVwapExt.padStart(5)} | BBext ${trade.dirBbExt.padStart(5)} | breakout ${trade.breakoutExcessAtr.padStart(5)} | close1 ${trade.close1Atr.padStart(5)} | MFE3 ${trade.maxFav3Atr.padStart(5)}`);
  }
}

try {
  await main();
  process.exit(0);
} catch (error) {
  console.error(error);
  process.exit(1);
}

import unittest
import sys
import types

pybit_module = types.ModuleType("pybit")
unified_trading_module = types.ModuleType("pybit.unified_trading")
unified_trading_module.HTTP = object
unified_trading_module.WebSocket = object
pybit_module.unified_trading = unified_trading_module
sys.modules.setdefault("pybit", pybit_module)
sys.modules.setdefault("pybit.unified_trading", unified_trading_module)

from rl_execution_bot import summarize_wolfe_quality_results, wolfe_quality_result


class WolfeQualityCheckpointTests(unittest.TestCase):
    def test_extracts_only_scored_wolfe_results(self):
        row = {
            "decision_id": "abc",
            "completed_at": "2026-06-15T10:00:00+00:00",
            "strategy": "wolfe_wave_v2",
            "source_status": "accepted",
            "source_features": {"research_quality_score": 93.5},
            "reward": {"reward_actual_r": 1.25, "closed_pnl": 12.5},
        }
        self.assertEqual(
            wolfe_quality_result(row),
            {
                "decision_id": "abc",
                "completed_at": "2026-06-15T10:00:00+00:00",
                "strategy": "wolfe_wave_v2",
                "source_status": "accepted",
                "quality": 93.5,
                "actual_r": 1.25,
                "pnl": 12.5,
            },
        )
        self.assertIsNone(wolfe_quality_result({**row, "strategy": "million_moves"}))
        self.assertIsNone(wolfe_quality_result({**row, "source_features": {}}))
        self.assertIsNone(wolfe_quality_result({**row, "reversed_trade": True}))

    def test_summarizes_strategy_and_quality_buckets(self):
        rows = [
            {"strategy": "wolfe_wave", "quality": 70, "actual_r": -1, "pnl": -10},
            {"strategy": "wolfe_wave", "quality": 80, "actual_r": 0.5, "pnl": 5},
            {"strategy": "wolfe_wave", "quality": 95, "actual_r": 2, "pnl": 20},
            {"strategy": "wolfe_wave_v2", "quality": 94, "actual_r": 1, "pnl": 12},
        ]
        summary = summarize_wolfe_quality_results(rows)
        self.assertEqual(summary["total"]["count"], 4)
        self.assertAlmostEqual(summary["total"]["net_r"], 2.5)
        self.assertAlmostEqual(summary["old"]["avg_r"], 0.5)
        self.assertEqual(summary["old"]["buckets"]["lt75"]["count"], 1)
        self.assertEqual(summary["old"]["buckets"]["75to92"]["count"], 1)
        self.assertEqual(summary["old"]["buckets"]["gt92"]["count"], 1)
        self.assertAlmostEqual(summary["old"]["top_bottom_delta_r"], 3.0)
        self.assertEqual(summary["v2"]["count"], 1)


if __name__ == "__main__":
    unittest.main()

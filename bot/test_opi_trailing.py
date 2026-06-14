import unittest
import threading

from matrix_signal_bot import _parse_opi_signal
from rl_execution_bot import RLExecutionService, trailing_stop_state


class OpiBrainParserTests(unittest.TestCase):
    def test_refined_brain_alert_uses_trigger_trailing(self) -> None:
        signal = _parse_opi_signal(
            "🚨 Opᶦ // BTC 5m LONG @ $79,832.0\n"
            "TP $80,398.8 · SL $78,235.4 · ★ 5.0/8\n"
            "ℂC BOTTOMING 5TF [1m,2m,3m,5m,30m]\n"
            "Refined $79,823.7→$79,832.0"
        )
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal["strategy"], "opi_brain")
        self.assertEqual(signal["exit_style"], "trigger_trailing")
        self.assertAlmostEqual(signal["trail_trigger_price"], 80398.8)
        self.assertAlmostEqual(signal["trail_distance_pct"], 0.004)
        self.assertTrue(signal["move_stop_to_breakeven_on_trigger"])

    def test_brain_allowlist_keeps_existing_xlm_five_minute_candidate(self) -> None:
        signal = _parse_opi_signal(
            "🚨 Opᶦ // XLM 5m SHORT @ $0.28203\n"
            "TP $0.27583 · SL $0.28513\n"
            "FVG BURNOUT — BURNING OUT + wick burnout"
        )
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal["strategy"], "opi_brain")
        self.assertEqual(signal["timeframe"], "5m")


class TrailingStopStateTests(unittest.TestCase):
    def test_long_arms_and_locks_profit(self) -> None:
        state = trailing_stop_state(
            direction="long",
            entry=100.0,
            market_price=101.0,
            previous_best=100.0,
            previous_stop=98.0,
            trigger_price=100.75,
            trail_distance_pct=0.004,
            move_to_breakeven=True,
        )
        self.assertTrue(state["armed"])
        self.assertAlmostEqual(float(state["best_price"]), 101.0)
        self.assertAlmostEqual(float(state["stop_price"]), 100.596)

    def test_short_stop_only_moves_down(self) -> None:
        state = trailing_stop_state(
            direction="short",
            entry=100.0,
            market_price=98.0,
            previous_best=98.5,
            previous_stop=100.0,
            trigger_price=99.25,
            trail_distance_pct=0.004,
            move_to_breakeven=True,
            was_armed=True,
        )
        self.assertTrue(state["armed"])
        self.assertAlmostEqual(float(state["best_price"]), 98.0)
        self.assertAlmostEqual(float(state["stop_price"]), 98.392)

    def test_unarmed_stop_stays_at_initial_level(self) -> None:
        state = trailing_stop_state(
            direction="long",
            entry=100.0,
            market_price=100.5,
            previous_best=100.0,
            previous_stop=98.0,
            trigger_price=100.75,
            trail_distance_pct=0.004,
            move_to_breakeven=True,
        )
        self.assertFalse(state["armed"])
        self.assertAlmostEqual(float(state["stop_price"]), 98.0)


class PartialStopAmendTests(unittest.TestCase):
    def test_amend_targets_the_child_stop_for_the_entry_order(self) -> None:
        class FakeHttp:
            def __init__(self) -> None:
                self.amended = None

            def get_open_orders(self, **_kwargs):
                return {
                    "retCode": 0,
                    "result": {
                        "list": [
                            {
                                "orderId": "wrong-stop",
                                "parentOrderId": "another-entry",
                                "stopOrderType": "PartialStopLoss",
                                "positionIdx": 1,
                                "side": "Sell",
                                "qty": "2",
                                "triggerPrice": "98",
                                "createdTime": "1001",
                            },
                            {
                                "orderId": "right-stop",
                                "parentOrderId": "entry-order",
                                "stopOrderType": "PartialStopLoss",
                                "positionIdx": 1,
                                "side": "Sell",
                                "qty": "2",
                                "triggerPrice": "98",
                                "createdTime": "1001",
                            },
                        ]
                    },
                }

            def amend_order(self, **kwargs):
                self.amended = kwargs
                return {"retCode": 0, "result": {"orderId": kwargs["orderId"]}}

        service = RLExecutionService.__new__(RLExecutionService)
        service.http = FakeHttp()
        service.lock = threading.RLock()
        service.instrument_cache = {
            "BTCUSDT": {
                "tick_size": 0.1,
                "qty_step": 0.001,
                "min_qty": 0.001,
                "max_leverage": 100.0,
            }
        }
        service.order_to_decision = {}
        decision = {
            "decision_id": "decision-1",
            "symbol": "BTCUSDT",
            "effective_direction": "long",
            "entry_order_id": "entry-order",
            "entry_order_link_id": "entry-link",
            "position_idx": 1,
            "exit_side": "Sell",
            "qty": "2",
            "qty_float": 2.0,
            "opened_at_ms": 1000,
            "trailing_current_stop": 98.0,
            "setup": {"entry": 100.0, "stop_loss": 98.0},
        }

        result = service._amend_decision_stop(decision, 100.59, reason="test")

        self.assertTrue(result["ok"])
        self.assertEqual(service.http.amended["orderId"], "right-stop")
        self.assertEqual(service.http.amended["triggerPrice"], "100.5")
        self.assertEqual(decision["trailing_stop_order_id"], "right-stop")


if __name__ == "__main__":
    unittest.main()

import copy
import json
import unittest

import numpy as np
import pandas as pd

from lib import realtime_signal


ET = "America/New_York"


def minute_bars(periods: int = 41, *, benchmark: bool = False,
                start: str = "2026-08-24 09:30") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1min", tz=ET)
    if benchmark:
        steps = np.resize(np.array([0.05, -0.04]), periods)
    else:
        steps = np.resize(np.array([0.12, -0.08]), periods)
    # 最後の2本をともに強い確定足にし、2本確認の状態遷移を決定的にする。
    if periods >= 2:
        steps[-2:] = [0.05, 0.05] if benchmark else [0.10, 0.10]
    close = 100.0 + np.cumsum(steps)
    open_ = np.r_[100.0, close[:-1]]
    volume = np.full(periods, 1_000.0)
    volume[-2:] = 2_000.0
    return pd.DataFrame({
        "Open": open_,
        "High": np.maximum(open_, close) + 0.50,
        "Low": np.minimum(open_, close) - 0.50,
        "Close": close,
        "Volume": volume,
    }, index=index)


def session(now: pd.Timestamp, name: str = "regular") -> dict:
    return {
        "session": name, "calendar_session": name,
        "tradable": True, "as_of": now,
        "reason": f"現在は{name}セッションです",
    }


def snapshot(frame: pd.DataFrame, now: pd.Timestamp, *, bid=None, ask=None,
             age_seconds: float = 5.0, session_name: str = "regular") -> dict:
    price = float(frame["Close"].iloc[-1]) if not frame.empty else 100.0
    return {
        "source": "moomoo OpenAPI",
        "price": price,
        "bid": price - 0.01 if bid is None else bid,
        "ask": price + 0.01 if ask is None else ask,
        "update_time": (now - pd.Timedelta(seconds=age_seconds)).isoformat(),
        "suspension": False,
        "decision_ready": True,
        "decision_session": session_name,
        "uses_daily_bars": False,
    }


def ready_plan(*, daily: bool = True, status: str = "READY") -> dict:
    return {
        "status": status,
        "description_ja": "購入プランの確認結果",
        "daily_signal": {
            "is_buy_candidate": daily,
            "reason_ja": "確定日足の買い条件",
        },
        "buy_zone": {"maximum_price": 200.0, "high": 200.0},
        "risk": {"stop": 95.0, "target": 120.0},
        "chase_warning": {"minimum_rr": 1.5},
    }


def evaluate(frame: pd.DataFrame, benchmark: pd.DataFrame,
             now: pd.Timestamp, **kwargs) -> dict:
    session_value = kwargs.pop("session_value", session(now))
    snapshot_value = kwargs.pop(
        "snapshot_value",
        snapshot(frame, now, session_name=session_value.get("session", "regular")),
    )
    return realtime_signal.evaluate_realtime_signal(
        "AAPL", frame,
        benchmark_bars=benchmark,
        snapshot=snapshot_value,
        session=session_value,
        purchase_plan=kwargs.pop("purchase_plan", None),
        now=now,
        **kwargs,
    )


class CompletedBarTests(unittest.TestCase):
    def test_current_forming_one_minute_bar_is_excluded(self):
        frame = minute_bars()
        now = frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
        partial_time = now.floor("min")
        partial = frame.iloc[-1].copy()
        partial[["Open", "High", "Low", "Close"]] = [999, 1001, 998, 1000]
        with_partial = pd.concat([
            frame,
            pd.DataFrame([partial], index=pd.DatetimeIndex([partial_time])),
        ])

        completed, meta = realtime_signal.completed_intraday_bars(
            with_partial, now=now, session=session(now))

        self.assertEqual(completed.index[-1], frame.index[-1])
        self.assertEqual(meta["dropped_incomplete"], 1)
        self.assertNotEqual(float(completed["Close"].iloc[-1]), 1000.0)

    def test_partial_bar_cannot_change_features_or_score(self):
        frame = minute_bars()
        benchmark = minute_bars(benchmark=True)
        now = frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
        base = evaluate(frame, benchmark, now)

        partial_time = now.floor("min")
        row = frame.iloc[-1].copy()
        row[["Open", "High", "Low", "Close", "Volume"]] = [999, 1002, 998, 1000, 9e9]
        bench_row = benchmark.iloc[-1].copy()
        bench_row[["Open", "High", "Low", "Close", "Volume"]] = [1, 2, 0.5, 1.5, 9e9]
        with_partial = pd.concat([
            frame, pd.DataFrame([row], index=pd.DatetimeIndex([partial_time]))])
        benchmark_partial = pd.concat([
            benchmark,
            pd.DataFrame([bench_row], index=pd.DatetimeIndex([partial_time]))])

        result = evaluate(with_partial, benchmark_partial, now)
        self.assertEqual(result["features"], base["features"])
        self.assertEqual(result["score"], base["score"])
        self.assertEqual(result["signal_bar_time"], base["signal_bar_time"])

    def test_invalid_values_in_partial_bar_do_not_block_completed_bars(self):
        frame = minute_bars()
        benchmark = minute_bars(benchmark=True)
        now = frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
        partial_time = now.floor("min")
        partial = frame.iloc[-1].copy()
        partial["High"] = np.nan
        with_partial = pd.concat([
            frame, pd.DataFrame([partial], index=pd.DatetimeIndex([partial_time]))])

        result = evaluate(
            with_partial, benchmark, now,
            snapshot_value=snapshot(frame, now),
        )

        self.assertNotEqual(result["state"], "DATA_WAIT")
        self.assertTrue(result["data_quality"]["bars"]["valid"])
        self.assertEqual(result["data_quality"]["bars"]["dropped_incomplete"], 1)

    def test_early_close_afterhours_starts_at_regular_close(self):
        index = pd.date_range(
            "2026-11-27 13:00", periods=40, freq="1min", tz=ET)
        close = pd.Series(np.linspace(100, 101, len(index)), index=index)
        frame = pd.DataFrame({
            "Open": close - 0.01, "High": close + 0.05,
            "Low": close - 0.05, "Close": close, "Volume": 1_000.0,
        }, index=index)
        now = pd.Timestamp("2026-11-27 13:40:30", tz=ET)
        completed, meta = realtime_signal.completed_intraday_bars(
            frame, now=now,
            session={
                "session": "afterhours", "tradable": True,
                "regular_close": pd.Timestamp("2026-11-27 13:00", tz=ET),
                "afterhours_close": pd.Timestamp("2026-11-27 17:00", tz=ET),
            },
        )
        self.assertEqual(len(completed), 40)
        self.assertEqual(meta["session"], "afterhours")


class EntryStateTests(unittest.TestCase):
    def setUp(self):
        self.frame = minute_bars()
        self.benchmark = minute_bars(benchmark=True)

    def test_two_distinct_closed_bars_are_required(self):
        first_frame = self.frame.iloc[:-1]
        first_benchmark = self.benchmark.iloc[:-1]
        now1 = first_frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
        first = evaluate(first_frame, first_benchmark, now1)

        self.assertEqual(first["state"], "BUY_SETUP")
        self.assertEqual(first["confirmation"]["streak"], 1)
        self.assertFalse(first["actionable"])

        now2 = self.frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
        second = evaluate(
            self.frame, self.benchmark, now2,
            previous_memory=first["memory"],
        )
        self.assertEqual(second["state"], "BUY_READY")
        self.assertTrue(second["actionable"])
        self.assertTrue(second["alert"]["emit"])

    def test_same_bar_is_idempotent_for_confirmation(self):
        frame = self.frame.iloc[:-1]
        benchmark = self.benchmark.iloc[:-1]
        now = frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
        first = evaluate(frame, benchmark, now)
        repeated = evaluate(
            frame, benchmark, now, previous_memory=first["memory"])

        self.assertEqual(first["state"], "BUY_SETUP")
        self.assertEqual(repeated["state"], "BUY_SETUP")
        self.assertEqual(repeated["confirmation"]["streak"], 1)
        self.assertEqual(repeated["memory"]["pending_count"], 1)

    def test_ready_state_needs_two_distinct_bars_to_release(self):
        cfg = realtime_signal.realtime_config()
        key = "2026-08-24:regular"
        first_time = pd.Timestamp("2026-08-24 10:00", tz=ET).isoformat()
        second_time = pd.Timestamp("2026-08-24 10:01", tz=ET).isoformat()
        third_time = pd.Timestamp("2026-08-24 10:02", tz=ET).isoformat()
        fourth_time = pd.Timestamp("2026-08-24 10:03", tz=ET).isoformat()
        _, memory, _ = realtime_signal._entry_state(
            100.0, True, first_time, key, None, cfg, process_bar=True)
        ready, memory, _ = realtime_signal._entry_state(
            100.0, True, second_time, key, memory, cfg, process_bar=True)
        first_dip, memory, _ = realtime_signal._entry_state(
            50.0, True, third_time, key, memory, cfg, process_bar=True)
        repeated, memory, _ = realtime_signal._entry_state(
            50.0, True, third_time, key, memory, cfg, process_bar=True)
        released, memory, _ = realtime_signal._entry_state(
            50.0, True, fourth_time, key, memory, cfg, process_bar=True)

        self.assertEqual(ready, "BUY_READY")
        self.assertEqual(first_dip, "BUY_READY")
        self.assertEqual(repeated, "BUY_READY")
        self.assertEqual(released, "NEUTRAL")
        self.assertIsNone(memory["latched_state"])

    def test_daily_and_purchase_plan_do_not_affect_realtime_entry(self):
        first_frame = self.frame.iloc[:-1]
        first_benchmark = self.benchmark.iloc[:-1]
        now1 = first_frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
        first = evaluate(
            first_frame, first_benchmark, now1,
            purchase_plan=ready_plan(daily=False, status="WAIT"),
        )
        now2 = self.frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
        second = evaluate(
            self.frame, self.benchmark, now2,
            purchase_plan={"status": "UNAVAILABLE", "daily_signal": {}},
            previous_memory=first["memory"],
        )

        self.assertEqual(first["state"], "BUY_SETUP")
        self.assertEqual(second["state"], "BUY_READY")
        self.assertTrue(second["actionable"])
        keys = {gate["key"] for gate in second["gates"]}
        for removed in ("daily_buy", "purchase_plan", "buy_ceiling", "live_rr"):
            self.assertNotIn(removed, keys)
        self.assertIsNone(second["prices"]["maximum_buy_price"])
        self.assertIsNone(second["prices"]["current_rr"])

    def test_same_intraday_inputs_are_invariant_to_any_purchase_plan(self):
        now = self.frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
        without_plan = evaluate(
            self.frame, self.benchmark, now, purchase_plan=None)
        with_conflicting_daily_plan = evaluate(
            self.frame, self.benchmark, now,
            purchase_plan={
                "status": "READY",
                "daily_signal": {"is_buy_candidate": False, "verdict": "WAIT"},
                "buy_zone": {"maximum_price": 0.01},
                "risk": {"stop": 9999.0, "target": 0.01},
                "chase_warning": {"minimum_rr": 999.0},
            },
        )
        for key in (
            "state", "actionable", "score", "checks", "gates", "features",
            "prices", "confirmation", "memory",
        ):
            self.assertEqual(with_conflicting_daily_plan[key], without_plan[key])

    def test_all_four_us_sessions_are_actionable_with_valid_intraday_data(self):
        starts = {
            "premarket": "2026-08-24 04:00",
            "regular": "2026-08-24 09:30",
            "afterhours": "2026-08-24 16:00",
            "overnight": "2026-08-23 20:00",
        }
        for name, start in starts.items():
            with self.subTest(session=name):
                frame = minute_bars(start=start)
                benchmark = minute_bars(benchmark=True, start=start)
                first_frame = frame.iloc[:-1]
                first_benchmark = benchmark.iloc[:-1]
                now1 = first_frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
                first = evaluate(
                    first_frame, first_benchmark, now1,
                    session_value=session(now1, name),
                )
                now2 = frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
                second = evaluate(
                    frame, benchmark, now2,
                    session_value=session(now2, name),
                    previous_memory=first["memory"],
                )
                self.assertEqual(second["state"], "BUY_READY")
                self.assertTrue(second["actionable"])

    def test_entry_never_returns_sell(self):
        now = self.frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
        result = evaluate(self.frame, self.benchmark, now)
        allowed = {"DATA_WAIT", "WAIT", "NEUTRAL", "BUY_SETUP", "BUY_READY"}
        self.assertIn(result["state"], allowed)
        self.assertNotEqual(result["state"], "SELL")
        self.assertNotEqual(result["action"]["code"], "SELL")

    def test_stale_quote_fails_closed(self):
        now = self.frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
        stale = snapshot(self.frame, now, age_seconds=60)
        result = evaluate(
            self.frame, self.benchmark, now, snapshot_value=stale)
        self.assertEqual(result["state"], "DATA_WAIT")
        self.assertIn("quote_freshness", result["components"]["execution"]["blocking_gate_keys"])

    def test_unverified_or_wrong_session_quote_fails_closed(self):
        now = self.frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
        for changes in (
            {"decision_ready": False},
            {"decision_session": "afterhours"},
            {"uses_daily_bars": True},
        ):
            with self.subTest(changes=changes):
                unsafe = snapshot(self.frame, now)
                unsafe.update(changes)
                result = evaluate(
                    self.frame, self.benchmark, now,
                    snapshot_value=unsafe,
                )
                self.assertEqual(result["state"], "DATA_WAIT")
                self.assertFalse(result["actionable"])
                self.assertIn(
                    "session_quote",
                    result["components"]["execution"]["blocking_gate_keys"],
                )


class HoldingStateTests(unittest.TestCase):
    def setUp(self):
        self.now = pd.Timestamp("2026-08-24 11:00:30", tz=ET)
        self.empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def quote(self, bid: float, *, age_seconds: float = 5) -> dict:
        return {
            "source": "moomoo OpenAPI", "price": bid,
            "bid": bid, "ask": bid + 0.02,
            "update_time": (self.now - pd.Timedelta(seconds=age_seconds)).isoformat(),
            "suspension": False,
            "decision_ready": True,
            "decision_session": "regular",
            "uses_daily_bars": False,
        }

    def evaluate_holding(self, bid: float, position: dict, *, age_seconds: float = 5,
                         previous_memory=None) -> dict:
        return realtime_signal.evaluate_realtime_signal(
            "AAPL", self.empty,
            benchmark_bars=self.empty,
            snapshot=self.quote(bid, age_seconds=age_seconds),
            session=session(self.now), purchase_plan={}, now=self.now,
            position=position, previous_memory=previous_memory,
        )

    def test_fresh_bid_stop_has_priority_even_without_bars(self):
        result = self.evaluate_holding(
            99.0, {"held": True, "stop": 100.0, "target": 120.0})
        self.assertEqual(result["state"], "RISK_EXIT")
        self.assertEqual(result["position_mode"], "holding")
        self.assertTrue(result["actionable"])
        self.assertTrue(result["alert"]["emit"])

    def test_fresh_bid_target_is_take_profit(self):
        result = self.evaluate_holding(
            121.0, {"held": True, "stop": 95.0, "target": 120.0})
        self.assertEqual(result["state"], "TAKE_PROFIT")
        self.assertTrue(result["actionable"])

    def test_fresh_bid_does_not_require_ask_for_holding_exit(self):
        quote = self.quote(99.0)
        quote["ask"] = None
        result = realtime_signal.evaluate_realtime_signal(
            "AAPL", self.empty,
            benchmark_bars=self.empty,
            snapshot=quote, session=session(self.now),
            purchase_plan={}, now=self.now,
            position={"held": True, "stop": 100.0, "target": 120.0},
        )
        self.assertEqual(result["state"], "RISK_EXIT")
        self.assertTrue(result["actionable"])
        self.assertNotIn(
            "bid_ask", result["components"]["execution"]["blocking_gate_keys"])

    def test_invalid_holding_price_relationship_waits(self):
        result = self.evaluate_holding(
            121.0, {"held": True, "stop": 122.0, "target": 120.0})
        self.assertEqual(result["state"], "DATA_WAIT")
        self.assertFalse(result["actionable"])
        self.assertIn("低く", result["reason_ja"])

    def test_stale_bid_never_asserts_an_exit(self):
        result = self.evaluate_holding(
            90.0, {"held": True, "stop": 100.0, "target": 120.0},
            age_seconds=60,
        )
        self.assertEqual(result["state"], "DATA_WAIT")
        self.assertFalse(result["actionable"])

    def test_suspended_quote_never_asserts_an_exit(self):
        quote = self.quote(90.0)
        quote["suspension"] = True
        result = realtime_signal.evaluate_realtime_signal(
            "AAPL", self.empty,
            benchmark_bars=self.empty,
            snapshot=quote,
            session=session(self.now), purchase_plan={}, now=self.now,
            position={"held": True, "stop": 100.0, "target": 120.0},
        )
        self.assertEqual(result["state"], "DATA_WAIT")
        self.assertFalse(result["actionable"])

    def test_holding_gates_do_not_show_new_purchase_requirements(self):
        result = self.evaluate_holding(
            110.0, {"held": True, "stop": 100.0, "target": 120.0})
        keys = {gate["key"] for gate in result["gates"]}
        self.assertNotIn("daily_buy", keys)
        self.assertNotIn("purchase_plan", keys)
        self.assertNotIn("buy_ceiling", keys)
        self.assertIn("holding_stop", keys)
        self.assertIn("holding_target", keys)

    def test_exit_during_nontradable_session_is_informational_only(self):
        closed = session(self.now)
        closed.update({"tradable": False, "reason": "現在は取引時間外です"})
        result = realtime_signal.evaluate_realtime_signal(
            "AAPL", self.empty,
            benchmark_bars=self.empty,
            snapshot=self.quote(99.0), session=closed,
            purchase_plan={}, now=self.now,
            position={"held": True, "stop": 100.0, "target": 120.0},
        )
        self.assertEqual(result["state"], "RISK_EXIT")
        self.assertFalse(result["actionable"])

    def test_extended_session_exit_is_actionable_by_default(self):
        extended = session(self.now)
        extended.update({
            "session": "premarket", "calendar_session": "premarket",
            "tradable": True,
        })
        extended_quote = self.quote(99.0)
        extended_quote["decision_session"] = "premarket"
        result = realtime_signal.evaluate_realtime_signal(
            "AAPL", self.empty,
            benchmark_bars=self.empty,
            snapshot=extended_quote, session=extended,
            purchase_plan={}, now=self.now,
            position={"held": True, "stop": 100.0, "target": 120.0},
        )
        self.assertEqual(result["state"], "RISK_EXIT")
        self.assertTrue(result["actionable"])
        self.assertTrue(result["components"]["execution"]["ready"])

    def test_calendar_session_blocks_stale_regular_market_state(self):
        stale_regular = session(self.now)
        stale_regular["calendar_session"] = "afterhours"
        result = realtime_signal.evaluate_realtime_signal(
            "AAPL", self.empty,
            benchmark_bars=self.empty,
            snapshot=self.quote(99.0), session=stale_regular,
            purchase_plan={}, now=self.now,
            position={"held": True, "stop": 100.0, "target": 120.0},
        )
        self.assertEqual(result["state"], "RISK_EXIT")
        self.assertFalse(result["actionable"])
        self.assertIn(
            "actionable_session",
            result["components"]["execution"]["blocking_gate_keys"],
        )

    def test_missing_risk_levels_are_explicit_and_not_guessed(self):
        result = self.evaluate_holding(100.0, {"held": True})

        self.assertEqual(result["state"], "DATA_WAIT")
        self.assertFalse(result["actionable"])
        self.assertIn("入力", result["reason_ja"])
        risk_gates = {gate["key"]: gate for gate in result["gates"]}
        self.assertFalse(risk_gates["holding_stop"]["passed"])
        self.assertFalse(risk_gates["holding_target"]["passed"])
        self.assertIsNone(result["prices"]["stop"])
        self.assertIsNone(result["prices"]["target"])

    def test_new_purchase_plan_prices_are_not_used_for_holding(self):
        result = realtime_signal.evaluate_realtime_signal(
            "AAPL", self.empty, benchmark_bars=self.empty,
            snapshot=self.quote(90.0), session=session(self.now),
            purchase_plan=ready_plan(), now=self.now,
            position={"held": True, "daily_verdict": "HOLD"},
        )
        self.assertEqual(result["state"], "DATA_WAIT")
        self.assertIsNone(result["prices"]["stop"])
        self.assertIsNone(result["prices"]["target"])

    def test_daily_verdict_is_ignored_for_holding(self):
        results = [
            self.evaluate_holding(
                110.0,
                {"held": True, "daily_verdict": verdict,
                 "stop": 100.0, "target": 120.0},
            )
            for verdict in ("RISK_EXIT", "TAKE_PROFIT", "WAIT", "HOLD", None)
        ]
        for result in results:
            self.assertEqual(result["state"], "HOLD")
            self.assertNotIn("日足", result["reason_ja"])
            self.assertNotIn(
                "holding_daily", {gate["key"] for gate in result["gates"]})

    def test_daily_verdict_cannot_rescue_missing_or_invalid_monitor_prices(self):
        for prices in ({}, {"stop": 122.0, "target": 120.0}):
            with self.subTest(prices=prices):
                result = self.evaluate_holding(
                    110.0,
                    {"held": True, "daily_verdict": "RISK_EXIT", **prices},
                )
                self.assertEqual(result["state"], "DATA_WAIT")
                self.assertFalse(result["actionable"])
                self.assertFalse(result["components"]["execution"]["ready"])
                self.assertEqual(result["data_quality"]["status"], "INVALID")

    def test_holding_components_use_only_holding_gates(self):
        result = self.evaluate_holding(
            110.0,
            {"held": True, "stop": 100.0, "target": 120.0},
        )
        self.assertTrue(result["components"]["execution"]["ready"])
        self.assertEqual(result["components"]["execution"]["blocking_gate_keys"], [])
        self.assertEqual(result["data_quality"]["status"], "GOOD")

    def test_exit_alert_is_edge_triggered_on_same_bar(self):
        first = self.evaluate_holding(
            99.0, {"held": True, "stop": 100.0, "target": 120.0})
        repeated = self.evaluate_holding(
            99.0, {"held": True, "stop": 100.0, "target": 120.0},
            previous_memory=first["memory"],
        )
        self.assertTrue(first["alert"]["emit"])
        self.assertFalse(repeated["alert"]["emit"])


class ContractTests(unittest.TestCase):
    def test_result_is_json_serializable_and_inputs_are_not_mutated(self):
        frame = minute_bars()
        benchmark = minute_bars(benchmark=True)
        now = frame.index[-1] + pd.Timedelta(minutes=1, seconds=30)
        plan = ready_plan()
        snap = snapshot(frame, now)
        frame_before = frame.copy(deep=True)
        benchmark_before = benchmark.copy(deep=True)
        plan_before = copy.deepcopy(plan)
        snap_before = copy.deepcopy(snap)

        result = evaluate(
            frame, benchmark, now, purchase_plan=plan,
            snapshot_value=snap)
        json.dumps(result, ensure_ascii=False)

        pd.testing.assert_frame_equal(frame, frame_before)
        pd.testing.assert_frame_equal(benchmark, benchmark_before)
        self.assertEqual(plan, plan_before)
        self.assertEqual(snap, snap_before)
        for key in (
                "state", "status", "action", "components", "memory",
                "checks", "gates", "features", "prices", "data_quality"):
            self.assertIn(key, result)
        self.assertTrue(result["read_only"])
        self.assertFalse(result["places_orders"])
        self.assertFalse(result["uses_daily_data"])
        self.assertFalse(result["uses_purchase_plan"])
        self.assertFalse(result["uses_moomoo_history_quota"])

    def test_invalid_hysteresis_config_is_rejected(self):
        with self.assertRaises(ValueError):
            realtime_signal.realtime_config({
                "setup_score": 60, "release_score": 80, "entry_score": 75,
            })

    def test_default_sessions_are_all_supported_us_trading_sessions(self):
        self.assertEqual(
            realtime_signal.realtime_config()["actionable_sessions"],
            realtime_signal.TRADING_SESSIONS,
        )
        with self.assertRaises(ValueError):
            realtime_signal.realtime_config({"actionable_sessions": ("regular", "other")})


if __name__ == "__main__":
    unittest.main()

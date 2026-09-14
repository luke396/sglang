"""Unit tests for srt/disaggregation/common/conn bootstrap registration and route caps."""

import asyncio
import json
import struct
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1,
    DCP_KV_LAYOUT_TOKEN_STRIPE_V1,
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    PageRegistrationState,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _run_control_message(manager, start_listener, message):
    """Feed one wire message through the real control loop, then stop at recv."""

    class Socket:
        def recv_multipart(self):
            nonlocal message
            if message is None:
                raise SystemExit
            current, message = message, None
            return current

    class Thread:
        def __init__(self, *, target):
            self.target = target

        def start(self):
            try:
                self.target()
            except SystemExit:
                pass

    manager.server_socket = Socket()
    with patch("sglang.srt.disaggregation.mooncake.conn.threading.Thread", Thread):
        start_listener()


class TestRegisterToBootstrap(CustomTestCase):
    """Tests for CommonKVManager.register_to_bootstrap retry/backoff behavior."""

    def setUp(self):
        # register_to_bootstrap reads get_parallel().load_balance_method /
        # .enable_dsa_cache_layer_split and get_serving().port from the
        # published config.
        override = get_context().override_server_args(
            load_balance_method="follow_bootstrap_room", port=30000
        )
        override.install()
        self.addCleanup(override.restore)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_succeeds_on_first_attempt(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_put.return_value = mock_response

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        mock_put.assert_called_once()
        mock_time.sleep.assert_not_called()

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_succeeds_after_retries(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.side_effect = [fail_resp, fail_resp, success_resp]

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 3)
        self.assertEqual(mock_time.sleep.call_count, 2)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_all_retries_exhausted(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_put.return_value = fail_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 5)
        # Sleep is only called between attempts, not after the final failure
        self.assertEqual(mock_time.sleep.call_count, 4)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_exception_with_nested_cause(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0

        root_exc = ConnectionRefusedError("connection refused")
        inner_exc = OSError("os error")
        inner_exc.__cause__ = root_exc
        outer_exc = Exception("wrapped")
        outer_exc.__cause__ = inner_exc

        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.side_effect = [outer_exc, success_resp]

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 2)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_exception_with_no_cause(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0

        exc = ConnectionError("plain connection error")
        exc.__cause__ = None

        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.side_effect = [exc, success_resp]

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 2)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_backoff_delay_exponential(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_put.return_value = fail_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        # With monotonic() = 0.0, jitter factor = 0.75 + 0.25 * (0.0 % 1) = 0.75
        # delay = min(1.0 * 2^attempt, 30.0) * 0.75
        # Sleep happens only between attempts (attempt 0..3), not after the final failure
        expected_calls = [call(0.75), call(1.5), call(3.0), call(6.0)]
        self.assertEqual(mock_time.sleep.call_args_list, expected_calls)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_jitter_never_exceeds_max_delay(self, mock_put, mock_time):
        """Guard against operator-precedence regressions in the jitter factor.

        The jitter factor must stay in [0.75, 1.0), so a delay capped at
        max_delay must never exceed max_delay after applying jitter.
        """
        # monotonic() returns a value whose fractional part is close to 1.
        # If the parentheses around `time.monotonic() % 1` were dropped, the
        # jitter factor could grow up to ~1.75 and blow past max_delay.
        mock_time.monotonic.return_value = 999.9999
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_put.return_value = fail_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        max_delay = 30.0
        for sleep_call in mock_time.sleep.call_args_list:
            actual_delay = sleep_call[0][0]
            self.assertLess(actual_delay, max_delay)
            self.assertGreaterEqual(actual_delay, 0.75)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_payload_contains_required_fields(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        call_kwargs = mock_put.call_args
        payload = call_kwargs[1]["json"]
        required_fields = [
            "attn_tp_size",
            "attn_tp_rank",
            "attn_cp_size",
            "attn_cp_rank",
            "attn_dp_size",
            "attn_dp_rank",
            "pp_size",
            "pp_rank",
            "system_dp_size",
            "system_dp_rank",
            "rank_ip",
            "rank_port",
            "page_size",
            "kv_cache_dtype",
            # Self-registered HTTP API port used to derive the PD retract
            # rebootstrap /generate URL on the decode side.
            "prefill_http_port",
        ]
        for field in required_fields:
            self.assertIn(field, payload)
        self.assertEqual(payload["prefill_http_port"], 30000)
        self.assertEqual(payload["dcp_kv_layouts"], [DCP_KV_LAYOUT_TOKEN_STRIPE_V1])

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_url_with_dist_init_addr(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager(dist_init_addr="10.0.0.1:12345")
        mgr.register_to_bootstrap()

        url_used = mock_put.call_args[0][0]
        self.assertIn("10.0.0.1", url_used)

    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    @patch("sglang.srt.disaggregation.common.conn.get_world_group")
    def test_rust_attention_dp_replicates_complete_topology_across_hosts(
        self, mock_world_group, mock_put
    ):
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        schedulers = (
            (0, 0, "10.0.0.1", 17000, 8765),
            (0, 1, "10.0.0.1", 17001, None),
            (1, 0, "10.0.0.2", 17002, 8766),
            (1, 1, "10.0.0.2", 17003, None),
        )

        def gather_topology(payload):
            return [
                {
                    **payload,
                    "attn_dp_rank": dp_rank,
                    "attn_tp_rank": tp_rank,
                    "rank_ip": host,
                    "rank_port": rank_port,
                }
                for dp_rank, tp_rank, host, rank_port, _ in schedulers
            ]

        mock_world_group.return_value.all_gather_object.side_effect = gather_topology

        with envs.SGLANG_RUST_SERVER.override(True):
            for dp_rank, tp_rank, local_ip, _, rust_http_port in schedulers:
                manager = self._make_manager()
                manager.attn_dp_size = 2
                manager.attn_dp_rank = dp_rank
                manager.attn_tp_size = 2
                manager.attn_tp_rank = tp_rank
                manager.local_ip = local_ip
                manager.bootstrap_host = local_ip
                manager.kv_args.rust_http_port = rust_http_port
                manager.register_to_bootstrap()

        topology_by_registry = {}
        for put_call in mock_put.call_args_list:
            payload = put_call.kwargs["json"]
            topology_by_registry.setdefault(put_call.args[0], {})[
                (payload["attn_dp_rank"], payload["attn_tp_rank"])
            ] = (payload["rank_ip"], payload["rank_port"])
        complete_topology = {
            (dp, tp): (host, rank_port) for dp, tp, host, rank_port, _ in schedulers
        }
        self.assertEqual(
            topology_by_registry,
            {
                "http://10.0.0.1:8765/route": complete_topology,
                "http://10.0.0.2:8766/route": complete_topology,
            },
        )
        self.assertEqual(mock_put.call_count, 8)
        self.assertEqual(
            {
                (put_call.args[0], put_call.kwargs["json"]["prefill_http_port"])
                for put_call in mock_put.call_args_list
            },
            {
                ("http://10.0.0.1:8765/route", 8765),
                ("http://10.0.0.2:8766/route", 8766),
            },
        )
        self.assertEqual(
            [
                (
                    gather_call.args[0]["attn_dp_rank"],
                    gather_call.args[0]["attn_tp_rank"],
                )
                for gather_call in mock_world_group.return_value.all_gather_object.call_args_list
            ],
            [(dp, tp) for dp, tp, _, _, _ in schedulers],
        )

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_wildcard_host_0000_uses_ipv4_loopback(self, mock_put, mock_time):
        """When --host 0.0.0.0 is used, the PUT must target IPv4 loopback.

        Scenario: cross-node P/D disagg where each role runs on a single node
        (tp=1).  Each machine runs its own SGLang instance with --host 0.0.0.0
        to accept remote connections.  dist_init_addr is None because tp=1
        needs no multi-node rendezvous, so register_to_bootstrap takes the
        else-branch and would use bootstrap_host="0.0.0.0" as the PUT target.
        aiohttp >=3.9 rejects that with HTTP 403 because 0.0.0.0 is not a
        valid Host header value.

        Fix: substitute same-family loopback when bootstrap_host is a wildcard.
        """
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager()
        mgr.bootstrap_host = "0.0.0.0"
        mgr.local_ip = "192.168.1.10"
        mgr.register_to_bootstrap()

        url_used = mock_put.call_args[0][0]
        self.assertNotIn("0.0.0.0", url_used)
        self.assertIn("127.0.0.1", url_used)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_wildcard_host_ipv6_uses_ipv6_loopback(self, mock_put, mock_time):
        """Same fix for the IPv6 wildcard \"::\": must use IPv6 loopback."""
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager()
        mgr.bootstrap_host = "::"
        mgr.local_ip = "fd00::1"
        mgr.register_to_bootstrap()

        url_used = mock_put.call_args[0][0]
        # "::" bracketed as "[::]:port" should not appear; loopback should.
        self.assertNotIn("[::]", url_used)
        self.assertIn("[::1]", url_used)

    def _make_manager(self, dist_init_addr=None):
        """Create a lightweight mock manager that has the attributes needed
        by register_to_bootstrap, without going through CommonKVManager.__init__
        (which requires zmq, ServerArgs model resolution, etc.)."""
        mgr = MagicMock(spec=CommonKVManager)
        # Bind the real method to the mock
        mgr.register_to_bootstrap = CommonKVManager.register_to_bootstrap.__get__(
            mgr, CommonKVManager
        )
        mgr._register_topology_row = CommonKVManager._register_topology_row.__get__(
            mgr, CommonKVManager
        )
        mgr._local_dcp_kv_layouts = CommonKVManager._local_dcp_kv_layouts.__get__(
            mgr, CommonKVManager
        )

        # Set attributes that register_to_bootstrap reads
        mgr.dist_init_addr = dist_init_addr
        mgr.bootstrap_host = "127.0.0.1"
        mgr.bootstrap_port = 8765
        mgr.attn_tp_size = 1
        mgr.attn_tp_rank = 0
        mgr.attn_cp_size = 1
        mgr.attn_cp_rank = 0
        mgr.attn_dp_size = 1
        mgr.attn_dp_rank = 0
        mgr.pp_size = 1
        mgr.pp_rank = 0
        mgr.system_dp_size = 1
        mgr.system_dp_rank = 0
        mgr.local_ip = "127.0.0.1"
        mgr.rank_port = 12345

        mgr.kv_args = MagicMock()
        mgr.kv_args.page_size = 16
        mgr.kv_args.rust_http_port = None
        # Resolved per-runner value threaded through KVArgs (the payload field).
        mgr.kv_cache_dtype_str = "auto"

        return mgr


class _RouteRequest:
    def __init__(self, *, data=None, query=None):
        self._data = data
        self.query = query or {}

    async def json(self):
        return self._data


class TestBootstrapDcpKvCapabilities(CustomTestCase):
    def _make_server(self, *, dp_size=1, tp_size=1):
        server = object.__new__(CommonKVBootstrapServer)
        server.attn_tp_size = tp_size
        server.attn_cp_size = 1
        server.dp_size = dp_size
        server.pp_size = 1
        server.page_size = 16
        server.kv_cache_dtype = "auto"
        server.follow_bootstrap_room = True
        server.enable_dsa_cache_layer_split = False
        server.prefill_http_port = None
        server.prefill_port_table = {}
        server._registered_count = 0
        server.lock = asyncio.Lock()
        return server

    @staticmethod
    def _registration(*, dp_rank=0, tp_rank=0, dcp_kv_layouts=None):
        data = {
            "attn_tp_size": 2,
            "attn_tp_rank": tp_rank,
            "attn_cp_size": 1,
            "attn_cp_rank": 0,
            "attn_dp_size": 1,
            "attn_dp_rank": dp_rank,
            "pp_size": 1,
            "pp_rank": 0,
            "system_dp_size": 1,
            "system_dp_rank": 0,
            "rank_ip": "127.0.0.1",
            "rank_port": 12345 + tp_rank,
            "page_size": 16,
            "kv_cache_dtype": "auto",
        }
        if dcp_kv_layouts is not None:
            data["dcp_kv_layouts"] = dcp_kv_layouts
        return data

    @staticmethod
    def _static_query(*, want_caps=False):
        query = {
            "prefill_dp_rank": "-1",
            "prefill_cp_rank": "-1",
            "target_tp_rank": "-1",
            "target_pp_rank": "-1",
        }
        if want_caps:
            query["want_dcp_kv_caps"] = "1"
        return query

    def test_route_caps_are_opt_in_and_intersect_all_ranks(self):
        server = self._make_server(tp_size=2)
        asyncio.run(
            server._handle_route_put(
                _RouteRequest(
                    data=self._registration(
                        tp_rank=0,
                        dcp_kv_layouts=[
                            DCP_KV_LAYOUT_TOKEN_STRIPE_V1,
                            DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1,
                        ],
                    )
                )
            )
        )
        asyncio.run(
            server._handle_route_put(
                _RouteRequest(
                    data=self._registration(
                        tp_rank=1,
                        dcp_kv_layouts=[DCP_KV_LAYOUT_TOKEN_STRIPE_V1],
                    )
                )
            )
        )

        legacy = asyncio.run(
            server._handle_route_get(_RouteRequest(query=self._static_query()))
        )
        self.assertEqual(legacy.status, 200)
        self.assertNotIn("dcp_kv_layouts", json.loads(legacy.text))

        caps = asyncio.run(
            server._handle_route_get(
                _RouteRequest(query=self._static_query(want_caps=True))
            )
        )
        self.assertEqual(
            json.loads(caps.text)["dcp_kv_layouts"], [DCP_KV_LAYOUT_TOKEN_STRIPE_V1]
        )

    def test_route_caps_ignore_malformed_rank_advertisement(self):
        server = self._make_server(tp_size=2)
        for tp_rank, layouts in (
            (0, [DCP_KV_LAYOUT_TOKEN_STRIPE_V1, DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1]),
            (1, DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1),
        ):
            asyncio.run(
                server._handle_route_put(
                    _RouteRequest(
                        data=self._registration(tp_rank=tp_rank, dcp_kv_layouts=layouts)
                    )
                )
            )

        response = asyncio.run(
            server._handle_route_get(
                _RouteRequest(query=self._static_query(want_caps=True))
            )
        )
        self.assertNotIn("dcp_kv_layouts", json.loads(response.text))

    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    @patch("sglang.srt.disaggregation.common.conn.get_parallel")
    def test_page_decode_requests_caps_and_rejects_missing_page_capability(
        self, mock_parallel, mock_get
    ):
        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=16)
        manager.kv_cache_dtype_str = "auto"
        manager.dcp_size = 1
        manager._resolve_rank_mapping = MagicMock()
        mock_parallel.return_value = SimpleNamespace(dcp_kv_layout="page")
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "attn_tp_size": 1,
            "attn_cp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "page_size": 16,
            "kv_cache_dtype": "auto",
            "follow_bootstrap_room": True,
        }
        mock_get.return_value = response

        self.assertFalse(manager.try_ensure_parallel_info("127.0.0.1:30000"))
        self.assertIn("want_dcp_kv_caps=1", mock_get.call_args.args[0])
        self.assertEqual(manager.prefill_info_table, {})

    def test_page_invalidation_drops_only_its_cached_capability_generation(self):
        receiver = object.__new__(MooncakeKVReceiver)
        info = object()
        replacement = object()
        receiver.bootstrap_addr = "127.0.0.1:30000"
        receiver.prefill_info = info
        receiver._is_page_layout = True
        receiver._page_registration_records = {}
        receiver._connection_pool_entries = {"key": [{"rank_ip": "127.0.0.1"}]}
        receiver.kv_mgr = SimpleNamespace(
            connection_lock=threading.Lock(),
            connection_pool={"key": receiver._connection_pool_entries["key"]},
            prefill_info_table={receiver.bootstrap_addr: info},
            _page_registration_states={},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {})
        self.assertNotIn(receiver.bootstrap_addr, receiver.kv_mgr.prefill_info_table)

        receiver.kv_mgr.prefill_info_table[receiver.bootstrap_addr] = replacement
        receiver.invalidate_cached_bootstrap_infos()
        self.assertIs(
            receiver.kv_mgr.prefill_info_table[receiver.bootstrap_addr], replacement
        )

    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    @patch("sglang.srt.disaggregation.common.conn.get_parallel")
    def test_page_reconnect_refreshes_evicted_capability_after_cache_reuse(
        self, mock_parallel, mock_get
    ):
        bootstrap_addr = "127.0.0.1:30000"
        cached_info = object()
        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {bootstrap_addr: cached_info}
        manager.connection_lock = threading.Lock()
        manager.connection_pool = {}
        manager._page_registration_states = {}
        manager.kv_args = SimpleNamespace(page_size=16)
        manager.kv_cache_dtype_str = "auto"
        manager.dcp_size = 1
        manager._resolve_rank_mapping = MagicMock()
        mock_parallel.return_value = SimpleNamespace(dcp_kv_layout="page")

        self.assertTrue(manager.try_ensure_parallel_info(bootstrap_addr))
        mock_get.assert_not_called()

        receiver = object.__new__(MooncakeKVReceiver)
        receiver.bootstrap_addr = bootstrap_addr
        receiver.prefill_info = cached_info
        receiver._is_page_layout = True
        receiver._page_registration_records = {}
        receiver._connection_pool_entries = {}
        receiver.kv_mgr = manager
        receiver.invalidate_cached_bootstrap_infos()

        response = MagicMock(status_code=200)
        response.json.return_value = {
            "attn_tp_size": 1,
            "attn_cp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "page_size": 16,
            "kv_cache_dtype": "auto",
            "follow_bootstrap_room": True,
            "dcp_kv_layouts": [DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1],
        }
        mock_get.return_value = response

        self.assertTrue(manager.try_ensure_parallel_info(bootstrap_addr))
        self.assertIn("want_dcp_kv_caps=1", mock_get.call_args.args[0])
        self.assertIsNot(manager.prefill_info_table[bootstrap_addr], cached_info)

    @patch("sglang.srt.arg_groups.overrides.model_config_of")
    def test_k3_hybrid_linear_pool_advertises_page_capability(self, model_config_of):
        manager = object.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.dcp_size = 1
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = True
        manager.kv_args = SimpleNamespace(num_draft_entries=0)
        manager.server_args = object()
        model_config_of.return_value = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"])
        )

        self.assertEqual(
            manager._local_dcp_kv_layouts(),
            [DCP_KV_LAYOUT_TOKEN_STRIPE_V1, DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1],
        )

    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    @patch("sglang.srt.disaggregation.common.conn.get_parallel")
    def test_decode_memory_reregistration_evicts_page_capability(
        self, mock_parallel, mock_get
    ):
        manager = object.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.DECODE
        manager._registerable_regions = lambda: []
        manager.connection_lock = threading.Lock()
        manager.connection_pool = {"cached": object()}
        manager._page_registration_generation = 4
        manager._page_registration_states = {"registration": object()}
        page_addr = "page:30000"
        token_addr = "token:30000"
        manager.prefill_info_table = {
            page_addr: SimpleNamespace(
                dcp_kv_layouts=[DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1]
            ),
            token_addr: SimpleNamespace(dcp_kv_layouts=None),
        }
        manager.kv_args = SimpleNamespace(page_size=16)
        manager.kv_cache_dtype_str = "auto"
        manager.dcp_size = 1
        manager._resolve_rank_mapping = MagicMock()
        mock_parallel.return_value = SimpleNamespace(dcp_kv_layout="page")

        manager.deregister_buffer_to_engine()

        self.assertEqual(manager.connection_pool, {})
        self.assertEqual(manager._page_registration_generation, 5)
        self.assertEqual(manager._page_registration_states, {})
        self.assertNotIn(page_addr, manager.prefill_info_table)
        self.assertIn(token_addr, manager.prefill_info_table)

        response = MagicMock(status_code=200)
        response.json.return_value = {
            "attn_tp_size": 1,
            "attn_cp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "page_size": 16,
            "kv_cache_dtype": "auto",
            "follow_bootstrap_room": True,
            "dcp_kv_layouts": [DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1],
        }
        mock_get.return_value = response
        self.assertTrue(manager.try_ensure_parallel_info(page_addr))
        self.assertIn("want_dcp_kv_caps=1", mock_get.call_args.args[0])

    def test_token_registration_and_metadata_keep_legacy_frame_counts(self):
        class Socket:
            sent = []

            def send_multipart(self, frames):
                self.sent.append(frames)

        socket = Socket()
        receiver = object.__new__(MooncakeKVReceiver)
        receiver._is_page_layout = False
        receiver.bootstrap_infos = [
            {"rank_ip": "127.0.0.1", "rank_port": 30000, "is_dummy": False}
        ]
        receiver.session_id = "session"
        receiver.bootstrap_room = 9
        receiver.required_dst_info_num = 1
        receiver.kv_mgr = SimpleNamespace(
            kv_args=SimpleNamespace(
                kv_data_ptrs=[],
                aux_data_ptrs=[],
                state_data_ptrs=[],
                state_item_lens=[],
                state_dim_per_tensor=[],
                state_layer_ids=[],
                kv_layer_ids=[],
                engine_rank=0,
                kv_item_lens=[],
            ),
            attn_tp_size=1,
            dcp_size=1,
            dcp_rank=0,
            enable_staging=False,
            local_ip="127.0.0.1",
            rank_port=30001,
        )
        receiver._connect_to_bootstrap_server = lambda _info: (
            socket,
            threading.Lock(),
        )

        self.assertTrue(receiver._register_kv_args())
        receiver.send_metadata(np.array([1], dtype=np.int32))

        self.assertEqual([len(frames) for frames in socket.sent], [19, 10])

    def test_page_metadata_waits_for_the_shared_registration_ack(self):
        receiver = object.__new__(MooncakeKVReceiver)
        endpoint = "tcp://127.0.0.1:30000"
        key = (endpoint, "session", DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1, 16, 0)
        socket = MagicMock()
        receiver._is_page_layout = True
        receiver._page_registration_records = {
            key: PageRegistrationState("nonce", started_at=0.0)
        }
        receiver._page_metadata_sent = False
        receiver.bootstrap_infos = [
            {"rank_ip": "127.0.0.1", "rank_port": 30000, "is_dummy": False}
        ]
        receiver.session_id = "session"
        receiver.bootstrap_room = 9
        receiver.conclude_state = None
        receiver.kv_mgr = SimpleNamespace(
            connection_lock=threading.Lock(),
            _page_registration_states=dict(receiver._page_registration_records),
            kv_args=SimpleNamespace(page_size=16),
            request_status={},
            record_failure=MagicMock(),
            update_status=MagicMock(),
        )
        receiver._connect_to_bootstrap_server = MagicMock(return_value=(socket, None))

        receiver.send_metadata(np.array([1], dtype=np.int32))

        socket.send_multipart.assert_not_called()
        receiver.kv_mgr.record_failure.assert_called_once()
        receiver.kv_mgr.update_status.assert_called_once_with(9, KVPoll.Failed)

    def test_registration_extension_rejects_unknown_layout_but_legacy_token(self):
        manager = object.__new__(MooncakeKVManager)

        self.assertIsNone(
            manager._validate_page_registration(
                SimpleNamespace(dcp_kv_layout=DCP_KV_LAYOUT_TOKEN_STRIPE_V1)
            )
        )
        self.assertEqual(
            manager._validate_page_registration(
                SimpleNamespace(dcp_kv_layout="unknown_layout")
            ),
            "unsupported_dcp_kv_layout",
        )

    def test_old_receiver_invalidation_keeps_replaced_shared_registration(self):
        receiver = object.__new__(MooncakeKVReceiver)
        key = ("tcp://127.0.0.1:30000", "session", "page_interleave_v1", 16, 0)
        old_state = PageRegistrationState("old", started_at=0.0)
        new_state = PageRegistrationState("new", started_at=1.0)
        receiver.bootstrap_addr = "127.0.0.1:30000"
        receiver.prefill_info = object()
        receiver._is_page_layout = True
        receiver._page_registration_records = {key: old_state}
        receiver._connection_pool_entries = {}
        receiver.kv_mgr = SimpleNamespace(
            connection_lock=threading.Lock(),
            connection_pool={},
            _page_registration_states={key: new_state},
            prefill_info_table={receiver.bootstrap_addr: object()},
        )

        receiver.invalidate_cached_bootstrap_infos()

        self.assertIs(receiver.kv_mgr._page_registration_states[key], new_state)

    def test_shared_page_registration_waits_for_its_bound_ack(self):
        key = ("tcp://127.0.0.1:30000", "session", "page_interleave_v1", 16, 0)
        state = PageRegistrationState("nonce", started_at=0.0)
        manager = SimpleNamespace(
            connection_lock=threading.Lock(), _page_registration_states={key: state}
        )
        receivers = []
        for _ in range(2):
            receiver = object.__new__(MooncakeKVReceiver)
            receiver._is_page_layout = True
            receiver.conclude_state = None
            receiver.kv_mgr = manager
            receiver._page_registration_records = {key: state}
            receivers.append(receiver)

        self.assertFalse(receivers[0]._registration_ready())
        self.assertFalse(receivers[1]._registration_ready())
        state.accepted = True
        self.assertTrue(receivers[0]._registration_ready())
        self.assertTrue(receivers[1]._registration_ready())

    def test_decode_listener_accepts_only_matching_page_registration_ack(self):
        def state_after(message):
            key = (
                "tcp://127.0.0.1:30000",
                "session",
                DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1,
                16,
                0,
            )
            state = PageRegistrationState("nonce", started_at=0.0)
            manager = object.__new__(MooncakeKVManager)
            manager.connection_lock = threading.Lock()
            manager._page_registration_states = {key: state}
            manager.enable_deferred_decode_kv_release = False
            manager._start_heartbeat_checker_thread = lambda: None
            _run_control_message(manager, manager.start_decode_thread, message)
            return state

        payload = json.dumps(
            {
                "version": 1,
                "layout": DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1,
                "physical_page_size": 16,
            }
        ).encode()
        wrong_endpoint = [
            b"DCP_LAYOUT_REGISTER_ACK",
            b"nonce",
            b"tcp://127.0.0.1:30001",
            b"session",
            payload,
        ]
        wrong_nonce = [
            b"DCP_LAYOUT_REGISTER_ACK",
            b"other-nonce",
            b"tcp://127.0.0.1:30000",
            b"session",
            payload,
        ]
        wrong_session = [
            b"DCP_LAYOUT_REGISTER_ACK",
            b"nonce",
            b"tcp://127.0.0.1:30000",
            b"other-session",
            payload,
        ]
        matching = [
            b"DCP_LAYOUT_REGISTER_ACK",
            b"nonce",
            b"tcp://127.0.0.1:30000",
            b"session",
            payload,
        ]

        for unmatched in (wrong_endpoint, wrong_nonce, wrong_session):
            self.assertFalse(state_after(unmatched).accepted)
        self.assertTrue(state_after(matching).accepted)

    @patch("sglang.srt.arg_groups.overrides.model_config_of")
    def test_prefill_rejects_page_geometry_mismatch_before_registration(
        self, model_config_of
    ):
        model_config_of.return_value = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"])
        )
        manager = object.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.server_args = object()
        manager.kv_args = SimpleNamespace(
            page_size=4, num_draft_entries=0, kv_item_lens=[8]
        )
        manager.dcp_size = 1
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.decode_kv_args_table = {}
        manager.transfer_infos = {}
        manager._page_transfer_rooms = set()
        manager.local_ip = "127.0.0.1"
        manager.rank_port = 30000
        manager._send_multipart_locked = MagicMock()
        message = [
            b"None",
            b"127.0.0.1",
            b"30001",
            b"session",
            struct.pack("Q", 4096),
            b"",
            b"",
            b"0",
            b"2",
            b"12",
            b"",
            b"",
            struct.pack("I", 0),
            b"",
            b"",
            b"",
            b"2",
            b"0",
            b"",
            json.dumps(
                {
                    "version": 1,
                    "layout": DCP_KV_LAYOUT_PAGE_INTERLEAVE_V1,
                    "physical_page_size": 4,
                    "registration_id": "nonce",
                }
            ).encode(),
        ]

        _run_control_message(manager, manager.start_prefill_thread, message)

        self.assertEqual(manager.decode_kv_args_table, {})
        manager._send_multipart_locked.assert_called_once()
        endpoint, response = manager._send_multipart_locked.call_args.args
        self.assertEqual(endpoint, "tcp://127.0.0.1:30001")
        self.assertEqual(
            response[:4],
            [
                b"DCP_LAYOUT_REGISTER_REJECT",
                b"nonce",
                b"tcp://127.0.0.1:30000",
                b"session",
            ],
        )
        payload = json.loads(response[4])
        self.assertEqual(payload["reason"], "kv_geometry_mismatch")
        self.assertIn("KV geometry differs", payload["description"])

    @patch("sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=2.0)
    def test_page_registration_timeout_marks_the_original_shared_record(
        self, _monotonic
    ):
        key = ("tcp://127.0.0.1:30000", "session", "page_interleave_v1", 16, 0)
        state = PageRegistrationState("nonce", started_at=0.0)
        receiver = object.__new__(MooncakeKVReceiver)
        receiver._is_page_layout = True
        receiver.conclude_state = None
        receiver.bootstrap_room = 7
        receiver._page_registration_records = {key: state}
        receiver.invalidate_cached_bootstrap_infos = MagicMock()
        receiver.kv_mgr = SimpleNamespace(
            connection_lock=threading.Lock(),
            _page_registration_states={key: state},
            waiting_timeout=1.0,
            check_status=MagicMock(return_value=KVPoll.Bootstrapping),
            record_failure=MagicMock(),
            update_status=MagicMock(),
        )

        self.assertEqual(receiver.poll(), KVPoll.Failed)

        self.assertEqual(state.failed_reason, "acknowledgement_timeout")
        self.assertIs(receiver._page_registration_records[key], state)
        receiver.kv_mgr.update_status.assert_called_once_with(7, KVPoll.Failed)


if __name__ == "__main__":
    unittest.main()

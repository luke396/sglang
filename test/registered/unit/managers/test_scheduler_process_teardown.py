"""run_scheduler_process teardown when the forward cost profiler fails to close.

CPU-only: runs the real run_scheduler_process with the Scheduler, process
configuration and distributed teardown replaced by recorders.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import PortArgs, ServerArgs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRunSchedulerProcessTeardown(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        self.actions = []

    def _run(self, *, gracefully_exit):
        actions = self.actions

        class _Profiler:
            def close(self, *, drain):
                actions.append(("profile_close", drain))
                raise OSError("profile flush: disk full")

        def make_scheduler(server_args, port_args):
            s = MagicMock()
            s.gracefully_exit = gracefully_exit
            s.tp_worker.model_runner.forward_cost_profiler = _Profiler()
            s.metrics_reporter._shutdown_fpm.side_effect = lambda: actions.append(
                "fpm_shutdown"
            )
            s.release_host_resources.side_effect = lambda: actions.append(
                "release_host_resources"
            )
            return s

        server_args = ServerArgs(model_path="dummy", device="cpu")
        with (
            patch.object(scheduler_module, "load_plugins"),
            patch.object(scheduler_module, "configure_scheduler_process"),
            patch.object(scheduler_module, "psutil"),
            patch.object(scheduler_module, "Scheduler", side_effect=make_scheduler),
            patch.object(
                scheduler_module,
                "abort_distributed_environment",
                side_effect=lambda: actions.append("abort_distributed_environment"),
            ),
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                scheduler_module.run_scheduler_process(
                    server_args,
                    PortArgs.init_new(server_args),
                    gpu_id=0,
                    tp_rank=0,
                    pp_rank=0,
                    dp_rank=None,
                    pipe_writer=MagicMock(),
                )

    def test_graceful_teardown_runs_when_profiler_close_raises(self):
        """A profiler close error must not skip host release or communicator abort."""
        self._run(gracefully_exit=True)
        self.assertEqual(
            self.actions,
            [
                "fpm_shutdown",
                ("profile_close", True),
                "release_host_resources",
                "abort_distributed_environment",
            ],
        )

    def test_non_graceful_path_still_skips_device_teardown(self):
        self._run(gracefully_exit=False)
        self.assertEqual(self.actions, ["fpm_shutdown", ("profile_close", False)])


if __name__ == "__main__":
    unittest.main()

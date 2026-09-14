"""Small CPU-only transport boundary helpers shared by disaggregation tests."""

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=0,
    suite="base-a-test-cpu",
    disabled="helper module - exported CPU transport boundary, not a test",
)


class CopyTransport:
    """Execute Mooncake transfer descriptors against registered CPU buffers."""

    def __init__(self, sources, destinations):
        self.sources = sources
        self.destinations = destinations
        self.bytes_sent = 0

    @staticmethod
    def _region(buffers, address, size):
        for buffer in buffers:
            raw = np.asarray(buffer).view(np.uint8).reshape(-1)
            offset = address - raw.ctypes.data
            if 0 <= offset and offset + size <= raw.nbytes:
                return raw[offset : offset + size]
        raise AssertionError("transfer descriptor exceeds a registered CPU buffer")

    def __call__(self, _session, blocks):
        for source, destination, size in blocks:
            self._region(self.destinations, destination, size)[:] = self._region(
                self.sources, source, size
            )
            self.bytes_sent += size
        return 0

"""
Minimal stand-in for the PYNQ runtime, used ONLY by tests/driver_mock_test.py
to exercise pynq/accelerator_driver.py on a PC.  The fake MMIO emulates the
alexnet_accel register file; on ap_start it runs the bit-exact layer model
(alexnet_common.run_layer_int) on the buffers addressed by the written
physical addresses.  It is a behavioural model of the HLS IP, not the IP.
"""
import numpy as np

import alexnet_common as ac

_MEM = {}            # physical_address -> buffer
_NEXT = [0x1000_0000]
REGS = {}            # name -> offset, filled by the test from its fake header
STATE = {"ap_ctrl": 0x4, "ap_return": 0}


class PynqBuffer(np.ndarray):
    def flush(self):
        pass

    def invalidate(self):
        pass

    def freebuffer(self):
        _MEM.pop(self.physical_address, None)


def allocate(shape, dtype=np.uint64, **_):
    a = np.zeros(shape, dtype=dtype).view(PynqBuffer)
    a.physical_address = _NEXT[0]
    _NEXT[0] += (a.nbytes + 0xFFF) & ~0xFFF
    _MEM[a.physical_address] = a
    return a


def _buf(addr):
    return _MEM[addr]


class MMIO:
    def __init__(self, base, length):
        self.base, self.length, self.mem = base, length, {}

    def read(self, off):
        if off == REGS["AP_CTRL"]:
            v = STATE["ap_ctrl"]
            STATE["ap_ctrl"] &= ~0x2          # ap_done is clear-on-read
            return v
        if off == REGS["AP_RETURN"]:
            return STATE["ap_return"]
        return self.mem.get(off, 0)

    def write(self, off, val):
        if off == REGS["AP_CTRL"]:
            if val & 1:
                self._run()
            return
        self.mem[off] = val

    def _arg(self, name):
        return self.mem.get(REGS[name.upper() + "_DATA"], 0)

    def _run(self):
        g = self._arg
        L = dict(op=g("op"), cin=g("cin"), hin=g("hin"), win=g("win"), cout=g("cout"),
                 k=g("ksize"), pad=g("pad"), hout=g("hout"), wout=g("wout"),
                 pool=g("pool"), relu=g("relu"), raw=g("raw_out"))
        src, dst = _buf(g("act_in")), _buf(g("act_out"))
        w, p = _buf(g("weights")), _buf(g("params"))
        b, M, sh = ac.unpack_params(np.array(p))
        if L["op"] == ac.OP_CONV:
            wq = ac.unpack_conv_weights(np.array(w), L["cout"], L["cin"], L["k"])
        else:
            wq = np.array(w).view(np.int8).reshape(L["cout"], L["cin"])
        out = ac.run_layer_int(L, np.array(src), wq, b, M, sh)
        dst[: out.size] = out
        STATE["ap_return"] = 0
        STATE["ap_ctrl"] = 0x4 | 0x2          # idle + done


class Overlay:
    def __init__(self, path):
        self.ip_dict = {"alexnet_accel_0": {"type": "xilinx.com:hls:alexnet_accel:1.0",
                                            "phys_addr": 0x40000000, "addr_range": 0x10000,
                                            "registers": {}}}

    def free(self):
        pass

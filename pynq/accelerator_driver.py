"""
accelerator_driver.py -- PYNQ (v3.x) driver for the alexnet_accel HLS IP.

* finds the IP in the overlay by its VLNV type (no assumed instance name)
* resolves register offsets from the HLS header / .hwh (hw_header.py)
* owns all physically-contiguous buffers (pynq.allocate):
      per-layer weight + param blobs (loaded once), two ping-pong activation buffers
* runs one layer per accelerator invocation, polls ap_done with a timeout and
  checks the kernel's ap_return status word
* frees every buffer in close() (also via context manager)
"""
import os
import time

import numpy as np

import alexnet_common as ac
import hw_header

AP_START, AP_DONE, AP_IDLE = 0x1, 0x2, 0x4
STATUS_TEXT = {0: "OK", 1: "ERR_BAD_OP", 2: "ERR_CHANNELS", 3: "ERR_KERNEL", 4: "ERR_CAPACITY"}


class AcceleratorError(RuntimeError):
    pass


def find_ip(overlay, vlnv_substr):
    hits = [k for k, v in overlay.ip_dict.items() if vlnv_substr in v.get("type", "")]
    if len(hits) != 1:
        raise AcceleratorError(f"expected exactly one IP of type *{vlnv_substr}*, found {hits}. "
                               f"IPs in overlay: {sorted(overlay.ip_dict)}")
    return hits[0]


class AlexNetAccel:
    def __init__(self, overlay, deploy_dir, header_path=None, timeout_s=5.0):
        from pynq import MMIO
        self.name = find_ip(overlay, "xilinx.com:hls:alexnet_accel")
        entry = overlay.ip_dict[self.name]
        self.mmio = MMIO(entry["phys_addr"], entry["addr_range"])
        self.regs = hw_header.resolve(entry, header_path)
        self.timeout_s = timeout_s
        self.deploy_dir = deploy_dir
        self._bufs = []
        self.layer_w = []
        self.layer_p = []
        self._load_model()

    # ------------------------------------------------------------ buffers --
    def _alloc(self, n_words):
        from pynq import allocate
        b = allocate(shape=(int(n_words),), dtype=np.uint64)
        self._bufs.append(b)
        return b

    def _load_model(self):
        """Copy every layer's weights/params into its own contiguous buffer
        (largest = fc6, 37.7 MB; per-layer buffers limit CMA fragmentation)."""
        t0 = time.perf_counter()
        for i, L in enumerate(ac.HW_LAYERS):
            w = np.load(os.path.join(self.deploy_dir, f"layer{i}_weights.npy"))
            p = np.load(os.path.join(self.deploy_dir, f"layer{i}_params.npy"))
            if p.size != L["cout"]:
                raise AcceleratorError(f"layer{i}: params size {p.size} != cout {L['cout']}")
            bw, bp = self._alloc(w.size), self._alloc(p.size)
            bw[:] = w
            bp[:] = p
            bw.flush()
            bp.flush()
            self.layer_w.append(bw)
            self.layer_p.append(bp)
        n = max(max(ac.in_words(L), ac.out_words(L)) for L in ac.HW_LAYERS)
        self.act = [self._alloc(n), self._alloc(n)]
        for a in self.act:
            a[:] = 0
            a.flush()
        self.load_time_s = time.perf_counter() - t0

    def close(self):
        for b in self._bufs:
            try:
                b.freebuffer()
            except Exception:
                pass
        self._bufs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ registers --
    def _write_arg(self, name, value):
        off, bits = self.regs.args[name]
        self.mmio.write(off, int(value) & 0xFFFFFFFF)
        if bits > 32:
            self.mmio.write(off + 4, (int(value) >> 32) & 0xFFFFFFFF)

    def _read_arg(self, name):
        off, _ = self.regs.args[name]
        return self.mmio.read(off)

    def self_test_registers(self):
        """Write/read-back of plain scalar registers (proves the register map)."""
        for k, v in [("cin", 0x1234), ("hout", 0x55), ("relu", 1)]:
            self._write_arg(k, v)
            got = self._read_arg(k)
            if got != v:
                raise AcceleratorError(f"register '{k}' read back 0x{got:x}, wrote 0x{v:x}")
        return True

    def is_idle(self):
        return bool(self.mmio.read(self.regs.ap_ctrl) & AP_IDLE)

    # ------------------------------------------------------------ execution --
    def run_layer(self, i, src, dst):
        L = ac.HW_LAYERS[i]
        if not self.is_idle():
            raise AcceleratorError("accelerator not idle before start (previous run hung?)")
        self._write_arg("act_in", src.physical_address)
        self._write_arg("act_out", dst.physical_address)
        self._write_arg("weights", self.layer_w[i].physical_address)
        self._write_arg("params", self.layer_p[i].physical_address)
        for reg, key in [("op", "op"), ("cin", "cin"), ("hin", "hin"), ("win", "win"),
                         ("cout", "cout"), ("ksize", "k"), ("pad", "pad"), ("hout", "hout"),
                         ("wout", "wout"), ("pool", "pool"), ("relu", "relu"), ("raw_out", "raw")]:
            self._write_arg(reg, L[key])
        t0 = time.perf_counter()
        self.mmio.write(self.regs.ap_ctrl, AP_START)
        while True:
            ctrl = self.mmio.read(self.regs.ap_ctrl)       # ap_done is clear-on-read
            if ctrl & AP_DONE:
                break
            if time.perf_counter() - t0 > self.timeout_s:
                raise AcceleratorError(f"layer {L['name']}: timeout after {self.timeout_s}s "
                                       f"(AP_CTRL=0x{ctrl:x}); check HP-port addresses / reset")
        dt = time.perf_counter() - t0
        status = self.mmio.read(self.regs.ap_return)
        if status != 0:
            raise AcceleratorError(f"layer {L['name']}: kernel status {status} "
                                   f"({STATUS_TEXT.get(status, '?')})")
        return dt

    def infer(self, input_words, keep_intermediates=False):
        """Full AlexNet: 8 accelerator invocations, ping-pong activations.
        Returns (fc8 raw int32[1000], timing dict, optional per-layer outputs)."""
        timing = {}
        t0 = time.perf_counter()
        src, dst = self.act
        src[: input_words.size] = input_words
        src.flush()
        timing["input_copy"] = time.perf_counter() - t0
        inter = []
        for i, L in enumerate(ac.HW_LAYERS):
            timing[L["name"]] = self.run_layer(i, src, dst)
            if keep_intermediates:
                dst.invalidate()
                inter.append(np.array(dst[: ac.out_words(L)]))
            src, dst = dst, src
        t1 = time.perf_counter()
        src.invalidate()
        raw = ac.unpack_raw_i32(np.array(src[:500]), 1000)
        timing["output_copy"] = time.perf_counter() - t1
        timing["accelerator_total"] = sum(timing[L["name"]] for L in ac.HW_LAYERS)
        timing["end_to_end"] = time.perf_counter() - t0
        return raw, timing, inter

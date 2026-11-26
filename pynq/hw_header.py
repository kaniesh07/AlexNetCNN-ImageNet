"""
hw_header.py -- resolve the alexnet_accel AXI4-Lite register map WITHOUT
hard-coding offsets.

Primary source : the Vitis HLS generated header xalexnet_accel_hw.h
                 (copied next to the overlay by vivado/build.tcl)
Secondary      : the register list PYNQ parses from the .hwh
                 (overlay.ip_dict[<ip>]['registers'])
When both are available they are cross-checked and any disagreement is an
error (it means the .bit/.hwh and the header come from different builds).
"""
import re

REQUIRED_ARGS = ["act_in", "act_out", "weights", "params", "op", "cin", "hin", "win",
                 "cout", "ksize", "pad", "hout", "wout", "pool", "relu", "raw_out"]


class RegMap:
    def __init__(self, ctrl, ret, args, source):
        self.ap_ctrl = ctrl          # offset of AP_CTRL
        self.ap_return = ret         # offset of AP_RETURN
        self.args = args             # name -> (offset, bits)
        self.source = source

    def __repr__(self):
        rows = [f"  AP_CTRL   0x{self.ap_ctrl:02x}", f"  AP_RETURN 0x{self.ap_return:02x}"]
        rows += [f"  {k:9s} 0x{o:02x} ({b} bit)" for k, (o, b) in self.args.items()]
        return f"RegMap(source={self.source})\n" + "\n".join(rows)


def parse_hls_header(path):
    txt = open(path).read()
    addr = {m.group(1): int(m.group(2), 16) for m in
            re.finditer(r"#define\s+X\w+?_CONTROL_ADDR_(\w+)\s+0x([0-9a-fA-F]+)", txt)}
    bits = {m.group(1): int(m.group(2)) for m in
            re.finditer(r"#define\s+X\w+?_CONTROL_BITS_(\w+)\s+(\d+)", txt)}
    if "AP_CTRL" not in addr or "AP_RETURN" not in addr:
        raise ValueError(f"{path}: AP_CTRL/AP_RETURN not found -- not an alexnet_accel header?")
    args = {}
    for name in REQUIRED_ARGS:
        key = name.upper() + "_DATA"
        if key not in addr:
            raise ValueError(f"{path}: register for argument '{name}' missing")
        args[name] = (addr[key], bits.get(key, 32))
    return RegMap(addr["AP_CTRL"], addr["AP_RETURN"], args, path)


def parse_hwh_registers(ip_entry):
    """ip_entry = overlay.ip_dict[name]; returns RegMap or None."""
    regs = ip_entry.get("registers") or {}
    if not regs:
        return None
    low = {k.lower(): v for k, v in regs.items()}

    def off(k):
        return int(low[k]["address_offset"])
    ctrl = off("ctrl") if "ctrl" in low else None
    ret = off("ap_return") if "ap_return" in low else None
    args = {}
    for name in REQUIRED_ARGS:
        if name in low:
            args[name] = (off(name), int(low[name].get("size", 32)))
        elif name + "_1" in low:               # 64-bit pointer split into _1/_2
            args[name] = (off(name + "_1"), 64 if name + "_2" in low else 32)
        else:
            return None
    if ctrl is None or ret is None:
        return None
    return RegMap(ctrl, ret, args, "hwh")


def resolve(ip_entry, header_path=None):
    hdr = parse_hls_header(header_path) if header_path else None
    hwh = parse_hwh_registers(ip_entry)
    if hdr and hwh:
        bad = [k for k in REQUIRED_ARGS if hdr.args[k][0] != hwh.args[k][0]]
        if bad or hdr.ap_ctrl != hwh.ap_ctrl or hdr.ap_return != hwh.ap_return:
            raise RuntimeError("register map mismatch between xalexnet_accel_hw.h and .hwh "
                               f"for {bad}: header and bitstream come from different builds")
    rm = hdr or hwh
    if rm is None:
        raise RuntimeError("no register map: copy xalexnet_accel_hw.h next to the overlay")
    return rm

# build.tcl -- synthesis, implementation, timing/utilisation checks, bitstream,
# and PYNQ overlay packaging (alexnet.bit + alexnet.hwh + register header).
#
#   vivado -mode batch -source vivado/build.tcl          (after create_project.tcl)
# Env: JOBS (default 4)
# Output: build/overlay/{alexnet.bit, alexnet.hwh, xalexnet_accel_hw.h}
#         build/reports/{utilization.rpt, timing_summary.rpt, power.rpt}

set root     [file normalize [file join [file dirname [info script]] ..]]
set proj     [file join $root build vivado alexnet_pynq.xpr]
set out_dir  [file join $root build overlay]
set rpt_dir  [file join $root build reports]
set jobs 4
if {[info exists ::env(JOBS)]} { set jobs $::env(JOBS) }
file mkdir $out_dir $rpt_dir

if {[catch {current_project}]} { open_project $proj }

reset_run synth_1
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "synthesis failed" }

launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "implementation failed" }

open_run impl_1
report_utilization    -file [file join $rpt_dir utilization.rpt]
report_utilization    -hierarchical -file [file join $rpt_dir utilization_hier.rpt]
report_timing_summary -max_paths 20 -file [file join $rpt_dir timing_summary.rpt]
report_power          -file [file join $rpt_dir power.rpt]

set wns [get_property STATS.WNS [get_runs impl_1]]
set whs [get_property STATS.WHS [get_runs impl_1]]
puts "TIMING: WNS = $wns ns, WHS = $whs ns"
if {$wns < 0 || $whs < 0} {
    puts "ERROR: timing NOT met -- do not deploy this bitstream (see docs/verification.md, 'Timing')"
    exit 2
}

# --- package for PYNQ: .bit and .hwh MUST share the base name -------------
set bit [lindex [glob [file join $root build vivado alexnet_pynq.runs impl_1 *.bit]] 0]
set hwh [lindex [glob [file join $root build vivado alexnet_pynq.gen sources_1 bd design_1 hw_handoff design_1.hwh]] 0]
file copy -force $bit [file join $out_dir alexnet.bit]
file copy -force $hwh [file join $out_dir alexnet.hwh]
set hdr [file join $root hls alexnet_hls solution1 impl misc drivers alexnet_accel_v1_0 src xalexnet_accel_hw.h]
if {[file exists $hdr]} {
    file copy -force $hdr [file join $out_dir xalexnet_accel_hw.h]
} else {
    puts "WARNING: $hdr not found; the PYNQ driver will fall back to the .hwh register map"
}
# optional: Vitis/XSA hand-off for other flows
write_hw_platform -fixed -include_bit -force [file join $out_dir alexnet.xsa]
puts "DONE: overlay files in $out_dir"

set src [file dirname [file normalize [info script]]]
set root [file dirname $src]
set build [file join $src build]
set board_dir [file join [file dirname $root] pynq]
file mkdir $build

# Open a copy because Vivado can refresh module-reference metadata on open.
set template [file join $build baseline_[pid]]
file mkdir $template
file copy [file join $root Mixed_Precision_Acc.xpr] $template
file copy [file join $root Mixed_Precision_Acc.srcs] $template
open_project [file join $template Mixed_Precision_Acc.xpr]
set part [get_property PART [current_project]]
set board [get_property BOARD_PART [current_project]]
open_bd_design [get_files */conv1_bd.bd]
set exported [file join $build wiring.tcl]
write_bd_tcl -force $exported
close_project
file delete -force $template
set handle [open $exported r]
set wiring [string map {conv1_mmio programmable_mmio conv1_bd programmable_bd} [read $handle]]
close $handle
set handle [open $exported w]
puts $handle $wiring
close $handle

create_project -force programmable [file join $build project] -part $part
if {$board ne "" && [llength [get_board_parts -quiet $board]]} {
    set_property BOARD_PART $board [current_project]
}
add_files [list [file join $src programmable_mmio.v] \
    [file join $root Mixed_Precision_Acc.srcs sources_1 new dotprod.v]]
source $exported
set bd [get_files */programmable_bd.bd]
set_property synth_checkpoint_mode None $bd
generate_target all $bd
add_files [make_wrapper -files $bd -top]
set_property top programmable_bd_wrapper [current_fileset]
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "Implementation failed; see the impl_1 run log."
}
open_run impl_1
report_timing_summary -file [file join $build timing.rpt]
report_utilization -file [file join $build utilization.rpt]
set worst [get_timing_paths -max_paths 1]
set hold [get_timing_paths -delay_type min -max_paths 1]
if {[llength $worst] == 0 || [llength $hold] == 0 ||
    [get_property SLACK $worst] < 0 || [get_property SLACK $hold] < 0} {
    error "Timing failed; see timing.rpt."
}
file mkdir $board_dir
file copy -force [file join $build project programmable.runs impl_1 programmable_bd_wrapper.bit] \
    [file join $board_dir programmable.bit]
file copy -force [file join $build project programmable.gen sources_1 bd programmable_bd hw_handoff programmable_bd.hwh] \
    [file join $board_dir programmable.hwh]

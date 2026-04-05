##
## DevWorkspace — 5-tab dev/diagnostics panel.
## Tabs: Reward | Sleep/Dream | Hardcoded | Pipeline | Plumbing
##
extends VBoxContainer

const _RewardSystemPanel      = preload("res://scripts/panels/dev/RewardSystemPanel.gd")
const _SleepDreamPanel        = preload("res://scripts/panels/dev/SleepDreamPanel.gd")
const _HardcodedPanel         = preload("res://scripts/panels/dev/HardcodedPanel.gd")
const _PipelineDiagnosticsPanel = preload("res://scripts/panels/dev/PipelineDiagnosticsPanel.gd")
const _PlumbingTestPanel      = preload("res://scripts/panels/dev/PlumbingTestPanel.gd")

var _tabs   : TabContainer
var _panels : Array = []

func _ready() -> void:
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_theme_constant_override("separation", 0)
	_build_ui()


func _build_ui() -> void:
	_tabs = TabContainer.new()
	_tabs.size_flags_vertical = SIZE_EXPAND_FILL
	_tabs.add_theme_constant_override("side_margin", 0)
	add_child(_tabs)

	var reward : Control = _RewardSystemPanel.new()
	reward.name = "Reward"
	var sleep  : Control = _SleepDreamPanel.new()
	sleep.name = "Sleep / Dream"
	var hc     : Control = _HardcodedPanel.new()
	hc.name    = "Hardcoded"
	var pipe   : Control = _PipelineDiagnosticsPanel.new()
	pipe.name  = "Pipeline"
	var plumb  : Control = _PlumbingTestPanel.new()
	plumb.name = "Plumbing"

	_panels = [reward, sleep, hc, pipe, plumb]
	for p in _panels:
		_tabs.add_child(p)

	_tabs.tab_changed.connect(_on_tab_changed)

	# Kick off initial load on the first visible tab
	reward.refresh()


func _on_tab_changed(tab: int) -> void:
	if tab >= 0 and tab < _panels.size():
		var p : Control = _panels[tab]
		if p.has_method("refresh"):
			p.refresh()

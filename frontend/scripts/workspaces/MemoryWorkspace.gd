##
## MemoryWorkspace — 3-tier memory browser.
## Top-level TabContainer: STMM | MTMM | LTMM
##
extends Control

const _STMMPanel = preload("res://scripts/panels/memory/STMMPanel.gd")
const _MTMMPanel = preload("res://scripts/panels/memory/MTMMPanel.gd")
const _LTMMPanel = preload("res://scripts/panels/memory/LTMMPanel.gd")

var _tabs   : TabContainer
var _panels : Array = []

func _ready() -> void:
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_build_ui()


func _build_ui() -> void:
	_tabs = TabContainer.new()
	_tabs.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(_tabs)

	var stmm : Control = _STMMPanel.new()
	stmm.name = "STMM"
	var mtmm : Control = _MTMMPanel.new()
	mtmm.name = "MTMM"
	var ltmm : Control = _LTMMPanel.new()
	ltmm.name = "LTMM"

	_panels = [stmm, mtmm, ltmm]
	for p in _panels:
		_tabs.add_child(p)

	_tabs.tab_changed.connect(_on_tab_changed)

	# Load STMM immediately on open
	stmm.refresh()


func _on_tab_changed(tab: int) -> void:
	if tab >= 0 and tab < _panels.size():
		var p : Control = _panels[tab]
		if p.has_method("refresh"):
			p.refresh()

##
## StatsPanel — collapsible right panel with 3 tabs (Reward / Neurochem / Engines).
##
class_name StatsPanel
extends Control

const EXPANDED_W   := 320.0
const ANIM_SECONDS := 0.18

const _RewardTab   = preload("res://scripts/panels/tabs/RewardTab.gd")
const _NeurochemTab = preload("res://scripts/panels/tabs/NeurochemTab.gd")
const _EnginesTab  = preload("res://scripts/panels/tabs/EnginesTab.gd")

var _expanded   : bool  = false
var _tabs       : TabContainer
var _reward_tab : Control
var _neuro_tab  : Control
var _engines_tab: Control
var _turn_index : int   = 0

# ---------------------------------------------------------------------------

func _ready() -> void:
	_build_ui()
	ZADOSClient.turn_complete.connect(_on_turn_complete)
	ZADOSClient.turn_phase_updated.connect(_on_phase_updated)


func _build_ui() -> void:
	clip_contents = true

	# Left border line
	var style := StyleBoxFlat.new()
	style.bg_color            = Color(0.08, 0.08, 0.10)
	style.border_width_left   = 1
	style.border_color        = Color(0.18, 0.18, 0.22)
	add_theme_stylebox_override("panel", style)

	_tabs = TabContainer.new()
	_tabs.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_tabs.add_theme_constant_override("side_margin", 0)
	add_child(_tabs)

	_reward_tab  = _RewardTab.new()
	_reward_tab.name = "Reward"
	_tabs.add_child(_reward_tab)

	_neuro_tab   = _NeurochemTab.new()
	_neuro_tab.name = "Neurochem"
	_tabs.add_child(_neuro_tab)

	_engines_tab = _EnginesTab.new()
	_engines_tab.name = "Engines"
	_tabs.add_child(_engines_tab)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

func toggle() -> void:
	_expanded = not _expanded
	var target := EXPANDED_W if _expanded else 0.0
	var tween  := create_tween()
	tween.set_ease(Tween.EASE_OUT)
	tween.set_trans(Tween.TRANS_QUART)
	tween.tween_property(self, "custom_minimum_size:x", target, ANIM_SECONDS)


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

func _on_turn_complete(result: Dictionary) -> void:
	_turn_index += 1
	_reward_tab.refresh(result)

	var state    : Dictionary = result.get("state", {})
	var mod      : Dictionary = state.get("modulation", {})
	var dispatch : Dictionary = state.get("dispatch", {})

	_neuro_tab.add_turn(_turn_index, mod)
	_engines_tab.refresh(_turn_index, dispatch)


func _on_phase_updated(phase: int, data: Dictionary) -> void:
	# Pass phase 3 dispatch data to the engines tab immediately
	# so the grid updates as the streaming turn progresses.
	if phase == 3:
		_engines_tab.refresh(_turn_index + 1, data)

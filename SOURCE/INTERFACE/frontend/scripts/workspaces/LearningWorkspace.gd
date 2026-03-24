##
## LearningWorkspace — Mode selector (left) + Pipeline view (right).
##
extends HSplitContainer

const _ModeSelector = preload("res://scripts/panels/learning/ModeSelector.gd")
const _PipelineView = preload("res://scripts/panels/learning/PipelineView.gd")

var _mode_selector : Control
var _pipeline_view : Control

func _ready() -> void:
	split_offset = 230
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_build_ui()
	ZADOSClient.session_state_received.connect(_on_session_state)
	ZADOSClient.get_session_state()


func _build_ui() -> void:
	_mode_selector = _ModeSelector.new()
	_mode_selector.custom_minimum_size = Vector2(220, 0)
	add_child(_mode_selector)

	_pipeline_view = _PipelineView.new()
	_pipeline_view.size_flags_horizontal = SIZE_EXPAND_FILL
	add_child(_pipeline_view)

	_mode_selector.mode_selected.connect(_on_mode_selected)


func _on_mode_selected(mode_key: String) -> void:
	_pipeline_view.show_mode(mode_key)


func _on_session_state(state: Dictionary) -> void:
	var mode : String = state.get("initial_mode", "Normal")
	_pipeline_view.show_mode(mode)

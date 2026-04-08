##
## MapWorkspace — Inspector (left) + Force-directed GraphCanvas (right).
##
## Addendum B.5.2: Live Link wiring (cap warning forwarding).
## Addendum B.5.5: Keyboard shortcuts (Ctrl+F search, Space restart, Delete remove).
##
extends HSplitContainer

const _GraphCanvas   = preload("res://scripts/panels/map/GraphCanvas.gd")
const _GraphInspector = preload("res://scripts/panels/map/GraphInspector.gd")

var _inspector : Control
var _canvas    : Control

func _ready() -> void:
	split_offset = 260
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_build_ui()


func _build_ui() -> void:
	_inspector = _GraphInspector.new()
	_inspector.custom_minimum_size = Vector2(240, 0)
	add_child(_inspector)

	_canvas = _GraphCanvas.new()
	_canvas.size_flags_horizontal = SIZE_EXPAND_FILL
	_canvas.size_flags_vertical   = SIZE_EXPAND_FILL
	add_child(_canvas)

	_inspector.source_changed.connect(_on_source_changed)
	_canvas.node_selected.connect(_inspector.show_node)
	ZADOSClient.map_data_received.connect(_on_map_data)


## B.5.5: Keyboard shortcuts for the Map workspace.
func _unhandled_key_input(event: InputEvent) -> void:
	if not is_visible_in_tree():
		return
	if event is InputEventKey and event.pressed:
		var key := event as InputEventKey
		# Ctrl+F → focus search / filter in inspector
		if key.ctrl_pressed and key.keycode == KEY_F:
			if _inspector.has_method("focus_search"):
				_inspector.focus_search()
			get_viewport().set_input_as_handled()
		# Space → restart simulation
		elif key.keycode == KEY_SPACE and not key.ctrl_pressed:
			# Only when no text field is focused
			var focused := get_viewport().gui_get_focus_owner()
			if focused == null or not (focused is LineEdit or focused is TextEdit):
				if _canvas.has_method("restart_simulation"):
					_canvas.restart_simulation()
				get_viewport().set_input_as_handled()
		# Delete → remove selected node
		elif key.keycode == KEY_DELETE:
			var focused := get_viewport().gui_get_focus_owner()
			if focused == null or not (focused is LineEdit or focused is TextEdit):
				if _inspector.has_method("delete_selected"):
					_inspector.delete_selected()
				get_viewport().set_input_as_handled()


func _on_source_changed(key: String) -> void:
	_canvas.clear()
	ZADOSClient.get_map(key)


func _on_map_data(key: String, data: Dictionary) -> void:
	# Ignore knowledge_maps list (inspector handles that itself)
	if key == "knowledge_maps":
		return
	_canvas.load_graph(data)
	_inspector.update_stats(
		data.get("node_count", 0),
		data.get("edge_count",  0))
	# B.5.2: Forward cap state from canvas to inspector
	if _canvas.has_method("is_capped") and _canvas.is_capped():
		_inspector.show_cap_warning(
			_canvas.get_node_count(),
			_canvas.get_total_atom_count())

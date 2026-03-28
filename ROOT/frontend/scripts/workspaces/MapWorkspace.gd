##
## MapWorkspace — Inspector (left) + Force-directed GraphCanvas (right).
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

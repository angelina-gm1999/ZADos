##
## GraphInspector — Source selector, graph stats, and selected-node details.
## Emits source_changed(key) when the user picks a different data source.
##
extends VBoxContainer

signal source_changed(key: String)

var _source_btn  : OptionButton
var _stats_lbl   : Label
var _node_lbl    : Label
var _type_lbl    : Label
var _str_bar     : ProgressBar
var _conf_bar    : ProgressBar
var _sti_bar     : ProgressBar
var _meta_grid   : GridContainer

# map_id → option index (for knowledge maps added dynamically)
var _map_options : Dictionary = {}

func _ready() -> void:
	add_theme_constant_override("separation", 6)
	_build_ui()
	ZADOSClient.map_data_received.connect(_on_map_data)
	# Load knowledge map list on open
	ZADOSClient.get_map("knowledge_maps")


func _build_ui() -> void:
	# Header
	var hdr := Label.new()
	hdr.text = "Map (Cognitools)"
	hdr.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	add_child(hdr)

	# Source selector
	var src_lbl := Label.new()
	src_lbl.text = "Data source"
	src_lbl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	src_lbl.add_theme_font_size_override("font_size", 10)
	add_child(src_lbl)

	_source_btn = OptionButton.new()
	_source_btn.add_item("AtomSpace (E9)")
	_source_btn.focus_mode = Control.FOCUS_NONE
	_source_btn.item_selected.connect(_on_source_selected)
	add_child(_source_btn)

	# Buttons row
	var btn_row := HBoxContainer.new()
	btn_row.add_theme_constant_override("separation", 6)
	add_child(btn_row)
	var load_btn := Button.new()
	load_btn.text = "↺ Load"
	load_btn.flat = true
	load_btn.focus_mode = Control.FOCUS_NONE
	load_btn.add_theme_font_size_override("font_size", 11)
	load_btn.pressed.connect(_emit_current_source)
	btn_row.add_child(load_btn)

	add_child(HSeparator.new())

	# Stats
	_stats_lbl = Label.new()
	_stats_lbl.text = "—"
	_stats_lbl.add_theme_color_override("font_color", Color(0.50, 0.52, 0.55))
	_stats_lbl.add_theme_font_size_override("font_size", 10)
	add_child(_stats_lbl)

	add_child(HSeparator.new())

	# Selected node section
	var sel_hdr := Label.new()
	sel_hdr.text = "Selected Node"
	sel_hdr.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	sel_hdr.add_theme_font_size_override("font_size", 11)
	add_child(sel_hdr)

	_node_lbl = Label.new()
	_node_lbl.text = "—"
	_node_lbl.add_theme_color_override("font_color", Color(0.85, 0.88, 0.92))
	_node_lbl.add_theme_font_size_override("font_size", 13)
	_node_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(_node_lbl)

	_type_lbl = Label.new()
	_type_lbl.text = ""
	_type_lbl.add_theme_color_override("font_color", Color(0.40, 0.70, 1.00))
	_type_lbl.add_theme_font_size_override("font_size", 10)
	add_child(_type_lbl)

	var bar_grid := GridContainer.new()
	bar_grid.columns = 2
	bar_grid.add_theme_constant_override("h_separation", 8)
	bar_grid.add_theme_constant_override("v_separation", 4)
	add_child(bar_grid)

	_str_bar  = _labeled_bar(bar_grid, "Strength")
	_conf_bar = _labeled_bar(bar_grid, "Confidence")
	_sti_bar  = _labeled_bar(bar_grid, "STI")

	add_child(HSeparator.new())

	var meta_hdr := Label.new()
	meta_hdr.text = "Metadata"
	meta_hdr.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	meta_hdr.add_theme_font_size_override("font_size", 10)
	add_child(meta_hdr)

	_meta_grid = GridContainer.new()
	_meta_grid.columns = 2
	_meta_grid.add_theme_constant_override("h_separation", 8)
	_meta_grid.add_theme_constant_override("v_separation", 2)
	add_child(_meta_grid)


func _labeled_bar(parent: GridContainer, title: String) -> ProgressBar:
	var lbl := Label.new()
	lbl.text = title + ":"
	lbl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	lbl.add_theme_font_size_override("font_size", 10)
	parent.add_child(lbl)
	var bar := ProgressBar.new()
	bar.min_value = 0.0
	bar.max_value = 1.0
	bar.value     = 0.0
	bar.size_flags_horizontal = SIZE_EXPAND_FILL
	bar.custom_minimum_size   = Vector2(0, 13)
	parent.add_child(bar)
	return bar


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

func show_node(node_id: String, node_data: Dictionary) -> void:
	_node_lbl.text  = str(node_data.get("label", node_id))
	_type_lbl.text  = str(node_data.get("type",  "—"))
	_str_bar.value  = float(node_data.get("strength",   0.0))
	_conf_bar.value = float(node_data.get("confidence", 0.0))
	_sti_bar.value  = clampf(float(node_data.get("sti", 0.0)), 0.0, 1.0)

	for child in _meta_grid.get_children():
		child.queue_free()
	var meta : Dictionary = node_data.get("metadata", {})
	for k in meta:
		var kl := Label.new()
		kl.text = str(k) + ":"
		kl.add_theme_color_override("font_color", Color(0.40, 0.40, 0.45))
		kl.add_theme_font_size_override("font_size", 10)
		kl.custom_minimum_size = Vector2(60, 0)
		_meta_grid.add_child(kl)
		var vl := Label.new()
		vl.text = str(meta[k]).left(80)
		vl.add_theme_color_override("font_color", Color(0.70, 0.73, 0.77))
		vl.add_theme_font_size_override("font_size", 10)
		vl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		_meta_grid.add_child(vl)


func update_stats(node_count: int, edge_count: int) -> void:
	_stats_lbl.text = "%d nodes  ·  %d edges" % [node_count, edge_count]


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------

func _on_map_data(key: String, data: Dictionary) -> void:
	if key != "knowledge_maps":
		return
	# Populate dropdown with knowledge maps
	for item in (data.get("items", []) as Array):
		var map_id : String = str(item.get("map_id", ""))
		var title  : String = str(item.get("title",  map_id.left(20)))
		if map_id.is_empty() or _map_options.has(map_id):
			continue
		var idx : int = _source_btn.get_item_count()
		_source_btn.add_item("KM: " + title.left(28))
		_map_options[map_id] = idx


func _on_source_selected(idx: int) -> void:
	_emit_for_index(idx)


func _emit_current_source() -> void:
	_emit_for_index(_source_btn.get_selected())


func _emit_for_index(idx: int) -> void:
	if idx == 0:
		source_changed.emit("atomspace")
		return
	# Find the map_id whose option index matches
	for map_id in _map_options:
		if _map_options[map_id] == idx:
			source_changed.emit("knowledge_maps/" + str(map_id) + "/graph")
			return

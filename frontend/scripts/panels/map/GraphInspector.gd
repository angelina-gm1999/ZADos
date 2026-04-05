##
## GraphInspector — Source selector, graph stats, selected-node details,
##   editable values, add link, delete atom, save/load.
##
## Addendum B.5.3: editable TruthValue/AttentionValue with PUT on change.
## Addendum B.5.4: Add Link dialog, Delete Atom confirmation, save/load flow.
##
extends VBoxContainer

signal source_changed(key: String)

const _ConfirmDialog = preload("res://scripts/components/ConfirmationDialog.gd")
const _Toast = preload("res://scripts/components/Toast.gd")
const _ErrorDisplay = preload("res://scripts/components/ErrorDisplay.gd")

var _source_btn  : OptionButton
var _stats_lbl   : Label
var _cap_lbl     : Label
var _node_lbl    : Label
var _type_lbl    : Label
var _str_slider  : HSlider
var _conf_slider : HSlider
var _sti_slider  : HSlider
var _str_val_lbl : Label
var _conf_val_lbl: Label
var _sti_val_lbl : Label
var _meta_grid   : GridContainer

# Current selection
var _selected_id   : String = ""
var _selected_data : Dictionary = {}

# map_id → option index (for knowledge maps added dynamically)
var _map_options : Dictionary = {}

func _ready() -> void:
	add_theme_constant_override("separation", 6)
	_build_ui()
	ZADOSClient.map_data_received.connect(_on_map_data)
	ZADOSClient.memory_post_result.connect(_on_post_result)
	ZADOSClient.request_failed.connect(_on_request_failed)
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

	# Save/Load buttons (B.5.4)
	var save_btn := Button.new()
	save_btn.text = "Save"
	save_btn.flat = true
	save_btn.focus_mode = Control.FOCUS_NONE
	save_btn.add_theme_font_size_override("font_size", 11)
	save_btn.add_theme_color_override("font_color", Color(0.40, 0.70, 0.40))
	save_btn.pressed.connect(_on_save_pressed)
	btn_row.add_child(save_btn)

	add_child(HSeparator.new())

	# Stats + cap warning
	_stats_lbl = Label.new()
	_stats_lbl.text = "—"
	_stats_lbl.add_theme_color_override("font_color", Color(0.50, 0.52, 0.55))
	_stats_lbl.add_theme_font_size_override("font_size", 10)
	add_child(_stats_lbl)

	_cap_lbl = Label.new()
	_cap_lbl.text = ""
	_cap_lbl.add_theme_color_override("font_color", Color(0.95, 0.80, 0.25))
	_cap_lbl.add_theme_font_size_override("font_size", 9)
	_cap_lbl.visible = false
	add_child(_cap_lbl)

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

	# Editable sliders (B.5.3)
	var slider_grid := GridContainer.new()
	slider_grid.columns = 3
	slider_grid.add_theme_constant_override("h_separation", 6)
	slider_grid.add_theme_constant_override("v_separation", 4)
	add_child(slider_grid)

	_str_slider  = _labeled_slider(slider_grid, "Strength")
	_str_val_lbl = slider_grid.get_child(slider_grid.get_child_count() - 1) as Label
	_conf_slider = _labeled_slider(slider_grid, "Confidence")
	_conf_val_lbl = slider_grid.get_child(slider_grid.get_child_count() - 1) as Label
	_sti_slider  = _labeled_slider(slider_grid, "STI")
	_sti_val_lbl = slider_grid.get_child(slider_grid.get_child_count() - 1) as Label

	_str_slider.value_changed.connect(func(v): _str_val_lbl.text = "%.2f" % v)
	_conf_slider.value_changed.connect(func(v): _conf_val_lbl.text = "%.2f" % v)
	_sti_slider.value_changed.connect(func(v): _sti_val_lbl.text = "%.2f" % v)

	# Apply button for value edits
	var apply_btn := Button.new()
	apply_btn.text = "Apply Changes"
	apply_btn.focus_mode = Control.FOCUS_NONE
	apply_btn.add_theme_font_size_override("font_size", 10)
	apply_btn.pressed.connect(_apply_value_changes)
	add_child(apply_btn)

	add_child(HSeparator.new())

	# Action buttons (B.5.4)
	var action_row := HBoxContainer.new()
	action_row.add_theme_constant_override("separation", 6)
	add_child(action_row)

	var add_link_btn := Button.new()
	add_link_btn.text = "+ Add Link"
	add_link_btn.flat = true
	add_link_btn.focus_mode = Control.FOCUS_NONE
	add_link_btn.add_theme_font_size_override("font_size", 10)
	add_link_btn.add_theme_color_override("font_color", Color(0.40, 0.70, 0.90))
	add_link_btn.pressed.connect(_show_add_link_dialog)
	action_row.add_child(add_link_btn)

	var delete_btn := Button.new()
	delete_btn.text = "Delete Atom"
	delete_btn.flat = true
	delete_btn.focus_mode = Control.FOCUS_NONE
	delete_btn.add_theme_font_size_override("font_size", 10)
	delete_btn.add_theme_color_override("font_color", Color(0.90, 0.35, 0.30))
	delete_btn.pressed.connect(_show_delete_confirm)
	action_row.add_child(delete_btn)

	add_child(HSeparator.new())

	# Metadata
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


func _labeled_slider(parent: GridContainer, title: String) -> HSlider:
	var lbl := Label.new()
	lbl.text = title + ":"
	lbl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	lbl.add_theme_font_size_override("font_size", 10)
	parent.add_child(lbl)
	var slider := HSlider.new()
	slider.min_value = 0.0
	slider.max_value = 1.0
	slider.step = 0.01
	slider.value = 0.0
	slider.size_flags_horizontal = SIZE_EXPAND_FILL
	slider.custom_minimum_size = Vector2(0, 16)
	parent.add_child(slider)
	var val_lbl := Label.new()
	val_lbl.text = "0.00"
	val_lbl.add_theme_color_override("font_color", Color(0.55, 0.58, 0.62))
	val_lbl.add_theme_font_size_override("font_size", 10)
	val_lbl.custom_minimum_size = Vector2(34, 0)
	parent.add_child(val_lbl)
	return slider


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

func show_node(node_id: String, node_data: Dictionary) -> void:
	_selected_id   = node_id
	_selected_data = node_data
	_node_lbl.text  = str(node_data.get("label", node_id))
	_type_lbl.text  = str(node_data.get("type",  "—"))

	var str_val  := float(node_data.get("strength",   0.0))
	var conf_val := float(node_data.get("confidence", 0.0))
	var sti_val  := clampf(float(node_data.get("sti", 0.0)), 0.0, 1.0)

	_str_slider.set_value_no_signal(str_val)
	_conf_slider.set_value_no_signal(conf_val)
	_sti_slider.set_value_no_signal(sti_val)
	_str_val_lbl.text = "%.2f" % str_val
	_conf_val_lbl.text = "%.2f" % conf_val
	_sti_val_lbl.text = "%.2f" % sti_val

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


func show_cap_warning(total: int, displayed: int) -> void:
	if total > displayed:
		_cap_lbl.text = "Showing %d / %d (capped at %d)" % [displayed, total, displayed]
		_cap_lbl.visible = true
	else:
		_cap_lbl.visible = false


# ---------------------------------------------------------------------------
# Value editing (B.5.3)
# ---------------------------------------------------------------------------

func _apply_value_changes() -> void:
	if _selected_id.is_empty():
		_show_toast("No node selected.", _Toast.Level.WARNING)
		return
	ZADOSClient.post_map("atoms/" + _selected_id, {
		"strength":   _str_slider.value,
		"confidence": _conf_slider.value,
		"sti":        _sti_slider.value,
	})
	_show_toast("Values updated for %s" % _node_lbl.text.left(20), _Toast.Level.SUCCESS)


# ---------------------------------------------------------------------------
# Add Link dialog (B.5.4)
# ---------------------------------------------------------------------------

func _show_add_link_dialog() -> void:
	if _selected_id.is_empty():
		_show_toast("Select a source node first.", _Toast.Level.WARNING)
		return
	var dlg := _ConfirmDialog.new()
	dlg.show_dialog_with_input(
		"Add Link from '%s'" % _node_lbl.text.left(30),
		"Enter target node ID and link type (comma-separated).\nFormat: target_id, link_type",
		"Add Link", "Cancel",
		"target_node_id, INHERITANCE_LINK", 5)
	var main := get_tree().get_root().get_node_or_null("Main/ModalContainer")
	if main:
		main.add_child(dlg)
	else:
		add_child(dlg)
	dlg.result_with_text.connect(func(confirmed: bool, text: String):
		if not confirmed:
			return
		var parts := text.split(",")
		if parts.size() < 2:
			_show_toast("Format: target_id, link_type", _Toast.Level.WARNING)
			return
		var target_id := parts[0].strip_edges()
		var link_type := parts[1].strip_edges()
		ZADOSClient.post_map("links", {
			"source": _selected_id,
			"target": target_id,
			"type": link_type,
			"weight": 0.5,
		})
		_show_toast("Link added: %s → %s" % [_selected_id.left(8), target_id.left(8)], _Toast.Level.SUCCESS)
	)


# ---------------------------------------------------------------------------
# Delete Atom confirmation (B.5.4)
# ---------------------------------------------------------------------------

func _show_delete_confirm() -> void:
	if _selected_id.is_empty():
		_show_toast("No node selected.", _Toast.Level.WARNING)
		return
	var dlg := _ConfirmDialog.new()
	dlg.show_dialog(
		"Delete atom '%s'?" % _node_lbl.text.left(30),
		"This will remove the atom and all connected links. This cannot be undone.",
		"Delete", "Cancel")
	var main := get_tree().get_root().get_node_or_null("Main/ModalContainer")
	if main:
		main.add_child(dlg)
	else:
		add_child(dlg)
	dlg.result.connect(func(confirmed: bool):
		if confirmed:
			ZADOSClient.post_map("atoms/" + _selected_id + "/delete", {})
			_show_toast("Atom deleted: %s" % _selected_id.left(12), _Toast.Level.SUCCESS)
			_selected_id = ""
			_node_lbl.text = "—"
			_type_lbl.text = ""
			_emit_current_source()  # Reload graph
	)


# ---------------------------------------------------------------------------
# Save/Load (B.5.4)
# ---------------------------------------------------------------------------

func _on_save_pressed() -> void:
	var dlg := _ConfirmDialog.new()
	dlg.show_dialog(
		"Save graph state?",
		"This will persist the current AtomSpace/knowledge map state to disk.",
		"Save", "Cancel")
	var main := get_tree().get_root().get_node_or_null("Main/ModalContainer")
	if main:
		main.add_child(dlg)
	else:
		add_child(dlg)
	dlg.result.connect(func(confirmed: bool):
		if confirmed:
			ZADOSClient.post_map("save", {})
			_show_toast("Graph state saved", _Toast.Level.SUCCESS)
	)


func _on_post_result(key: String, _data: Dictionary) -> void:
	if "atoms/" in key or "links" in key or "save" in key:
		# Already toasted in each action handler
		pass


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------

func _on_request_failed(path: String, error: Dictionary) -> void:
	if "/map/" not in path:
		return
	_show_toast("Map data error: HTTP %s" % str(error.get("http_code", "?")),
		_Toast.Level.ERROR)


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


func _show_toast(text: String, level: int) -> void:
	var tc = get_tree().get_root().get_node_or_null("Main/ToastContainer")
	if tc:
		tc.show_toast(text, level)

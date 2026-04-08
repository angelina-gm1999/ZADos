##
## HardcodedPanel — LTMM Identity hardcoded defaults viewer + editor.
##
## Addendum B.4.3: session-only vs persistent scope selector,
##   confirmation on disk write, change log with per-field revert.
##
extends VBoxContainer

const _ConfirmDialog = preload("res://scripts/components/ConfirmationDialog.gd")
const _Toast = preload("res://scripts/components/Toast.gd")
const _ErrorDisplay = preload("res://scripts/components/ErrorDisplay.gd")

var _list          : VBoxContainer
var _scope_btn     : OptionButton
var _change_log    : VBoxContainer
var _changes       : Dictionary = {}   # entry_id → {field → {old, new}}

func _ready() -> void:
	add_theme_constant_override("separation", 4)
	add_child(_make_header())

	# Scope selector (B.4.3)
	var scope_row := HBoxContainer.new()
	scope_row.add_theme_constant_override("separation", 8)
	var scope_lbl := Label.new()
	scope_lbl.text = "Edit scope:"
	scope_lbl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	scope_lbl.add_theme_font_size_override("font_size", 10)
	scope_row.add_child(scope_lbl)
	_scope_btn = OptionButton.new()
	_scope_btn.add_item("Session Only")
	_scope_btn.add_item("Persist to Disk")
	_scope_btn.focus_mode = Control.FOCUS_NONE
	_scope_btn.add_theme_font_size_override("font_size", 10)
	scope_row.add_child(_scope_btn)
	add_child(scope_row)

	var notice := Label.new()
	notice.text = "Session edits are lost on restart. Persistent edits write to the hardcoded defaults file."
	notice.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	notice.add_theme_font_size_override("font_size", 10)
	notice.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(notice)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical    = SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	add_child(scroll)

	var inner := VBoxContainer.new()
	inner.size_flags_horizontal = SIZE_EXPAND_FILL
	inner.add_theme_constant_override("separation", 6)
	scroll.add_child(inner)

	_list = VBoxContainer.new()
	_list.size_flags_horizontal = SIZE_EXPAND_FILL
	_list.add_theme_constant_override("separation", 6)
	inner.add_child(_list)

	# Change log section
	var cl_header := Label.new()
	cl_header.text = "Change Log"
	cl_header.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	cl_header.add_theme_font_size_override("font_size", 11)
	inner.add_child(cl_header)
	_change_log = VBoxContainer.new()
	_change_log.add_theme_constant_override("separation", 4)
	inner.add_child(_change_log)
	var cl_empty := Label.new()
	cl_empty.text = "No changes yet."
	cl_empty.add_theme_color_override("font_color", Color(0.35, 0.35, 0.40))
	cl_empty.add_theme_font_size_override("font_size", 10)
	_change_log.add_child(cl_empty)

	ZADOSClient.memory_data_received.connect(_on_data)
	ZADOSClient.memory_post_result.connect(_on_post_result)
	ZADOSClient.request_failed.connect(_on_request_failed)


func refresh() -> void:
	ZADOSClient.get_memory("ltmm/identity/hardcoded")


func _on_data(key: String, data: Dictionary) -> void:
	if key != "ltmm/identity/hardcoded":
		return
	_populate(data)


func _on_request_failed(path: String, error: Dictionary) -> void:
	if "/memory/ltmm/identity/hardcoded" not in path:
		return
	for child in _list.get_children():
		child.queue_free()
	var err := _ErrorDisplay.new()
	err.show_error("Hardcoded Defaults", "HTTP %s — %s" % [
		str(error.get("http_code", "?")),
		str(error.get("body", "Connection failed")).left(120)])
	err.retry_pressed.connect(refresh)
	_list.add_child(err)


func _on_post_result(key: String, _data: Dictionary) -> void:
	if "ltmm/identity/hardcoded" in key:
		_show_toast("Hardcoded entry updated", _Toast.Level.SUCCESS)
		refresh()


func _populate(d: Dictionary) -> void:
	for child in _list.get_children():
		child.queue_free()
	var items : Array = d.get("items", [])
	if items.is_empty():
		var lbl := Label.new()
		lbl.text = "No hardcoded entries found."
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		_list.add_child(lbl)
		return
	for entry in items:
		_list.add_child(_make_card(entry))


func _make_card(e: Dictionary) -> PanelContainer:
	var entry_id : String = e.get("entry_id", str(e.get("category", "")))
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.09, 0.09, 0.12)
	style.border_width_left = 2
	style.border_color      = Color(0.35, 0.55, 0.75)
	style.corner_radius_top_left     = 4; style.corner_radius_top_right    = 4
	style.corner_radius_bottom_left  = 4; style.corner_radius_bottom_right = 4
	style.content_margin_left = 10; style.content_margin_right  = 10
	style.content_margin_top  = 8;  style.content_margin_bottom = 8
	panel.add_theme_stylebox_override("panel", style)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 4)
	panel.add_child(vbox)

	# Category badge + tags
	var hdr := HBoxContainer.new()
	hdr.add_theme_constant_override("separation", 6)
	vbox.add_child(hdr)
	var cat := Label.new()
	cat.text = str(e.get("category", "—")).to_upper()
	cat.add_theme_color_override("font_color", Color(0.35, 0.65, 0.95))
	cat.add_theme_font_size_override("font_size", 10)
	hdr.add_child(cat)
	var tags_str : String = " / ".join(PackedStringArray((e.get("tags", []) as Array)))
	if not tags_str.is_empty():
		var tags_lbl := Label.new()
		tags_lbl.text = tags_str
		tags_lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.45))
		tags_lbl.add_theme_font_size_override("font_size", 10)
		hdr.add_child(tags_lbl)

	# Editable content
	var content_text : String = e.get("content", "")
	var content_edit := TextEdit.new()
	content_edit.text = content_text
	content_edit.custom_minimum_size = Vector2(0, 60)
	content_edit.size_flags_horizontal = SIZE_EXPAND_FILL
	content_edit.add_theme_font_size_override("font_size", 11)
	content_edit.add_theme_color_override("font_color", Color(0.85, 0.88, 0.92))
	vbox.add_child(content_edit)

	# Save button
	var btn_row := HBoxContainer.new()
	btn_row.add_theme_constant_override("separation", 8)
	vbox.add_child(btn_row)
	var save_btn := Button.new()
	save_btn.text = "Save"
	save_btn.focus_mode = Control.FOCUS_NONE
	save_btn.add_theme_font_size_override("font_size", 11)
	save_btn.pressed.connect(func():
		var new_text := content_edit.text.strip_edges()
		if new_text == content_text:
			_show_toast("No changes to save.", _Toast.Level.INFO)
			return
		_save_entry(entry_id, e.get("category",""), content_text, new_text)
	)
	btn_row.add_child(save_btn)
	var revert_btn := Button.new()
	revert_btn.text = "Revert"
	revert_btn.flat = true
	revert_btn.focus_mode = Control.FOCUS_NONE
	revert_btn.add_theme_font_size_override("font_size", 11)
	revert_btn.pressed.connect(func(): content_edit.text = content_text)
	btn_row.add_child(revert_btn)

	return panel


## Save with optional confirmation for disk writes (B.4.3)
func _save_entry(entry_id: String, category: String, old_text: String, new_text: String) -> void:
	var persist := _scope_btn.get_selected() == 1
	if persist:
		# Confirmation dialog for persistent writes
		var dlg := _ConfirmDialog.new()
		dlg.show_dialog(
			"Write to disk?",
			"This will permanently update the hardcoded defaults file for '%s'.\nThis change persists across restarts." % category,
			"Write to Disk", "Cancel")
		var main := get_tree().get_root().get_node_or_null("Main/ModalContainer")
		if main:
			main.add_child(dlg)
		else:
			add_child(dlg)
		dlg.result.connect(func(confirmed: bool):
			if confirmed:
				_do_save(entry_id, old_text, new_text, true)
		)
	else:
		_do_save(entry_id, old_text, new_text, false)


func _do_save(entry_id: String, old_text: String, new_text: String, persist: bool) -> void:
	# Record in change log
	_changes[entry_id] = {"old": old_text, "new": new_text, "persist": persist}
	_update_change_log()

	ZADOSClient.post_memory("ltmm/identity/hardcoded/" + entry_id, {
		"content": new_text,
		"persist": persist,
	})


func _update_change_log() -> void:
	for child in _change_log.get_children():
		child.queue_free()
	if _changes.is_empty():
		var lbl := Label.new()
		lbl.text = "No changes yet."
		lbl.add_theme_color_override("font_color", Color(0.35, 0.35, 0.40))
		lbl.add_theme_font_size_override("font_size", 10)
		_change_log.add_child(lbl)
		return
	for eid in _changes:
		var change : Dictionary = _changes[eid]
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 8)
		var id_lbl := Label.new()
		id_lbl.text = eid.left(16)
		id_lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.80))
		id_lbl.add_theme_font_size_override("font_size", 10)
		id_lbl.custom_minimum_size = Vector2(100, 0)
		row.add_child(id_lbl)
		var scope_lbl := Label.new()
		scope_lbl.text = "DISK" if change.get("persist", false) else "SESSION"
		scope_lbl.add_theme_color_override("font_color",
			Color(0.90, 0.60, 0.20) if change.get("persist", false) else Color(0.45, 0.55, 0.45))
		scope_lbl.add_theme_font_size_override("font_size", 9)
		row.add_child(scope_lbl)
		var revert_btn := Button.new()
		revert_btn.text = "Revert"
		revert_btn.flat = true
		revert_btn.focus_mode = Control.FOCUS_NONE
		revert_btn.add_theme_font_size_override("font_size", 10)
		var old_val : String = change.get("old", "")
		var ceid : String = eid
		revert_btn.pressed.connect(func():
			ZADOSClient.post_memory("ltmm/identity/hardcoded/" + ceid, {
				"content": old_val, "persist": change.get("persist", false)})
			_changes.erase(ceid)
			_update_change_log()
			_show_toast("Reverted %s" % ceid.left(12), _Toast.Level.INFO)
		)
		row.add_child(revert_btn)
		_change_log.add_child(row)


func _show_toast(text: String, level: int) -> void:
	var tc = get_tree().get_root().get_node_or_null("Main/ToastContainer")
	if tc:
		tc.show_toast(text, level)


func _make_header() -> HBoxContainer:
	var hbox := HBoxContainer.new()
	var lbl  := Label.new()
	lbl.text = "Hardcoded Defaults"
	lbl.size_flags_horizontal = SIZE_EXPAND_FILL
	lbl.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	hbox.add_child(lbl)
	var btn := Button.new()
	btn.text = "↺ Refresh"
	btn.flat = true
	btn.focus_mode = Control.FOCUS_NONE
	btn.add_theme_font_size_override("font_size", 11)
	btn.pressed.connect(refresh)
	hbox.add_child(btn)
	return hbox

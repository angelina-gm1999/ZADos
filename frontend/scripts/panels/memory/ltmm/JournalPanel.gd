##
## JournalPanel — LTMM Journal entries list.
##
## Addendum B.2.8: trigger journal entry with confirmation.
## Addendum C: pagination support.
##
extends VBoxContainer

const _ConfirmDialog = preload("res://scripts/components/ConfirmationDialog.gd")
const _Toast = preload("res://scripts/components/Toast.gd")
const _ErrorDisplay = preload("res://scripts/components/ErrorDisplay.gd")

var _list : VBoxContainer
var _filter_btn : OptionButton
var _active_filter : String = ""
var _trigger_btn : Button
var _load_more_btn : Button
var _count_label : Label
var _offset : int = 0
var _total  : int = 0
const PAGE_SIZE := 20

func _ready() -> void:
	add_theme_constant_override("separation", 4)
	add_child(_make_header())
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

	# Count label + load more
	_count_label = Label.new()
	_count_label.text = ""
	_count_label.add_theme_font_size_override("font_size", 10)
	_count_label.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	_count_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	inner.add_child(_count_label)

	_load_more_btn = Button.new()
	_load_more_btn.text = "Load more..."
	_load_more_btn.flat = true
	_load_more_btn.focus_mode = Control.FOCUS_NONE
	_load_more_btn.add_theme_font_size_override("font_size", 11)
	_load_more_btn.add_theme_color_override("font_color", Color(0.5, 0.65, 0.85))
	_load_more_btn.visible = false
	_load_more_btn.pressed.connect(_load_more)
	inner.add_child(_load_more_btn)

	ZADOSClient.memory_data_received.connect(_on_data)
	ZADOSClient.memory_post_result.connect(_on_post_result)
	ZADOSClient.request_failed.connect(_on_request_failed)


func refresh() -> void:
	_offset = 0
	for child in _list.get_children():
		child.queue_free()
	ZADOSClient.get_memory("ltmm/journal?offset=0&limit=%d" % PAGE_SIZE)


func _load_more() -> void:
	_offset += PAGE_SIZE
	ZADOSClient.get_memory("ltmm/journal?offset=%d&limit=%d" % [_offset, PAGE_SIZE])


func _on_data(key: String, data: Dictionary) -> void:
	if not key.begins_with("ltmm/journal"):
		return
	if key.begins_with("ltmm/journal?"):
		_populate(data, _offset > 0)
		return
	if key == "ltmm/journal":
		_populate(data, false)


func _on_request_failed(path: String, error: Dictionary) -> void:
	if "/memory/ltmm/journal" not in path:
		return
	for child in _list.get_children():
		child.queue_free()
	var err := _ErrorDisplay.new()
	err.show_error("Journal", "HTTP %s — %s" % [
		str(error.get("http_code", "?")),
		str(error.get("body", "Connection failed")).left(120)])
	err.retry_pressed.connect(refresh)
	_list.add_child(err)


func _on_post_result(key: String, _data: Dictionary) -> void:
	if "ltmm/journal/trigger" in key:
		_show_toast("Journal entry written", _Toast.Level.SUCCESS)
		refresh()


func _populate(d: Dictionary, append: bool) -> void:
	if not append:
		for child in _list.get_children():
			child.queue_free()

	_total = d.get("total", 0) as int
	var items : Array = d.get("items", [])

	if items.is_empty() and not append:
		var lbl := Label.new()
		lbl.text = "No journal entries yet."
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		_list.add_child(lbl)
		_count_label.text = ""
		_load_more_btn.visible = false
		return

	for entry in items:
		var trigger_str : String = str(entry.get("trigger", ""))
		if not _active_filter.is_empty() and trigger_str != _active_filter:
			continue
		_list.add_child(_make_card(entry))

	var loaded := _list.get_child_count()
	_count_label.text = "Showing %d / %d entries" % [loaded, _total]
	_load_more_btn.visible = loaded < _total


func _make_card(e: Dictionary) -> PanelContainer:
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.10, 0.10, 0.13)
	style.corner_radius_top_left = 5; style.corner_radius_top_right = 5
	style.corner_radius_bottom_left = 5; style.corner_radius_bottom_right = 5
	style.content_margin_left = 10; style.content_margin_right = 10
	style.content_margin_top = 8; style.content_margin_bottom = 8
	panel.add_theme_stylebox_override("panel", style)
	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 4)
	panel.add_child(vbox)

	var hdr := HBoxContainer.new()
	var trigger_lbl := Label.new()
	trigger_lbl.text = str(e.get("trigger", ""))
	trigger_lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.80))
	trigger_lbl.add_theme_font_size_override("font_size", 11)
	hdr.add_child(trigger_lbl)
	var spacer := Control.new(); spacer.size_flags_horizontal = SIZE_EXPAND
	hdr.add_child(spacer)
	var ts := Label.new()
	ts.text = (str(e.get("timestamp", ""))).left(19)
	ts.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
	ts.add_theme_font_size_override("font_size", 10)
	hdr.add_child(ts)
	vbox.add_child(hdr)

	var tags : Array = e.get("tags", [])
	if not tags.is_empty():
		var tag_lbl := Label.new()
		tag_lbl.text = "  ".join(PackedStringArray(tags))
		tag_lbl.add_theme_color_override("font_color", Color(0.45, 0.65, 0.45))
		tag_lbl.add_theme_font_size_override("font_size", 10)
		tag_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		vbox.add_child(tag_lbl)

	var prose : String = e.get("prose", "")
	if not prose.is_empty():
		var plbl := Label.new()
		plbl.text = prose.left(300)
		plbl.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
		plbl.add_theme_font_size_override("font_size", 11)
		plbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		vbox.add_child(plbl)

	var prompts : Array = e.get("reflection_prompts", [])
	if not prompts.is_empty():
		var plbl2 := Label.new()
		plbl2.text = "? " + "\n? ".join(PackedStringArray(prompts.slice(0, 3)))
		plbl2.add_theme_color_override("font_color", Color(0.60, 0.62, 0.50))
		plbl2.add_theme_font_size_override("font_size", 10)
		plbl2.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		vbox.add_child(plbl2)

	return panel


## Trigger journal entry with confirmation (B.2.8)
func _trigger_journal() -> void:
	var dlg := _ConfirmDialog.new()
	dlg.show_dialog(
		"Write a journal entry for the current session state?",
		"This will invoke the JournalTool to write a reflective entry.",
		"Write Entry", "Cancel")
	var main := get_tree().get_root().get_node_or_null("Main/ModalContainer")
	if main:
		main.add_child(dlg)
	else:
		add_child(dlg)
	dlg.result.connect(func(confirmed: bool):
		if confirmed:
			_trigger_btn.text = "Writing..."
			_trigger_btn.disabled = true
			ZADOSClient.post_memory("ltmm/journal/trigger", {
				"trigger_source": "dev_interface",
				"notes": [],
			})
			# Re-enable after a brief delay
			await get_tree().create_timer(3.0).timeout
			_trigger_btn.text = "Trigger Entry"
			_trigger_btn.disabled = false
	)


func _on_filter_changed(idx: int) -> void:
	_active_filter = _filter_btn.get_item_text(idx) if idx > 0 else ""
	refresh()


func _show_toast(text: String, level: int) -> void:
	var tc = get_tree().get_root().get_node_or_null("Main/ToastContainer")
	if tc:
		tc.show_toast(text, level)


func _make_header() -> HBoxContainer:
	var hbox := HBoxContainer.new()
	var lbl  := Label.new()
	lbl.text = "Journal"
	lbl.size_flags_horizontal = SIZE_EXPAND_FILL
	lbl.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	hbox.add_child(lbl)
	_filter_btn = OptionButton.new()
	_filter_btn.add_item("All Triggers")
	_filter_btn.add_item("periodic")
	_filter_btn.add_item("ltmm_threshold")
	_filter_btn.add_item("rem_complete")
	_filter_btn.add_item("dream_scene_shift")
	_filter_btn.add_item("innovation_flag")
	_filter_btn.add_item("dev")
	_filter_btn.add_theme_font_size_override("font_size", 10)
	_filter_btn.item_selected.connect(_on_filter_changed)
	hbox.add_child(_filter_btn)
	_trigger_btn = Button.new()
	_trigger_btn.text = "Trigger Entry"
	_trigger_btn.flat = false
	_trigger_btn.focus_mode = Control.FOCUS_NONE
	_trigger_btn.add_theme_font_size_override("font_size", 11)
	_trigger_btn.pressed.connect(_trigger_journal)
	hbox.add_child(_trigger_btn)
	var btn := Button.new()
	btn.text = "↺ Refresh"
	btn.flat = true
	btn.focus_mode = Control.FOCUS_NONE
	btn.add_theme_font_size_override("font_size", 11)
	btn.pressed.connect(refresh)
	hbox.add_child(btn)
	return hbox

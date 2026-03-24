##
## JournalPanel — LTMM Journal entries list.
##
extends VBoxContainer

var _list : VBoxContainer
var _filter_btn : OptionButton
var _active_filter : String = ""

func _ready() -> void:
	add_theme_constant_override("separation", 4)
	add_child(_make_header())
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical    = SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	add_child(scroll)
	_list = VBoxContainer.new()
	_list.size_flags_horizontal = SIZE_EXPAND_FILL
	_list.add_theme_constant_override("separation", 6)
	scroll.add_child(_list)
	ZADOSClient.memory_data_received.connect(_on_data)


func refresh() -> void:
	ZADOSClient.get_memory("ltmm/journal")


func _on_data(key: String, data: Dictionary) -> void:
	if key == "ltmm/journal/trigger/result":
		refresh()
		return
	if key != "ltmm/journal":
		return
	_populate(data)


func _populate(d: Dictionary) -> void:
	for child in _list.get_children():
		child.queue_free()
	var items : Array = d.get("items", [])
	if items.is_empty():
		var lbl := Label.new()
		lbl.text = "No journal entries yet."
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		_list.add_child(lbl)
		return
	for entry in items:
		var trigger_str : String = str(entry.get("trigger", ""))
		if not _active_filter.is_empty() and trigger_str != _active_filter:
			continue
		_list.add_child(_make_card(entry))


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


func _trigger_journal() -> void:
	ZADOSClient.post_memory("ltmm/journal/trigger", {
		"trigger_source": "dev_interface",
		"notes": [],
	})


func _on_filter_changed(idx: int) -> void:
	_active_filter = _filter_btn.get_item_text(idx) if idx > 0 else ""
	refresh()


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
	_filter_btn.add_item("innovation_flag")
	_filter_btn.add_item("dev")
	_filter_btn.add_theme_font_size_override("font_size", 10)
	_filter_btn.item_selected.connect(_on_filter_changed)
	hbox.add_child(_filter_btn)
	var trigger_btn := Button.new()
	trigger_btn.text = "Trigger Entry"
	trigger_btn.flat = false
	trigger_btn.focus_mode = Control.FOCUS_NONE
	trigger_btn.add_theme_font_size_override("font_size", 11)
	trigger_btn.pressed.connect(_trigger_journal)
	hbox.add_child(trigger_btn)
	var btn := Button.new()
	btn.text = "↺ Refresh"
	btn.flat = true
	btn.focus_mode = Control.FOCUS_NONE
	btn.add_theme_font_size_override("font_size", 11)
	btn.pressed.connect(refresh)
	hbox.add_child(btn)
	return hbox

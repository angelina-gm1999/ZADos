##
## HardcodedPanel — Read-only viewer for LTMM Identity hardcoded defaults.
##
extends VBoxContainer

var _list : VBoxContainer

func _ready() -> void:
	add_theme_constant_override("separation", 4)
	add_child(_make_header())
	var notice := Label.new()
	notice.text = "These entries are baked into the system at build time and cannot be edited at runtime."
	notice.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	notice.add_theme_font_size_override("font_size", 10)
	notice.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	add_child(notice)
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
	ZADOSClient.get_memory("ltmm/identity/hardcoded")


func _on_data(key: String, data: Dictionary) -> void:
	if key != "ltmm/identity/hardcoded":
		return
	_populate(data)


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

	# Content
	var content_lbl := Label.new()
	content_lbl.text = (e.get("content", "") as String).left(500)
	content_lbl.add_theme_color_override("font_color", Color(0.85, 0.88, 0.92))
	content_lbl.add_theme_font_size_override("font_size", 11)
	content_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	vbox.add_child(content_lbl)

	return panel


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

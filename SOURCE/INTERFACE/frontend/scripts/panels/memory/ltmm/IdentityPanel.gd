##
## IdentityPanel — LTMM Identity: Core Memories | Hardcoded | Development | Alignment.
##
extends VBoxContainer

var _tabs           : TabContainer
var _core_list      : VBoxContainer
var _hardcoded_list : VBoxContainer
var _dev_list       : VBoxContainer
var _alignment_list : VBoxContainer

func _ready() -> void:
	add_theme_constant_override("separation", 0)
	_build_ui()
	ZADOSClient.memory_data_received.connect(_on_data)


func _build_ui() -> void:
	_tabs = TabContainer.new()
	_tabs.size_flags_vertical = SIZE_EXPAND_FILL
	_tabs.add_theme_constant_override("side_margin", 0)
	add_child(_tabs)

	_core_list      = _make_tab("Core Memories", func(): ZADOSClient.get_memory("ltmm/identity/core"))
	_hardcoded_list = _make_tab("Hardcoded", func(): ZADOSClient.get_memory("ltmm/identity/hardcoded"))
	_dev_list       = _make_tab("Development", func(): ZADOSClient.get_memory("ltmm/identity/development"))
	_alignment_list = _make_tab("Alignment", func(): ZADOSClient.get_memory("ltmm/identity/alignment"))


func _make_tab(tab_name: String, refresh_fn: Callable) -> VBoxContainer:
	var tab_wrap := VBoxContainer.new()
	tab_wrap.name = tab_name
	tab_wrap.add_theme_constant_override("separation", 4)
	var hdr := HBoxContainer.new()
	var lbl := Label.new()
	lbl.text = tab_name
	lbl.size_flags_horizontal = SIZE_EXPAND_FILL
	lbl.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	hdr.add_child(lbl)
	var btn := Button.new()
	btn.text = "↺ Refresh"
	btn.flat = true
	btn.focus_mode = Control.FOCUS_NONE
	btn.add_theme_font_size_override("font_size", 11)
	btn.pressed.connect(refresh_fn)
	hdr.add_child(btn)
	tab_wrap.add_child(hdr)
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical    = SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	tab_wrap.add_child(scroll)
	var list := VBoxContainer.new()
	list.size_flags_horizontal = SIZE_EXPAND_FILL
	list.add_theme_constant_override("separation", 6)
	scroll.add_child(list)
	_tabs.add_child(tab_wrap)
	return list


func refresh() -> void:
	ZADOSClient.get_memory("ltmm/identity/core")
	ZADOSClient.get_memory("ltmm/identity/hardcoded")
	ZADOSClient.get_memory("ltmm/identity/development")
	ZADOSClient.get_memory("ltmm/identity/alignment")


func _on_data(key: String, data: Dictionary) -> void:
	match key:
		"ltmm/identity/core":
			_populate_items(_core_list, data, _make_core_card)
		"ltmm/identity/hardcoded":
			_populate_items(_hardcoded_list, data, _make_hardcoded_card)
		"ltmm/identity/development":
			_populate_development(data)
		"ltmm/identity/alignment":
			_populate_alignment(data)


func _populate_items(list: VBoxContainer, d: Dictionary, fn: Callable) -> void:
	for child in list.get_children():
		child.queue_free()
	var items : Array = d.get("items", [])
	if items.is_empty():
		var lbl := Label.new()
		lbl.text = "Empty."
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		list.add_child(lbl)
		return
	for item in items:
		list.add_child(fn.call(item))


func _populate_development(d: Dictionary) -> void:
	for child in _dev_list.get_children():
		child.queue_free()
	# Conclusions section
	var conclusions : Array = d.get("conclusions", [])
	if not conclusions.is_empty():
		_dev_list.add_child(_section_lbl("Conclusions"))
		for c in conclusions:
			_dev_list.add_child(_card(
				c.get("content", ""),
				{
					"Type":          c.get("conclusion_type", "—"),
					"Confidence":    "%.2f" % float(c.get("confidence", 0.0)),
					"Reinforced":    str(c.get("reinforcement_count", 0)) + "×",
				},
				(c.get("created_at", "") as String).left(19)
			))
	# Identity journal section
	var journal_entries : Array = d.get("journal_entries", [])
	if not journal_entries.is_empty():
		_dev_list.add_child(_section_lbl("Identity Journal"))
		for e in journal_entries:
			_dev_list.add_child(_card(
				e.get("content", ""),
				{
					"Type":     str(e.get("entry_type", "—")),
					"Pipeline": e.get("source_pipeline", "—"),
					"Tags":     " / ".join(PackedStringArray((e.get("tags", []) as Array))),
				},
				(e.get("timestamp", "") as String).left(19)
			))
	if conclusions.is_empty() and journal_entries.is_empty():
		var lbl := Label.new()
		lbl.text = "No development data yet."
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		_dev_list.add_child(lbl)


func _populate_alignment(d: Dictionary) -> void:
	for child in _alignment_list.get_children():
		child.queue_free()
	var any_data := false
	for section_key in ["axiom_notes", "value_notes", "constraint_notes", "personality_prompts", "flags"]:
		var items : Array = d.get(section_key, [])
		if items.is_empty():
			continue
		any_data = true
		_alignment_list.add_child(_section_lbl(section_key.replace("_", " ").capitalize()))
		for item in items:
			var lbl := Label.new()
			lbl.text = str(item)
			lbl.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
			lbl.add_theme_font_size_override("font_size", 11)
			lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
			_alignment_list.add_child(lbl)
	if not any_data:
		var lbl := Label.new()
		lbl.text = "No alignment notes for current context."
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		_alignment_list.add_child(lbl)


func _section_lbl(title: String) -> Label:
	var lbl := Label.new()
	lbl.text = title
	lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.80))
	lbl.add_theme_font_size_override("font_size", 12)
	return lbl


func _make_core_card(m: Dictionary) -> PanelContainer:
	return _card(
		m.get("content", ""),
		{
			"Type":    m.get("memory_type", "—"),
			"Version": str(m.get("version", 1)),
			"Tags":    " / ".join(PackedStringArray((m.get("tags", []) as Array))),
		},
		(m.get("created_at", "") as String).left(19)
	)


func _make_hardcoded_card(e: Dictionary) -> PanelContainer:
	return _card(
		e.get("content", ""),
		{
			"Category": e.get("category", "—"),
			"Tags":     " / ".join(PackedStringArray((e.get("tags", []) as Array))),
		}, ""
	)


func _card(title: String, fields: Dictionary, timestamp: String) -> PanelContainer:
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

	if not timestamp.is_empty():
		var ts := Label.new()
		ts.text = timestamp
		ts.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		ts.add_theme_font_size_override("font_size", 10)
		vbox.add_child(ts)

	var tl := Label.new()
	tl.text = (title as String).left(400)
	tl.add_theme_color_override("font_color", Color(0.85, 0.88, 0.92))
	tl.add_theme_font_size_override("font_size", 11)
	tl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	vbox.add_child(tl)

	var grid := GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 12)
	grid.add_theme_constant_override("v_separation", 2)
	for k in fields:
		var kl := Label.new()
		kl.text = str(k) + ":"
		kl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		kl.add_theme_font_size_override("font_size", 10)
		kl.custom_minimum_size = Vector2(70, 0)
		grid.add_child(kl)
		var vl := Label.new()
		vl.text = str(fields[k])
		vl.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
		vl.add_theme_font_size_override("font_size", 10)
		vl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		grid.add_child(vl)
	vbox.add_child(grid)
	return panel

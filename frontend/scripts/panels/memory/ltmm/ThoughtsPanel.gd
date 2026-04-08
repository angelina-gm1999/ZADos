##
## ThoughtsPanel — LTMM Thoughts: Held Blocks | Overview Logs | General Questions
##
extends VBoxContainer

const _ErrorDisplay = preload("res://scripts/components/ErrorDisplay.gd")

var _tabs        : TabContainer
var _blocks_list : VBoxContainer
var _logs_list   : VBoxContainer
var _questions_list : VBoxContainer

func _ready() -> void:
	add_theme_constant_override("separation", 0)
	_build_ui()
	ZADOSClient.memory_data_received.connect(_on_data)
	ZADOSClient.request_failed.connect(_on_request_failed)


func _build_ui() -> void:
	_tabs = TabContainer.new()
	_tabs.size_flags_vertical = SIZE_EXPAND_FILL
	_tabs.add_theme_constant_override("side_margin", 0)
	add_child(_tabs)

	_blocks_list    = _make_tab_scroll("Held Blocks")
	_logs_list      = _make_tab_scroll("Overview Logs")
	_questions_list = _make_tab_scroll("General Questions")


func _make_tab_scroll(tab_name: String) -> VBoxContainer:
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
	btn.pressed.connect(func(): _refresh_tab(tab_name))
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


func _refresh_tab(tab_name: String) -> void:
	match tab_name:
		"Held Blocks":      ZADOSClient.get_memory("ltmm/thoughts/held_blocks")
		"Overview Logs":    ZADOSClient.get_memory("ltmm/thoughts/overview_logs")
		"General Questions":ZADOSClient.get_memory("ltmm/thoughts/general_questions")


func refresh() -> void:
	ZADOSClient.get_memory("ltmm/thoughts/held_blocks")
	ZADOSClient.get_memory("ltmm/thoughts/overview_logs")
	ZADOSClient.get_memory("ltmm/thoughts/general_questions")


func _on_data(key: String, data: Dictionary) -> void:
	match key:
		"ltmm/thoughts/held_blocks":
			_populate_list(_blocks_list, data, _make_block_card)
		"ltmm/thoughts/overview_logs":
			_populate_list(_logs_list, data, _make_log_card)
		"ltmm/thoughts/general_questions":
			_populate_list(_questions_list, data, _make_question_card)


func _on_request_failed(path: String, error: Dictionary) -> void:
	if "/memory/ltmm/thoughts" not in path:
		return
	var msg := "HTTP %s — %s" % [
		str(error.get("http_code", "?")),
		str(error.get("body", "Connection failed")).left(120)]
	var target_list : VBoxContainer = null
	var section_name := "Thoughts"
	if "held_blocks" in path:
		target_list = _blocks_list; section_name = "Held Blocks"
	elif "overview_logs" in path:
		target_list = _logs_list; section_name = "Overview Logs"
	elif "general_questions" in path:
		target_list = _questions_list; section_name = "General Questions"
	if target_list:
		for child in target_list.get_children():
			child.queue_free()
		var err := _ErrorDisplay.new()
		err.show_error(section_name, msg)
		err.retry_pressed.connect(refresh)
		target_list.add_child(err)


func _populate_list(list: VBoxContainer, d: Dictionary, card_fn: Callable) -> void:
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
		list.add_child(card_fn.call(item))


func _make_block_card(b: Dictionary) -> PanelContainer:
	var fields := {
		"Emotion":   b.get("emotion_tag",    "—"),
		"Phase":     b.get("pipeline_phase", "—"),
		"Context":   (b.get("context_summary","") as String).left(120),
		"Reviewed":  str(b.get("reviewed",   false)),
	}
	return _card_with_title(
		(b.get("thought_fragment", "") as String).left(200),
		fields,
		(b.get("timestamp","") as String).left(19)
	)


func _make_log_card(l: Dictionary) -> PanelContainer:
	var modes  := " → ".join(PackedStringArray((l.get("mode_sequence",[]) as Array).slice(0,5)))
	var fields := {
		"Modes":    modes if not modes.is_empty() else "—",
		"Subjects": " / ".join(PackedStringArray((l.get("subject_tags",[]) as Array).slice(0,4))),
		"Emotions": " / ".join(PackedStringArray((l.get("dominant_emotions",[]) as Array).slice(0,3))),
	}
	return _card_with_title(
		(l.get("summary","") as String).left(300),
		fields,
		(l.get("timestamp","") as String).left(19)
	)


func _make_question_card(q: Dictionary) -> PanelContainer:
	var fields := {
		"Source":     q.get("source",          "—"),
		"Priority":   "%.2f" % float(q.get("priority", 0.0)),
		"Stagnation": str(q.get("stagnation_count", 0)) + " cycles",
		"Resolved":   str(q.get("resolved", false)),
	}
	return _card_with_title(q.get("formulation","—"), fields, "")


func _card_with_title(title: String, fields: Dictionary, timestamp: String) -> PanelContainer:
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

	if not title.is_empty():
		var tl := Label.new()
		tl.text = title
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
		kl.custom_minimum_size = Vector2(80, 0)
		grid.add_child(kl)
		var vl := Label.new()
		vl.text = str(fields[k])
		vl.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
		vl.add_theme_font_size_override("font_size", 10)
		vl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		grid.add_child(vl)
	vbox.add_child(grid)
	return panel

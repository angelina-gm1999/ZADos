##
## UnsolvedPanel — LTMM Unsolved question queue with resolve action.
##
extends VBoxContainer

var _list : VBoxContainer

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
	ZADOSClient.get_memory("ltmm/unsolved")


func _on_data(key: String, data: Dictionary) -> void:
	if key != "ltmm/unsolved" and key != "ltmm/unsolved/result":
		return
	if key == "ltmm/unsolved/result":
		# After resolving, refresh the list
		refresh()
		return
	_populate(data)


func _populate(d: Dictionary) -> void:
	for child in _list.get_children():
		child.queue_free()
	var items : Array = d.get("items", [])
	if items.is_empty():
		var lbl := Label.new()
		lbl.text = "No unsolved questions. 🎉"
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		_list.add_child(lbl)
		return

	# Sort by stagnation_cycles descending (most stagnated first)
	items.sort_custom(func(a, b):
		return (a.get("stagnation_cycles",0) as int) > (b.get("stagnation_cycles",0) as int))

	for entry in items:
		_list.add_child(_make_card(entry))


func _make_card(e: Dictionary) -> PanelContainer:
	var is_dream : bool = e.get("dream_candidate", false)
	var resolved  : bool = e.get("resolved", false)

	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.10, 0.10, 0.13)
	if is_dream and not resolved:
		style.border_width_left  = 2
		style.border_color       = Color(0.65, 0.35, 0.85)
	style.corner_radius_top_left = 5; style.corner_radius_top_right = 5
	style.corner_radius_bottom_left = 5; style.corner_radius_bottom_right = 5
	style.content_margin_left = 10; style.content_margin_right = 10
	style.content_margin_top = 8; style.content_margin_bottom = 8
	panel.add_theme_stylebox_override("panel", style)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 5)
	panel.add_child(vbox)

	# Header row
	var hdr := HBoxContainer.new()
	if is_dream:
		var badge := Label.new()
		badge.text = "⚑ Dream"
		badge.add_theme_color_override("font_color", Color(0.65, 0.35, 0.85))
		badge.add_theme_font_size_override("font_size", 10)
		hdr.add_child(badge)
	var spacer := Control.new(); spacer.size_flags_horizontal = SIZE_EXPAND
	hdr.add_child(spacer)
	var ts := Label.new()
	ts.text = (str(e.get("timestamp",""))).left(19)
	ts.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
	ts.add_theme_font_size_override("font_size", 10)
	hdr.add_child(ts)
	vbox.add_child(hdr)

	# Question text
	var ql := Label.new()
	ql.text = (e.get("concept_formulation","") as String).left(400)
	ql.add_theme_color_override("font_color", Color(0.90, 0.92, 0.95))
	ql.add_theme_font_size_override("font_size", 11)
	ql.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	vbox.add_child(ql)

	# Meta fields
	var grid := GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 12)
	grid.add_theme_constant_override("v_separation", 2)
	var meta := {
		"Source":     e.get("source_engine",  "—"),
		"Stagnation": str(e.get("stagnation_cycles",0)) + " cycles",
		"Attempts":   str((e.get("resolution_attempts",[]) as Array).size()),
	}
	if not (e.get("blocking_reason","") as String).is_empty():
		meta["Blocking"] = (e.get("blocking_reason","") as String).left(80)
	for k in meta:
		var kl := Label.new()
		kl.text = str(k) + ":"
		kl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		kl.add_theme_font_size_override("font_size", 10)
		kl.custom_minimum_size = Vector2(70, 0)
		grid.add_child(kl)
		var vl := Label.new()
		vl.text = str(meta[k])
		vl.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
		vl.add_theme_font_size_override("font_size", 10)
		vl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		grid.add_child(vl)
	vbox.add_child(grid)

	# Resolve button (only if unresolved)
	if not resolved:
		var footer := HBoxContainer.new()
		footer.add_theme_constant_override("separation", 8)
		vbox.add_child(footer)
		var resolve_btn := Button.new()
		resolve_btn.text = "Mark Resolved"
		resolve_btn.flat = false
		resolve_btn.focus_mode = Control.FOCUS_NONE
		resolve_btn.add_theme_font_size_override("font_size", 11)
		var entry_id : String = e.get("entry_id","")
		resolve_btn.pressed.connect(func():
			ZADOSClient.post_memory("ltmm/unsolved/" + entry_id + "/resolve", {}))
		footer.add_child(resolve_btn)
		var convo_btn := Button.new()
		convo_btn.text = "Send to Chat"
		convo_btn.flat = true
		convo_btn.focus_mode = Control.FOCUS_NONE
		convo_btn.add_theme_font_size_override("font_size", 11)
		var question_text : String = e.get("concept_formulation","")
		convo_btn.pressed.connect(func(): _send_to_conversation(question_text))
		footer.add_child(convo_btn)
		var selfref_btn := Button.new()
		selfref_btn.text = "Self-Reflective"
		selfref_btn.flat = true
		selfref_btn.focus_mode = Control.FOCUS_NONE
		selfref_btn.add_theme_font_size_override("font_size", 11)
		selfref_btn.add_theme_color_override("font_color", Color(0.65, 0.35, 0.85))
		selfref_btn.pressed.connect(func(): _send_to_self_reflective(question_text))
		footer.add_child(selfref_btn)

	return panel


func _send_to_conversation(question: String) -> void:
	ZADOSClient.prefill_text = question
	var main := get_tree().get_root().get_node_or_null("Main")
	if main and main.has_method("_switch_to"):
		main._switch_to("conversation")


func _send_to_self_reflective(question: String) -> void:
	ZADOSClient.set_session_mode("SelfReflective")
	ZADOSClient.prefill_text = question
	var main := get_tree().get_root().get_node_or_null("Main")
	if main and main.has_method("_switch_to"):
		main._switch_to("conversation")


func _make_header() -> HBoxContainer:
	var hbox := HBoxContainer.new()
	var lbl  := Label.new()
	lbl.text = "Unsolved Questions"
	lbl.size_flags_horizontal = SIZE_EXPAND_FILL
	lbl.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	hbox.add_child(lbl)
	var count_lbl := Label.new()
	count_lbl.add_theme_color_override("font_color", Color(0.5, 0.5, 0.5))
	count_lbl.add_theme_font_size_override("font_size", 11)
	hbox.add_child(count_lbl)
	var spacer := Control.new(); spacer.size_flags_horizontal = SIZE_EXPAND
	hbox.add_child(spacer)
	var btn := Button.new()
	btn.text = "↺ Refresh"
	btn.flat = true
	btn.focus_mode = Control.FOCUS_NONE
	btn.add_theme_font_size_override("font_size", 11)
	btn.pressed.connect(refresh)
	hbox.add_child(btn)
	return hbox

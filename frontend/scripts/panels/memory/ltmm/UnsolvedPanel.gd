##
## UnsolvedPanel — LTMM Unsolved question queue with resolve action.
##
## Addendum B.2.5: resolve modal with required note.
## Addendum B.2.6: send to conversation with prefill.
## Addendum B.2.7: send to self-reflective with confirmation.
##
extends VBoxContainer

const _ConfirmDialog = preload("res://scripts/components/ConfirmationDialog.gd")
const _Toast = preload("res://scripts/components/Toast.gd")
const _ErrorDisplay = preload("res://scripts/components/ErrorDisplay.gd")

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
	ZADOSClient.memory_post_result.connect(_on_post_result)
	ZADOSClient.request_failed.connect(_on_request_failed)


func refresh() -> void:
	ZADOSClient.get_memory("ltmm/unsolved")


func _on_data(key: String, data: Dictionary) -> void:
	if key != "ltmm/unsolved":
		return
	_populate(data)


func _on_request_failed(path: String, error: Dictionary) -> void:
	if "/memory/ltmm/unsolved" not in path:
		return
	for child in _list.get_children():
		child.queue_free()
	var err := _ErrorDisplay.new()
	err.show_error("Unsolved Questions", "HTTP %s — %s" % [
		str(error.get("http_code", "?")),
		str(error.get("body", "Connection failed")).left(120)])
	err.retry_pressed.connect(refresh)
	_list.add_child(err)


func _on_post_result(key: String, _data: Dictionary) -> void:
	if "ltmm/unsolved" in key:
		_show_toast("Question resolved", _Toast.Level.SUCCESS)
		refresh()


func _populate(d: Dictionary) -> void:
	for child in _list.get_children():
		child.queue_free()
	var items : Array = d.get("items", [])
	if items.is_empty():
		var lbl := Label.new()
		lbl.text = "No unsolved questions."
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		_list.add_child(lbl)
		return

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
	var origin_tag : String = e.get("origin", "")
	if not origin_tag.is_empty():
		var origin_lbl := Label.new()
		origin_lbl.text = "  origin:%s" % origin_tag
		origin_lbl.add_theme_color_override("font_color", Color(0.5, 0.5, 0.55))
		origin_lbl.add_theme_font_size_override("font_size", 10)
		hdr.add_child(origin_lbl)
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
		vl.size_flags_horizontal = SIZE_EXPAND_FILL
		grid.add_child(vl)
	vbox.add_child(grid)

	# Partial answers (collapsible)
	var attempts : Array = e.get("resolution_attempts", [])
	if not attempts.is_empty():
		var toggle_btn := Button.new()
		toggle_btn.text = "▶ Partial answers (%d)" % attempts.size()
		toggle_btn.flat = true
		toggle_btn.focus_mode = Control.FOCUS_NONE
		toggle_btn.add_theme_font_size_override("font_size", 10)
		toggle_btn.add_theme_color_override("font_color", Color(0.45, 0.5, 0.55))
		var answers_box := VBoxContainer.new()
		answers_box.visible = false
		for attempt in attempts:
			var albl := Label.new()
			albl.text = "• " + str(attempt).left(200)
			albl.add_theme_font_size_override("font_size", 10)
			albl.add_theme_color_override("font_color", Color(0.6, 0.62, 0.66))
			albl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
			answers_box.add_child(albl)
		toggle_btn.pressed.connect(func():
			answers_box.visible = not answers_box.visible
			toggle_btn.text = ("▼" if answers_box.visible else "▶") + " Partial answers (%d)" % attempts.size()
		)
		vbox.add_child(toggle_btn)
		vbox.add_child(answers_box)

	# Action buttons (only if unresolved)
	if not resolved:
		var footer := HBoxContainer.new()
		footer.add_theme_constant_override("separation", 8)
		vbox.add_child(footer)

		var entry_id : String = e.get("entry_id","")
		var question_text : String = e.get("concept_formulation","")

		# Mark Resolved — opens modal with required note (B.2.5)
		var resolve_btn := Button.new()
		resolve_btn.text = "Mark Resolved"
		resolve_btn.flat = false
		resolve_btn.focus_mode = Control.FOCUS_NONE
		resolve_btn.add_theme_font_size_override("font_size", 11)
		resolve_btn.pressed.connect(func(): _show_resolve_modal(entry_id))
		footer.add_child(resolve_btn)

		# Send to Conversation (B.2.6)
		var convo_btn := Button.new()
		convo_btn.text = "Send to Chat"
		convo_btn.flat = true
		convo_btn.focus_mode = Control.FOCUS_NONE
		convo_btn.add_theme_font_size_override("font_size", 11)
		convo_btn.pressed.connect(func(): _send_to_conversation(question_text))
		footer.add_child(convo_btn)

		# Send to Self-Reflective (B.2.7)
		var selfref_btn := Button.new()
		selfref_btn.text = "Self-Reflective"
		selfref_btn.flat = true
		selfref_btn.focus_mode = Control.FOCUS_NONE
		selfref_btn.add_theme_font_size_override("font_size", 11)
		selfref_btn.add_theme_color_override("font_color", Color(0.65, 0.35, 0.85))
		selfref_btn.pressed.connect(func(): _show_self_reflective_confirm(question_text))
		footer.add_child(selfref_btn)

	return panel


## Resolve modal with required note (B.2.5)
func _show_resolve_modal(entry_id: String) -> void:
	var dlg := _ConfirmDialog.new()
	dlg.show_dialog_with_input(
		"How was this resolved?",
		"Provide a resolution note describing how this question was addressed.",
		"Resolve", "Cancel",
		"Resolution note...", 10)
	var main := get_tree().get_root().get_node_or_null("Main/ModalContainer")
	if main:
		main.add_child(dlg)
	else:
		add_child(dlg)
	dlg.result_with_text.connect(func(confirmed: bool, text: String):
		if confirmed and text.length() >= 10:
			ZADOSClient.post_memory(
				"ltmm/unsolved/" + entry_id + "/resolve",
				{"note": text})
	)


## Self-Reflective confirmation (B.2.7)
func _show_self_reflective_confirm(question: String) -> void:
	var dlg := _ConfirmDialog.new()
	dlg.show_dialog(
		"Start Self-Reflective Query on this question?",
		question.left(200),
		"Start", "Cancel")
	var main := get_tree().get_root().get_node_or_null("Main/ModalContainer")
	if main:
		main.add_child(dlg)
	else:
		add_child(dlg)
	dlg.result.connect(func(confirmed: bool):
		if confirmed:
			ZADOSClient.set_session_mode("SelfReflective")
			ZADOSClient.prefill_text = question
			var m := get_tree().get_root().get_node_or_null("Main")
			if m and m.has_method("_switch_to"):
				m._switch_to("conversation")
	)


func _send_to_conversation(question: String) -> void:
	ZADOSClient.prefill_text = question
	var main := get_tree().get_root().get_node_or_null("Main")
	if main and main.has_method("_switch_to"):
		main._switch_to("conversation")


func _show_toast(text: String, level: int) -> void:
	var tc = get_tree().get_root().get_node_or_null("Main/ToastContainer")
	if tc:
		tc.show_toast(text, level)


func _make_header() -> HBoxContainer:
	var hbox := HBoxContainer.new()
	var lbl  := Label.new()
	lbl.text = "Unsolved Questions"
	lbl.size_flags_horizontal = SIZE_EXPAND_FILL
	lbl.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	hbox.add_child(lbl)
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

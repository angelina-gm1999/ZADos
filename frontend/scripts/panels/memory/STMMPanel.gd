##
## STMMPanel — Short-Term Memory Bridge view.
## Read-only. Auto-refreshes after each turn, or manually via Refresh button.
##
## Addendum B.2.1: proper empty state display.
##
extends ScrollContainer

const _ErrorDisplay = preload("res://scripts/components/ErrorDisplay.gd")

var _content : VBoxContainer

func _ready() -> void:
	horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	_content = VBoxContainer.new()
	_content.size_flags_horizontal = SIZE_EXPAND_FILL
	_content.add_theme_constant_override("separation", 6)
	add_child(_content)
	_build_skeleton()
	ZADOSClient.memory_data_received.connect(_on_data)
	ZADOSClient.turn_complete.connect(_on_turn_complete)
	ZADOSClient.request_failed.connect(_on_request_failed)


func _build_skeleton() -> void:
	_content.add_child(_make_header())
	# B.2.1 — empty state
	var empty := VBoxContainer.new()
	empty.name = "EmptyState"
	empty.size_flags_horizontal = SIZE_EXPAND_FILL

	var spacer := Control.new()
	spacer.custom_minimum_size = Vector2(0, 40)
	empty.add_child(spacer)

	var icon := Label.new()
	icon.text = "○"
	icon.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	icon.add_theme_color_override("font_color", Color(0.25, 0.25, 0.30))
	icon.add_theme_font_size_override("font_size", 32)
	empty.add_child(icon)

	var title := Label.new()
	title.text = "No STMM data yet"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	title.add_theme_font_size_override("font_size", 13)
	empty.add_child(title)

	var hint := Label.new()
	hint.text = "Send a message in the Conversation tab to populate the Short-Term Memory Bridge.\nData auto-updates after each turn."
	hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	hint.add_theme_color_override("font_color", Color(0.35, 0.35, 0.40))
	hint.add_theme_font_size_override("font_size", 11)
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	empty.add_child(hint)

	_content.add_child(empty)


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

func refresh() -> void:
	ZADOSClient.get_memory("stmm")


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

func _on_turn_complete(_r: Dictionary) -> void:
	refresh()


func _on_data(key: String, data: Dictionary) -> void:
	if key != "stmm":
		return
	_populate(data)


func _on_request_failed(path: String, error: Dictionary) -> void:
	if "/memory/stmm" not in path:
		return
	for child in _content.get_children():
		child.queue_free()
	_content.add_child(_make_header())
	var err := _ErrorDisplay.new()
	err.show_error("STMM", "HTTP %s — %s" % [
		str(error.get("http_code", "?")),
		str(error.get("body", "Connection failed")).left(120)])
	err.retry_pressed.connect(refresh)
	_content.add_child(err)


# ---------------------------------------------------------------------------
# Populate
# ---------------------------------------------------------------------------

func _populate(d: Dictionary) -> void:
	for child in _content.get_children():
		child.queue_free()
	_content.add_child(_make_header())

	# Check for empty state
	var has_data := d.has("intent_archetype") or d.has("primary_intention") or d.has("latest_user")
	if not has_data:
		_build_skeleton()
		# Remove header duplicate (skeleton adds its own)
		if _content.get_child_count() > 1:
			_content.get_child(0).queue_free()
		return

	_section("Current Bundle")
	_fields({
		"Intent Archetype":     d.get("intent_archetype",    "—"),
		"Primary Intention":    d.get("primary_intention",   "—"),
		"Confidence":           "%.2f" % float(d.get("confidence",     0.0)),
		"Active Mode":          d.get("active_mode",         "—"),
		"Tone Valence":         "%.3f" % float(d.get("tone_valence",    0.0)),
		"Tone Warmth":          "%.3f" % float(d.get("tone_warmth",     0.0)),
		"Tone Coherence":       "%.3f" % float(d.get("tone_coherence",  0.0)),
		"Identity Coherence":   d.get("identity_coherence",  "—"),
		"Messages in Buffer":   str(d.get("message_count",   0)),
	})

	var user_text : String = d.get("latest_user", "")
	if not user_text.is_empty():
		_section("Latest User Input")
		_text_block(user_text)

	var anomalies : Array = d.get("processing_anomalies", [])
	if not anomalies.is_empty():
		_section("Processing Anomalies")
		_fields_list(anomalies)

	var emotions : Dictionary = d.get("user_emotions", {})
	if not emotions.is_empty():
		_section("User Emotion Signals  (top 5)")
		var keys : Array = emotions.keys()
		keys.sort_custom(func(a, b): return (emotions[a] as float) > (emotions[b] as float))
		var top5 := {}
		for i in range(mini(5, keys.size())):
			top5[keys[i]] = "%.3f" % float(emotions[keys[i]])
		_fields(top5)

	var stages : Dictionary = d.get("stage_flags", {})
	if not stages.is_empty():
		_section("Brain Process Tracker")
		_stage_badges(stages)

	var vr : String = d.get("verbal_reflection", "")
	if not vr.is_empty():
		_section("Cortical Reflection  (excerpt)")
		_text_block(vr)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

func _make_header() -> HBoxContainer:
	var hbox := HBoxContainer.new()
	var lbl  := Label.new()
	lbl.text = "Short-Term Memory Bridge"
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


func _section(title: String) -> void:
	var sep := HSeparator.new()
	sep.add_theme_color_override("color", Color(0.2, 0.2, 0.25))
	_content.add_child(sep)
	var lbl := Label.new()
	lbl.text = title
	lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.80))
	lbl.add_theme_font_size_override("font_size", 11)
	_content.add_child(lbl)


func _fields(data: Dictionary) -> void:
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.10, 0.10, 0.13)
	style.corner_radius_top_left = 4; style.corner_radius_top_right = 4
	style.corner_radius_bottom_left = 4; style.corner_radius_bottom_right = 4
	style.content_margin_left = 10; style.content_margin_right = 10
	style.content_margin_top = 6; style.content_margin_bottom = 6
	panel.add_theme_stylebox_override("panel", style)
	var grid := GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 12)
	grid.add_theme_constant_override("v_separation", 3)
	panel.add_child(grid)
	for k in data:
		var key_lbl := Label.new()
		key_lbl.text = str(k) + ":"
		key_lbl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		key_lbl.add_theme_font_size_override("font_size", 11)
		key_lbl.custom_minimum_size = Vector2(140, 0)
		grid.add_child(key_lbl)
		var val_lbl := Label.new()
		val_lbl.text = str(data[k])
		val_lbl.add_theme_color_override("font_color", Color(0.85, 0.88, 0.92))
		val_lbl.add_theme_font_size_override("font_size", 11)
		val_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		grid.add_child(val_lbl)
	_content.add_child(panel)


func _fields_list(items: Array) -> void:
	var data := {}
	for i in range(items.size()):
		data[str(i + 1)] = str(items[i])
	_fields(data)


func _text_block(text: String) -> void:
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.09, 0.09, 0.12)
	style.corner_radius_top_left = 4; style.corner_radius_top_right = 4
	style.corner_radius_bottom_left = 4; style.corner_radius_bottom_right = 4
	style.content_margin_left = 10; style.content_margin_right = 10
	style.content_margin_top = 6; style.content_margin_bottom = 6
	panel.add_theme_stylebox_override("panel", style)
	var lbl := Label.new()
	lbl.text = text
	lbl.add_theme_color_override("font_color", Color(0.70, 0.72, 0.75))
	lbl.add_theme_font_size_override("font_size", 11)
	lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	panel.add_child(lbl)
	_content.add_child(panel)


func _stage_badges(stages: Dictionary) -> void:
	var hflow := HFlowContainer.new()
	hflow.add_theme_constant_override("h_separation", 4)
	hflow.add_theme_constant_override("v_separation", 4)
	for stage in stages:
		var done : bool = stages[stage]
		var lbl  := Label.new()
		lbl.text = stage
		lbl.add_theme_font_size_override("font_size", 10)
		var clr := Color(0.25, 0.80, 0.45) if done else Color(0.35, 0.35, 0.40)
		lbl.add_theme_color_override("font_color", clr)
		var bg := StyleBoxFlat.new()
		bg.bg_color = Color(clr.r * 0.15, clr.g * 0.15, clr.b * 0.15)
		bg.corner_radius_top_left = 3; bg.corner_radius_top_right = 3
		bg.corner_radius_bottom_left = 3; bg.corner_radius_bottom_right = 3
		bg.content_margin_left = 6; bg.content_margin_right = 6
		bg.content_margin_top = 2; bg.content_margin_bottom = 2
		var pad := PanelContainer.new()
		pad.add_theme_stylebox_override("panel", bg)
		pad.add_child(lbl)
		hflow.add_child(pad)
	_content.add_child(hflow)

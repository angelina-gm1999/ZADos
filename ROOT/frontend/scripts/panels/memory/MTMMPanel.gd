##
## MTMMPanel — Mid-Term Memory view.
## Left: context summary + trends.  Right: scrollable packet list.
##
extends HSplitContainer

var _context_content : VBoxContainer
var _packets_list    : VBoxContainer
var _packet_count_lbl: Label

func _ready() -> void:
	split_offset = 320
	_build_ui()
	ZADOSClient.memory_data_received.connect(_on_data)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

func _build_ui() -> void:
	# ── Left pane — context summary ─────────────────────────────────────────
	var left_scroll := ScrollContainer.new()
	left_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	left_scroll.custom_minimum_size     = Vector2(300, 0)
	add_child(left_scroll)

	_context_content = VBoxContainer.new()
	_context_content.size_flags_horizontal = SIZE_EXPAND_FILL
	_context_content.add_theme_constant_override("separation", 6)
	left_scroll.add_child(_context_content)

	var lhdr := _make_panel_header("Context Summary", func(): ZADOSClient.get_memory("mtmm/context"))
	_context_content.add_child(lhdr)

	_packet_count_lbl = Label.new()
	_packet_count_lbl.text = "Packets: —"
	_packet_count_lbl.add_theme_color_override("font_color", Color(0.5, 0.5, 0.5))
	_packet_count_lbl.add_theme_font_size_override("font_size", 11)
	_context_content.add_child(_packet_count_lbl)

	# ── Right pane — packet list ─────────────────────────────────────────────
	var right_wrap := VBoxContainer.new()
	right_wrap.size_flags_horizontal = SIZE_EXPAND_FILL
	add_child(right_wrap)

	var rhdr := _make_panel_header("Interaction Log", func(): ZADOSClient.get_memory("mtmm/packets"))
	right_wrap.add_child(rhdr)

	var right_scroll := ScrollContainer.new()
	right_scroll.size_flags_vertical     = SIZE_EXPAND_FILL
	right_scroll.horizontal_scroll_mode  = ScrollContainer.SCROLL_MODE_DISABLED
	right_wrap.add_child(right_scroll)

	_packets_list = VBoxContainer.new()
	_packets_list.size_flags_horizontal = SIZE_EXPAND_FILL
	_packets_list.add_theme_constant_override("separation", 4)
	right_scroll.add_child(_packets_list)


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

func refresh() -> void:
	ZADOSClient.get_memory("mtmm/context")
	ZADOSClient.get_memory("mtmm/packets")


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

func _on_data(key: String, data: Dictionary) -> void:
	match key:
		"mtmm/context":
			_populate_context(data)
		"mtmm/packets":
			_populate_packets(data)


# ---------------------------------------------------------------------------
# Populate — context
# ---------------------------------------------------------------------------

func _populate_context(d: Dictionary) -> void:
	for child in _context_content.get_children():
		child.queue_free()
	_context_content.add_child(_make_panel_header("Context Summary",
		func(): ZADOSClient.get_memory("mtmm/context")))

	_packet_count_lbl = Label.new()
	_packet_count_lbl.text = "Packets stored: %d" % d.get("packet_count", 0)
	_packet_count_lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.80))
	_packet_count_lbl.add_theme_font_size_override("font_size", 12)
	_context_content.add_child(_packet_count_lbl)

	var intentions : Array = d.get("recent_intentions", [])
	if not intentions.is_empty():
		_context_content.add_child(_section_lbl("Recent Intentions"))
		for i in range(intentions.size()):
			var lbl := Label.new()
			lbl.text = "  %d. %s" % [intentions.size() - i, intentions[intentions.size() - 1 - i]]
			lbl.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
			lbl.add_theme_font_size_override("font_size", 11)
			lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
			_context_content.add_child(lbl)

	var trends : Dictionary = d.get("trends", {})
	if not trends.is_empty():
		_context_content.add_child(_section_lbl("Trends"))
		var trend_fields := {}
		for key in ["contradiction_trend", "emotional_trend"]:
			if key in trends:
				trend_fields[key.replace("_", " ").capitalize()] = str(trends[key])
		var reward_trend : Dictionary = trends.get("reward_trend", {})
		for domain in reward_trend:
			trend_fields["Reward / " + str(domain)] = str(reward_trend[domain])
		_context_content.add_child(_make_fields_card(trend_fields))


# ---------------------------------------------------------------------------
# Populate — packets
# ---------------------------------------------------------------------------

func _populate_packets(d: Dictionary) -> void:
	for child in _packets_list.get_children():
		child.queue_free()

	var items : Array = d.get("items", [])
	if items.is_empty():
		var lbl := Label.new()
		lbl.text = "No packets yet."
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		_packets_list.add_child(lbl)
		return

	# Newest first
	for i in range(items.size() - 1, -1, -1):
		var p : Dictionary = items[i]
		_packets_list.add_child(_make_packet_card(p))


func _make_packet_card(p: Dictionary) -> PanelContainer:
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

	# Header row: turn index + emotion labels
	var hdr := HBoxContainer.new()
	var turn_lbl := Label.new()
	turn_lbl.text = "Turn %d" % p.get("turn_index", 0)
	turn_lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.80))
	turn_lbl.add_theme_font_size_override("font_size", 11)
	hdr.add_child(turn_lbl)
	var spacer := Control.new()
	spacer.size_flags_horizontal = SIZE_EXPAND
	hdr.add_child(spacer)
	var emo_labels : Array = p.get("verbal_emotion_labels", [])
	if not emo_labels.is_empty():
		var emo_lbl := Label.new()
		emo_lbl.text = ", ".join(PackedStringArray(emo_labels.slice(0, 3)))
		emo_lbl.add_theme_color_override("font_color", Color(0.55, 0.75, 0.55))
		emo_lbl.add_theme_font_size_override("font_size", 10)
		hdr.add_child(emo_lbl)
	vbox.add_child(hdr)

	# Intent
	var intent : String = p.get("intention", "")
	if not intent.is_empty():
		var ilbl := Label.new()
		ilbl.text = intent
		ilbl.add_theme_color_override("font_color", Color(0.80, 0.82, 0.86))
		ilbl.add_theme_font_size_override("font_size", 11)
		ilbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		vbox.add_child(ilbl)

	# Trust / significance bars
	var metrics_row := HBoxContainer.new()
	metrics_row.add_theme_constant_override("separation", 16)
	_add_mini_bar(metrics_row, "Trust", float(p.get("trust_weight", 0.0)), Color(0.30, 0.75, 0.45))
	_add_mini_bar(metrics_row, "Significance", float(p.get("emotional_significance", 0.0)), Color(0.80, 0.55, 0.25))
	vbox.add_child(metrics_row)

	# User message excerpt
	var user_msg : String = p.get("user_message", "")
	if not user_msg.is_empty():
		var ulbl := Label.new()
		ulbl.text = "U: " + user_msg
		ulbl.add_theme_color_override("font_color", Color(0.60, 0.62, 0.66))
		ulbl.add_theme_font_size_override("font_size", 10)
		ulbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		vbox.add_child(ulbl)

	return panel


func _add_mini_bar(container: HBoxContainer, label: String, value: float, _color: Color) -> void:
	var lbl := Label.new()
	lbl.text = label + ":"
	lbl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	lbl.add_theme_font_size_override("font_size", 10)
	lbl.custom_minimum_size = Vector2(72, 0)
	container.add_child(lbl)
	var bar := ProgressBar.new()
	bar.min_value = 0.0
	bar.max_value = 1.0
	bar.value    = clampf(value, 0.0, 1.0)
	bar.custom_minimum_size = Vector2(80, 10)
	bar.show_percentage = false
	container.add_child(bar)


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------

func _make_panel_header(title: String, refresh_fn: Callable) -> HBoxContainer:
	var hbox := HBoxContainer.new()
	var lbl  := Label.new()
	lbl.text = title
	lbl.size_flags_horizontal = SIZE_EXPAND_FILL
	lbl.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	hbox.add_child(lbl)
	var btn := Button.new()
	btn.text = "↺ Refresh"
	btn.flat = true
	btn.focus_mode = Control.FOCUS_NONE
	btn.add_theme_font_size_override("font_size", 11)
	btn.pressed.connect(refresh_fn)
	hbox.add_child(btn)
	return hbox


func _section_lbl(title: String) -> Label:
	var lbl := Label.new()
	lbl.text = title
	lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.80))
	lbl.add_theme_font_size_override("font_size", 11)
	return lbl


func _make_fields_card(data: Dictionary) -> PanelContainer:
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
		var kl := Label.new()
		kl.text = str(k) + ":"
		kl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		kl.add_theme_font_size_override("font_size", 11)
		kl.custom_minimum_size = Vector2(130, 0)
		grid.add_child(kl)
		var vl := Label.new()
		vl.text = str(data[k])
		vl.add_theme_color_override("font_color", Color(0.85, 0.88, 0.92))
		vl.add_theme_font_size_override("font_size", 11)
		vl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		grid.add_child(vl)
	return panel

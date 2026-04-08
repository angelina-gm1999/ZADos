##
## MTMMPanel — Mid-Term Memory view.
## Left: context summary + trends.  Right: scrollable packet list.
##
## Addendum B.2.2: context editor read-only lock during generation.
## Addendum B.2.3: packet pagination with offset/limit.
##
extends HSplitContainer

const _Toast = preload("res://scripts/components/Toast.gd")
const _ErrorDisplay = preload("res://scripts/components/ErrorDisplay.gd")

var _context_content : VBoxContainer
var _packets_list    : VBoxContainer
var _packet_count_lbl: Label

# Pagination state (B.2.3)
var _offset        : int = 0
var _total_packets : int = 0
const PAGE_SIZE    := 30
var _load_more_btn : Button
var _pkt_count_lbl : Label

# Generation lock state (B.2.2)
var _context_locked : bool = false

func _ready() -> void:
	split_offset = 320
	_build_ui()
	ZADOSClient.memory_data_received.connect(_on_data)
	ZADOSClient.generation_started.connect(func(): _set_context_lock(true))
	ZADOSClient.turn_complete.connect(func(_r): _set_context_lock(false))
	ZADOSClient.generation_cancelled.connect(func(): _set_context_lock(false))
	ZADOSClient.request_failed.connect(_on_request_failed)


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

	# ── Right pane — packet list with pagination ────────────────────────────
	var right_wrap := VBoxContainer.new()
	right_wrap.size_flags_horizontal = SIZE_EXPAND_FILL
	add_child(right_wrap)

	var rhdr := _make_panel_header("Interaction Log", func(): _refresh_packets())
	right_wrap.add_child(rhdr)

	var right_scroll := ScrollContainer.new()
	right_scroll.size_flags_vertical     = SIZE_EXPAND_FILL
	right_scroll.horizontal_scroll_mode  = ScrollContainer.SCROLL_MODE_DISABLED
	right_wrap.add_child(right_scroll)

	var inner := VBoxContainer.new()
	inner.size_flags_horizontal = SIZE_EXPAND_FILL
	inner.add_theme_constant_override("separation", 4)
	right_scroll.add_child(inner)

	_packets_list = VBoxContainer.new()
	_packets_list.size_flags_horizontal = SIZE_EXPAND_FILL
	_packets_list.add_theme_constant_override("separation", 4)
	inner.add_child(_packets_list)

	# Pagination footer
	_pkt_count_lbl = Label.new()
	_pkt_count_lbl.text = ""
	_pkt_count_lbl.add_theme_font_size_override("font_size", 10)
	_pkt_count_lbl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	_pkt_count_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	inner.add_child(_pkt_count_lbl)

	_load_more_btn = Button.new()
	_load_more_btn.text = "Load more..."
	_load_more_btn.flat = true
	_load_more_btn.focus_mode = Control.FOCUS_NONE
	_load_more_btn.add_theme_font_size_override("font_size", 11)
	_load_more_btn.add_theme_color_override("font_color", Color(0.5, 0.65, 0.85))
	_load_more_btn.visible = false
	_load_more_btn.pressed.connect(_load_more_packets)
	inner.add_child(_load_more_btn)


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

func refresh() -> void:
	ZADOSClient.get_memory("mtmm/context")
	_offset = 0
	_refresh_packets()


func _refresh_packets() -> void:
	_offset = 0
	for child in _packets_list.get_children():
		child.queue_free()
	ZADOSClient.get_memory("mtmm/packets?offset=0&limit=%d" % PAGE_SIZE)


func _load_more_packets() -> void:
	_offset += PAGE_SIZE
	ZADOSClient.get_memory("mtmm/packets?offset=%d&limit=%d" % [_offset, PAGE_SIZE])


# ---------------------------------------------------------------------------
# Context lock (B.2.2)
# ---------------------------------------------------------------------------

func _set_context_lock(locked: bool) -> void:
	_context_locked = locked
	# Visual indicator
	if _packet_count_lbl:
		if locked:
			_packet_count_lbl.text = "⏳ Context locked during generation..."
			_packet_count_lbl.add_theme_color_override("font_color", Color(0.85, 0.70, 0.25))
		else:
			_packet_count_lbl.add_theme_color_override("font_color", Color(0.5, 0.5, 0.5))
			refresh()


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

func _on_data(key: String, data: Dictionary) -> void:
	if key == "mtmm/context":
		_populate_context(data)
	elif key.begins_with("mtmm/packets"):
		_populate_packets(data, _offset > 0)


func _on_request_failed(path: String, error: Dictionary) -> void:
	if "/memory/mtmm" not in path:
		return
	var msg := "HTTP %s — %s" % [
		str(error.get("http_code", "?")),
		str(error.get("body", "Connection failed")).left(120)]
	if "context" in path:
		for child in _context_content.get_children():
			child.queue_free()
		_context_content.add_child(_make_panel_header("Context Summary",
			func(): ZADOSClient.get_memory("mtmm/context")))
		var err := _ErrorDisplay.new()
		err.show_error("MTMM Context", msg)
		err.retry_pressed.connect(func(): ZADOSClient.get_memory("mtmm/context"))
		_context_content.add_child(err)
	elif "packets" in path:
		for child in _packets_list.get_children():
			child.queue_free()
		var err := _ErrorDisplay.new()
		err.show_error("MTMM Packets", msg)
		err.retry_pressed.connect(_refresh_packets)
		_packets_list.add_child(err)


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

	# Reapply lock indicator if still locked
	if _context_locked:
		_packet_count_lbl.text += "  ⏳ locked"
		_packet_count_lbl.add_theme_color_override("font_color", Color(0.85, 0.70, 0.25))

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
		for k in ["contradiction_trend", "emotional_trend"]:
			if k in trends:
				trend_fields[k.replace("_", " ").capitalize()] = str(trends[k])
		var reward_trend : Dictionary = trends.get("reward_trend", {})
		for domain in reward_trend:
			trend_fields["Reward / " + str(domain)] = str(reward_trend[domain])
		_context_content.add_child(_make_fields_card(trend_fields))


# ---------------------------------------------------------------------------
# Populate — packets (with pagination support B.2.3)
# ---------------------------------------------------------------------------

func _populate_packets(d: Dictionary, append: bool) -> void:
	if not append:
		for child in _packets_list.get_children():
			child.queue_free()

	_total_packets = d.get("total", 0) as int
	var items : Array = d.get("items", [])

	if items.is_empty() and not append:
		var lbl := Label.new()
		lbl.text = "No packets yet."
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		_packets_list.add_child(lbl)
		_pkt_count_lbl.text = ""
		_load_more_btn.visible = false
		return

	# Newest first
	for i in range(items.size() - 1, -1, -1):
		var p : Dictionary = items[i]
		_packets_list.add_child(_make_packet_card(p))

	var loaded := _packets_list.get_child_count()
	_pkt_count_lbl.text = "Showing %d / %d packets" % [loaded, _total_packets]
	_load_more_btn.visible = loaded < _total_packets


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
		vl.size_flags_horizontal = SIZE_EXPAND_FILL
		grid.add_child(vl)
	return panel

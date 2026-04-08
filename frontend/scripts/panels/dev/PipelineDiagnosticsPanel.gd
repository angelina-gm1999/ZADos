##
## PipelineDiagnosticsPanel — Last-turn pipeline state (engines, phases, modulation).
##
## Addendum B.4.4: search/filter in engine lists, phase timing bar chart.
##
extends VBoxContainer

const _PHASE_NAMES := ["Perception", "Modulation", "Dispatch",
					   "Thinking",   "Reward",     "Answer"]

var _turn_lbl    : Label
var _profile_lbl : Label
var _mode_lbl    : Label
var _engines_run_box  : HFlowContainer
var _engines_skip_box : HFlowContainer
var _phase_grid  : GridContainer
var _weights_grid: GridContainer
var _search_edit : LineEdit
var _timing_box  : HBoxContainer

# Cached for filtering
var _cached_engines_run  : Array = []
var _cached_engines_skip : Array = []
var _cached_phase_timings : Dictionary = {}

func _ready() -> void:
	add_theme_constant_override("separation", 6)
	add_child(_make_header())
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical    = SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	add_child(scroll)
	var inner := VBoxContainer.new()
	inner.size_flags_horizontal = SIZE_EXPAND_FILL
	inner.add_theme_constant_override("separation", 10)
	scroll.add_child(inner)
	_build_body(inner)
	ZADOSClient.dev_data_received.connect(_on_dev_data)
	ZADOSClient.turn_complete.connect(func(_r): refresh())
	ZADOSClient.request_failed.connect(func(path: String, _error: Dictionary):
		if "/dev/pipeline" in path:
			_turn_lbl.text = "Error"
			_turn_lbl.add_theme_color_override("font_color", Color(0.90, 0.40, 0.35)))


func refresh() -> void:
	ZADOSClient.get_dev("pipeline")


func _on_dev_data(key: String, data: Dictionary) -> void:
	if key != "pipeline":
		return
	if data.get("status", "") in ["no_result_yet", "no_state"]:
		return
	_populate(data)


func _build_body(parent: VBoxContainer) -> void:
	# Session row
	var session_row := HBoxContainer.new()
	session_row.add_theme_constant_override("separation", 20)
	parent.add_child(session_row)
	_turn_lbl    = _meta_field(session_row, "Turn")
	_profile_lbl = _meta_field(session_row, "Reward Profile")
	_mode_lbl    = _meta_field(session_row, "Mode Token")

	# Search/filter (B.4.4)
	var search_row := HBoxContainer.new()
	search_row.add_theme_constant_override("separation", 8)
	parent.add_child(search_row)
	var search_lbl := Label.new()
	search_lbl.text = "Filter engines:"
	search_lbl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	search_lbl.add_theme_font_size_override("font_size", 10)
	search_row.add_child(search_lbl)
	_search_edit = LineEdit.new()
	_search_edit.placeholder_text = "Search..."
	_search_edit.size_flags_horizontal = SIZE_EXPAND_FILL
	_search_edit.add_theme_font_size_override("font_size", 10)
	_search_edit.text_changed.connect(_on_search_changed)
	search_row.add_child(_search_edit)

	# Engines run
	var run_title := Label.new()
	run_title.text = "Engines Run"
	run_title.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	run_title.add_theme_font_size_override("font_size", 11)
	parent.add_child(run_title)
	_engines_run_box = HFlowContainer.new()
	_engines_run_box.add_theme_constant_override("h_separation", 4)
	_engines_run_box.add_theme_constant_override("v_separation", 4)
	parent.add_child(_engines_run_box)

	# Engines skipped
	var skip_title := Label.new()
	skip_title.text = "Engines Skipped"
	skip_title.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	skip_title.add_theme_font_size_override("font_size", 11)
	parent.add_child(skip_title)
	_engines_skip_box = HFlowContainer.new()
	_engines_skip_box.add_theme_constant_override("h_separation", 4)
	_engines_skip_box.add_theme_constant_override("v_separation", 4)
	parent.add_child(_engines_skip_box)

	# Phase timing bar chart (B.4.4)
	var timing_title := Label.new()
	timing_title.text = "Phase Timing (ms)"
	timing_title.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	timing_title.add_theme_font_size_override("font_size", 11)
	parent.add_child(timing_title)
	_timing_box = HBoxContainer.new()
	_timing_box.add_theme_constant_override("separation", 4)
	_timing_box.custom_minimum_size = Vector2(0, 80)
	parent.add_child(_timing_box)

	# Phase summary table
	var phase_title := Label.new()
	phase_title.text = "Phase Summary"
	phase_title.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	phase_title.add_theme_font_size_override("font_size", 11)
	parent.add_child(phase_title)
	_phase_grid = GridContainer.new()
	_phase_grid.columns = 2
	_phase_grid.add_theme_constant_override("h_separation", 12)
	_phase_grid.add_theme_constant_override("v_separation", 3)
	parent.add_child(_phase_grid)

	# Engine weights
	var weights_title := Label.new()
	weights_title.text = "Engine Weights"
	weights_title.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	weights_title.add_theme_font_size_override("font_size", 11)
	parent.add_child(weights_title)
	_weights_grid = GridContainer.new()
	_weights_grid.columns = 2
	_weights_grid.add_theme_constant_override("h_separation", 12)
	_weights_grid.add_theme_constant_override("v_separation", 2)
	parent.add_child(_weights_grid)


func _meta_field(parent: HBoxContainer, title: String) -> Label:
	var col := VBoxContainer.new()
	parent.add_child(col)
	var tl := Label.new()
	tl.text = title
	tl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	tl.add_theme_font_size_override("font_size", 10)
	col.add_child(tl)
	var vl := Label.new()
	vl.text = "—"
	vl.add_theme_color_override("font_color", Color(0.85, 0.88, 0.92))
	vl.add_theme_font_size_override("font_size", 12)
	col.add_child(vl)
	return vl


func _populate(d: Dictionary) -> void:
	_turn_lbl.text    = str(d.get("turn_index", "—"))
	var mod : Dictionary = d.get("modulation", {})
	_profile_lbl.text = str(mod.get("reward_profile_name", "—"))
	_mode_lbl.text    = str(mod.get("mode_token", "—"))

	# Cache engines for filtering
	var dispatch : Dictionary = d.get("dispatch", {})
	_cached_engines_run  = dispatch.get("engines_run", []) as Array
	_cached_engines_skip = dispatch.get("engines_skipped", []) as Array
	_render_engines("")

	# Phase timings (B.4.4)
	_cached_phase_timings.clear()
	var phase_keys := ["perception", "modulation", "dispatch", "thinking", "reward", "answer"]
	for i in phase_keys.size():
		var phase_data = d.get(phase_keys[i], null)
		if phase_data is Dictionary:
			var elapsed_ms : int = phase_data.get("elapsed_ms", 0) as int
			_cached_phase_timings[_PHASE_NAMES[i]] = elapsed_ms
	_render_timing_chart()

	# Phase table
	for child in _phase_grid.get_children():
		child.queue_free()
	for i in phase_keys.size():
		var kl := Label.new()
		kl.text = "P%d %s:" % [i + 1, _PHASE_NAMES[i]]
		kl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		kl.add_theme_font_size_override("font_size", 10)
		kl.custom_minimum_size = Vector2(110, 0)
		_phase_grid.add_child(kl)
		var phase_data = d.get(phase_keys[i], null)
		var status_lbl := Label.new()
		if phase_data == null:
			status_lbl.text = "—"
			status_lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		else:
			var ms_str := ""
			if phase_data is Dictionary and phase_data.has("elapsed_ms"):
				ms_str = " (%dms)" % (phase_data["elapsed_ms"] as int)
			status_lbl.text = "✓ data" + ms_str
			status_lbl.add_theme_color_override("font_color", Color(0.3, 0.7, 0.4))
		status_lbl.add_theme_font_size_override("font_size", 10)
		status_lbl.size_flags_horizontal = SIZE_EXPAND_FILL
		_phase_grid.add_child(status_lbl)

	# Engine weights
	for child in _weights_grid.get_children():
		child.queue_free()
	var weights : Dictionary = mod.get("engine_weights", {})
	for k in weights:
		var kl := Label.new()
		kl.text = str(k) + ":"
		kl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		kl.add_theme_font_size_override("font_size", 10)
		kl.custom_minimum_size = Vector2(80, 0)
		_weights_grid.add_child(kl)
		var vl := Label.new()
		vl.text = str(weights[k])
		vl.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
		vl.add_theme_font_size_override("font_size", 10)
		vl.size_flags_horizontal = SIZE_EXPAND_FILL
		_weights_grid.add_child(vl)


## Render engine chips with optional filter (B.4.4)
func _render_engines(filter: String) -> void:
	for child in _engines_run_box.get_children():
		child.queue_free()
	for child in _engines_skip_box.get_children():
		child.queue_free()

	var f := filter.to_lower()
	var run_count := 0
	for eng_name in _cached_engines_run:
		if not f.is_empty() and f not in str(eng_name).to_lower():
			continue
		_engines_run_box.add_child(_chip(str(eng_name), Color(0.2, 0.6, 0.3)))
		run_count += 1
	if run_count == 0:
		var lbl := Label.new()
		lbl.text = "none" if f.is_empty() else "(no matches)"
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		lbl.add_theme_font_size_override("font_size", 10)
		_engines_run_box.add_child(lbl)

	var skip_count := 0
	for eng_name in _cached_engines_skip:
		if not f.is_empty() and f not in str(eng_name).to_lower():
			continue
		_engines_skip_box.add_child(_chip(str(eng_name), Color(0.35, 0.35, 0.35)))
		skip_count += 1
	if skip_count == 0:
		var lbl := Label.new()
		lbl.text = "none" if f.is_empty() else "(no matches)"
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		lbl.add_theme_font_size_override("font_size", 10)
		_engines_skip_box.add_child(lbl)


func _on_search_changed(text: String) -> void:
	_render_engines(text)


## Phase timing bar chart (B.4.4)
func _render_timing_chart() -> void:
	for child in _timing_box.get_children():
		child.queue_free()

	if _cached_phase_timings.is_empty():
		var lbl := Label.new()
		lbl.text = "No timing data available."
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		lbl.add_theme_font_size_override("font_size", 10)
		_timing_box.add_child(lbl)
		return

	var max_ms : int = 1
	for k in _cached_phase_timings:
		max_ms = maxi(max_ms, _cached_phase_timings[k] as int)

	var bar_colors := [
		Color(0.3, 0.6, 0.9), Color(0.4, 0.8, 0.5), Color(0.9, 0.7, 0.2),
		Color(0.8, 0.4, 0.3), Color(0.6, 0.4, 0.9), Color(0.3, 0.8, 0.8),
	]
	var i := 0
	for phase_name in _cached_phase_timings:
		var ms : int = _cached_phase_timings[phase_name] as int
		var col := VBoxContainer.new()
		col.size_flags_horizontal = SIZE_EXPAND_FILL
		col.add_theme_constant_override("separation", 2)

		# Bar (bottom-aligned via spacer)
		var bar_height := maxf(4.0, (float(ms) / float(max_ms)) * 60.0)
		var spacer := Control.new()
		spacer.size_flags_vertical = SIZE_EXPAND_FILL
		col.add_child(spacer)
		var bar := ColorRect.new()
		bar.color = bar_colors[i % bar_colors.size()]
		bar.custom_minimum_size = Vector2(0, bar_height)
		col.add_child(bar)

		# Value label
		var val_lbl := Label.new()
		val_lbl.text = "%dms" % ms
		val_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		val_lbl.add_theme_color_override("font_color", Color(0.60, 0.62, 0.66))
		val_lbl.add_theme_font_size_override("font_size", 9)
		col.add_child(val_lbl)

		# Phase label
		var name_lbl := Label.new()
		name_lbl.text = phase_name.left(6)
		name_lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		name_lbl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		name_lbl.add_theme_font_size_override("font_size", 8)
		col.add_child(name_lbl)

		_timing_box.add_child(col)
		i += 1


func _chip(text: String, color: Color) -> PanelContainer:
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = color.darkened(0.50)
	style.corner_radius_top_left     = 3; style.corner_radius_top_right    = 3
	style.corner_radius_bottom_left  = 3; style.corner_radius_bottom_right = 3
	style.content_margin_left = 6;  style.content_margin_right  = 6
	style.content_margin_top  = 2;  style.content_margin_bottom = 2
	panel.add_theme_stylebox_override("panel", style)
	var lbl := Label.new()
	lbl.text = text
	lbl.add_theme_color_override("font_color", color.lightened(0.4))
	lbl.add_theme_font_size_override("font_size", 10)
	panel.add_child(lbl)
	return panel


func _make_header() -> HBoxContainer:
	var hbox := HBoxContainer.new()
	var lbl  := Label.new()
	lbl.text = "Pipeline Diagnostics"
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

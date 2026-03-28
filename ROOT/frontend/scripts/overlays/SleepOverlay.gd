##
## SleepOverlay — full-screen overlay for REM and Dream processing.
##
## Covers the WorkspaceContainer during active sleep cycles.
## Contains two views: REM Workspace and Dream Workspace, switched via tabs.
##
extends Control

# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

signal close_requested

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

const _PHASE_COLORS := {
	"WAKING":         Color(0.4, 0.8, 0.4),
	"TRIAGE":         Color(0.8, 0.8, 0.3),
	"REM_PROCESSING": Color(0.3, 0.6, 1.0),
	"DREAM":          Color(0.65, 0.35, 0.85),
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

var _tabs         : TabContainer
var _rem_panel    : VBoxContainer
var _dream_panel  : VBoxContainer

# REM sub-widgets
var _rem_phase_bar     : HBoxContainer
var _rem_packets_list  : VBoxContainer
var _rem_signals_grid  : GridContainer
var _rem_weights_grid  : GridContainer
var _rem_journal_label : RichTextLabel
var _rem_run_btn       : Button

# Dream sub-widgets
var _dream_candidates  : VBoxContainer
var _dream_recombo     : RichTextLabel
var _dream_drivers     : GridContainer
var _dream_plasticity  : ProgressBar
var _dream_journal     : RichTextLabel
var _dream_run_btn     : Button
var _dream_shift_btn   : Button

# Phase tracking
var _rem_phases   : Array = [false, false, false, false]

# ---------------------------------------------------------------------------

func _ready() -> void:
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_build_ui()
	ZADOSClient.sleep_state_received.connect(_on_sleep_state)
	ZADOSClient.rem_complete.connect(_on_rem_complete)
	ZADOSClient.dream_complete.connect(_on_dream_complete)
	ZADOSClient.get_sleep_state()


func _build_ui() -> void:
	# Background
	var bg := StyleBoxFlat.new()
	bg.bg_color = Color(0.06, 0.06, 0.08)
	add_theme_stylebox_override("panel", bg)

	var root := VBoxContainer.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	root.add_theme_constant_override("separation", 0)
	add_child(root)

	# Header bar
	var header := _build_header()
	root.add_child(header)

	# Tab container
	_tabs = TabContainer.new()
	_tabs.size_flags_vertical = SIZE_EXPAND_FILL
	_tabs.add_theme_constant_override("side_margin", 0)
	root.add_child(_tabs)

	_rem_panel = _build_rem_workspace()
	_tabs.add_child(_rem_panel)

	_dream_panel = _build_dream_workspace()
	_tabs.add_child(_dream_panel)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

func _build_header() -> PanelContainer:
	var hbox := HBoxContainer.new()
	hbox.custom_minimum_size = Vector2(0, 40)
	hbox.add_theme_constant_override("separation", 12)

	var bg := StyleBoxFlat.new()
	bg.bg_color = Color(0.08, 0.08, 0.12)
	bg.content_margin_left = 12
	bg.content_margin_right = 12
	bg.content_margin_top = 6
	bg.content_margin_bottom = 6
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", bg)
	panel.size_flags_horizontal = SIZE_EXPAND_FILL
	panel.add_child(hbox)

	var title := Label.new()
	title.text = "SLEEP PROCESSING"
	title.add_theme_color_override("font_color", Color(0.55, 0.45, 0.85))
	title.add_theme_font_size_override("font_size", 14)
	hbox.add_child(title)

	var spacer := Control.new()
	spacer.size_flags_horizontal = SIZE_EXPAND_FILL
	hbox.add_child(spacer)

	var refresh_btn := Button.new()
	refresh_btn.text = "↺ Refresh State"
	refresh_btn.flat = true
	refresh_btn.focus_mode = Control.FOCUS_NONE
	refresh_btn.add_theme_font_size_override("font_size", 11)
	refresh_btn.pressed.connect(func(): ZADOSClient.get_sleep_state())
	hbox.add_child(refresh_btn)

	var close_btn := Button.new()
	close_btn.text = "✕ Close Overlay"
	close_btn.flat = false
	close_btn.focus_mode = Control.FOCUS_NONE
	close_btn.add_theme_font_size_override("font_size", 11)
	close_btn.pressed.connect(func(): close_requested.emit())
	hbox.add_child(close_btn)

	return panel


# ---------------------------------------------------------------------------
# REM Workspace
# ---------------------------------------------------------------------------

func _build_rem_workspace() -> VBoxContainer:
	var vbox := VBoxContainer.new()
	vbox.name = "REM Processing"
	vbox.add_theme_constant_override("separation", 8)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	vbox.add_child(scroll)

	var inner := VBoxContainer.new()
	inner.size_flags_horizontal = SIZE_EXPAND_FILL
	inner.add_theme_constant_override("separation", 10)
	scroll.add_child(inner)

	# Phase progress tracker
	inner.add_child(_section_label("Phase Progress"))
	_rem_phase_bar = HBoxContainer.new()
	_rem_phase_bar.add_theme_constant_override("separation", 4)
	inner.add_child(_rem_phase_bar)
	var rem_phase_names := [
		"1: Retroactive Learning",
		"2: MTMM→LTMM Consolidation",
		"3: NT Stabilization",
		"4: Journal Write",
	]
	for i in 4:
		_rem_phase_bar.add_child(_make_phase_chip(rem_phase_names[i], false))

	# Phase 1 — Emotional signals from packets
	inner.add_child(_section_label("Phase 1 — Retroactive Learning (Emotional Signals)"))
	_rem_signals_grid = GridContainer.new()
	_rem_signals_grid.columns = 2
	_rem_signals_grid.add_theme_constant_override("h_separation", 12)
	_rem_signals_grid.add_theme_constant_override("v_separation", 3)
	inner.add_child(_rem_signals_grid)

	# Phase 2/3 — Domain weight adjustments
	inner.add_child(_section_label("Domain Weight Adjustments"))
	_rem_weights_grid = GridContainer.new()
	_rem_weights_grid.columns = 2
	_rem_weights_grid.add_theme_constant_override("h_separation", 12)
	_rem_weights_grid.add_theme_constant_override("v_separation", 3)
	inner.add_child(_rem_weights_grid)

	# MTMM packets scanned
	inner.add_child(_section_label("Consolidation — Packet Queue"))
	_rem_packets_list = VBoxContainer.new()
	_rem_packets_list.add_theme_constant_override("separation", 4)
	inner.add_child(_rem_packets_list)
	var placeholder := Label.new()
	placeholder.text = "Run REM to scan MTMM packets for emotional signals."
	placeholder.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
	placeholder.add_theme_font_size_override("font_size", 10)
	_rem_packets_list.add_child(placeholder)

	# Phase 4 — Journal preview
	inner.add_child(_section_label("Phase 4 — Journal Entry"))
	_rem_journal_label = RichTextLabel.new()
	_rem_journal_label.bbcode_enabled = true
	_rem_journal_label.fit_content = true
	_rem_journal_label.scroll_active = false
	_rem_journal_label.custom_minimum_size = Vector2(0, 60)
	_rem_journal_label.add_theme_color_override("default_color", Color(0.60, 0.62, 0.66))
	_rem_journal_label.add_theme_font_size_override("font_size", 11)
	_rem_journal_label.text = "[i]Journal entry will appear here after REM completes.[/i]"
	inner.add_child(_rem_journal_label)

	# Run button
	var sep := HSeparator.new()
	inner.add_child(sep)
	_rem_run_btn = Button.new()
	_rem_run_btn.text = "Run REM Pipeline"
	_rem_run_btn.focus_mode = Control.FOCUS_NONE
	_rem_run_btn.add_theme_font_size_override("font_size", 12)
	_rem_run_btn.add_theme_color_override("font_color", Color(0.3, 0.6, 1.0))
	_rem_run_btn.pressed.connect(_on_rem_run)
	inner.add_child(_rem_run_btn)

	return vbox


# ---------------------------------------------------------------------------
# Dream Workspace
# ---------------------------------------------------------------------------

func _build_dream_workspace() -> VBoxContainer:
	var vbox := VBoxContainer.new()
	vbox.name = "Dream Processing"
	vbox.add_theme_constant_override("separation", 8)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	vbox.add_child(scroll)

	var inner := VBoxContainer.new()
	inner.size_flags_horizontal = SIZE_EXPAND_FILL
	inner.add_theme_constant_override("separation", 10)
	scroll.add_child(inner)

	# Dream candidate queue
	inner.add_child(_section_label("Dream Candidate Queue (3-tier priority)"))
	_dream_candidates = VBoxContainer.new()
	_dream_candidates.add_theme_constant_override("separation", 4)
	inner.add_child(_dream_candidates)

	# Emotional driver profile
	inner.add_child(_section_label("Emotional Driver Profile"))
	_dream_drivers = GridContainer.new()
	_dream_drivers.columns = 2
	_dream_drivers.add_theme_constant_override("h_separation", 12)
	_dream_drivers.add_theme_constant_override("v_separation", 3)
	inner.add_child(_dream_drivers)

	# Narrative plasticity gauge
	inner.add_child(_section_label("Narrative Plasticity"))
	_dream_plasticity = ProgressBar.new()
	_dream_plasticity.min_value = 0.0
	_dream_plasticity.max_value = 1.0
	_dream_plasticity.value = 0.0
	_dream_plasticity.custom_minimum_size = Vector2(0, 18)
	inner.add_child(_dream_plasticity)

	# Recombination panel
	inner.add_child(_section_label("Creative Recombination Output"))
	_dream_recombo = RichTextLabel.new()
	_dream_recombo.bbcode_enabled = true
	_dream_recombo.fit_content = true
	_dream_recombo.scroll_active = false
	_dream_recombo.custom_minimum_size = Vector2(0, 80)
	_dream_recombo.add_theme_color_override("default_color", Color(0.65, 0.55, 0.85))
	_dream_recombo.add_theme_font_size_override("font_size", 11)
	_dream_recombo.text = "[i]Novel connections will appear here after Dream processing.[/i]"
	inner.add_child(_dream_recombo)

	# Journal integration
	inner.add_child(_section_label("Dream Journal"))
	_dream_journal = RichTextLabel.new()
	_dream_journal.bbcode_enabled = true
	_dream_journal.fit_content = true
	_dream_journal.scroll_active = false
	_dream_journal.custom_minimum_size = Vector2(0, 60)
	_dream_journal.add_theme_color_override("default_color", Color(0.60, 0.62, 0.66))
	_dream_journal.add_theme_font_size_override("font_size", 11)
	_dream_journal.text = "[i]Journal entry will appear here after Dream completes.[/i]"
	inner.add_child(_dream_journal)

	# Action buttons
	var sep := HSeparator.new()
	inner.add_child(sep)
	var btn_row := HBoxContainer.new()
	btn_row.add_theme_constant_override("separation", 8)
	inner.add_child(btn_row)

	_dream_run_btn = Button.new()
	_dream_run_btn.text = "Run Dream Pipeline"
	_dream_run_btn.focus_mode = Control.FOCUS_NONE
	_dream_run_btn.add_theme_font_size_override("font_size", 12)
	_dream_run_btn.add_theme_color_override("font_color", Color(0.65, 0.35, 0.85))
	_dream_run_btn.pressed.connect(_on_dream_run)
	btn_row.add_child(_dream_run_btn)

	_dream_shift_btn = Button.new()
	_dream_shift_btn.text = "Scene Shift"
	_dream_shift_btn.flat = true
	_dream_shift_btn.focus_mode = Control.FOCUS_NONE
	_dream_shift_btn.add_theme_font_size_override("font_size", 11)
	_dream_shift_btn.tooltip_text = "Trigger dream_scene_shift — writes journal for current scene, moves to next batch"
	_dream_shift_btn.pressed.connect(_on_scene_shift)
	btn_row.add_child(_dream_shift_btn)

	return vbox


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

func _on_sleep_state(data: Dictionary) -> void:
	# Populate dream candidates
	for child in _dream_candidates.get_children():
		child.queue_free()
	var candidates : Array = data.get("dream_candidates", [])
	if candidates.is_empty():
		var lbl := Label.new()
		lbl.text = "No dream candidates in unsolved buffer."
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		lbl.add_theme_font_size_override("font_size", 10)
		_dream_candidates.add_child(lbl)
	else:
		for c in candidates:
			_dream_candidates.add_child(_make_candidate_card(c))

	# MTMM packet summary
	for child in _rem_packets_list.get_children():
		child.queue_free()
	var mtmm : Dictionary = data.get("mtmm_packets", {})
	var summary_lbl := Label.new()
	summary_lbl.text = "Total packets: %d  |  High significance: %d  |  Low trust (decay): %d" % [
		mtmm.get("total", 0), mtmm.get("high_significance", 0), mtmm.get("low_trust", 0)]
	summary_lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.80))
	summary_lbl.add_theme_font_size_override("font_size", 11)
	_rem_packets_list.add_child(summary_lbl)

	# Plasticity from NT snapshot
	var nt : Dictionary = data.get("nt_snapshot", {})
	var cb1_val := _extract_float(nt.get("CB1", nt.get("cb1", 0.0)))
	var glu_val := _extract_float(nt.get("GLU", nt.get("glu", 0.0)))
	_dream_plasticity.value = clampf((cb1_val + glu_val) / 2.0, 0.0, 1.0)


func _on_rem_complete(result: Dictionary) -> void:
	_rem_run_btn.text = "Run REM Pipeline"
	_rem_run_btn.disabled = false

	# Update phase chips
	for i in 4:
		_rem_phases[i] = true
		var chip := _rem_phase_bar.get_child(i)
		if chip is PanelContainer:
			_update_phase_chip(chip, true)

	# Populate signals
	for child in _rem_signals_grid.get_children():
		child.queue_free()
	var signals : Array = result.get("dominant_signals", [])
	if signals.is_empty():
		signals = ["(none detected)"]
	for s in signals:
		var kl := Label.new()
		kl.text = "Signal:"
		kl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		kl.add_theme_font_size_override("font_size", 10)
		_rem_signals_grid.add_child(kl)
		var vl := Label.new()
		vl.text = str(s)
		vl.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
		vl.add_theme_font_size_override("font_size", 10)
		_rem_signals_grid.add_child(vl)

	# Populate weight adjustments
	for child in _rem_weights_grid.get_children():
		child.queue_free()
	var weights : Dictionary = result.get("domain_weight_adjustments", {})
	for k in weights:
		var kl := Label.new()
		kl.text = str(k) + ":"
		kl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		kl.add_theme_font_size_override("font_size", 10)
		_rem_weights_grid.add_child(kl)
		var val : float = float(weights[k])
		var vl := Label.new()
		vl.text = "%+.4f" % val
		vl.add_theme_color_override("font_color",
			Color(0.45, 0.80, 0.45) if val > 0 else Color(0.80, 0.45, 0.45))
		vl.add_theme_font_size_override("font_size", 10)
		_rem_weights_grid.add_child(vl)

	# Packets summary
	for child in _rem_packets_list.get_children():
		child.queue_free()
	var scanned : int = result.get("packets_scanned", 0)
	var consolidated : int = result.get("packets_consolidated", 0)
	var pkt_lbl := Label.new()
	pkt_lbl.text = "Scanned: %d  |  Consolidated to LTMM: %d" % [scanned, consolidated]
	pkt_lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.80))
	pkt_lbl.add_theme_font_size_override("font_size", 11)
	_rem_packets_list.add_child(pkt_lbl)

	# Journal preview
	_rem_journal_label.clear()
	_rem_journal_label.append_text("[color=#8888aa]REM cycle complete. Processing time: %.2fs[/color]" %
		float(result.get("processing_time_s", 0.0)))

	ZADOSClient.get_sleep_state()


func _on_dream_complete(result: Dictionary) -> void:
	_dream_run_btn.text = "Run Dream Pipeline"
	_dream_run_btn.disabled = false

	# Drivers
	for child in _dream_drivers.get_children():
		child.queue_free()
	var signals : Array = result.get("dominant_signals", [])
	for s in signals:
		var kl := Label.new()
		kl.text = "Driver:"
		kl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		kl.add_theme_font_size_override("font_size", 10)
		_dream_drivers.add_child(kl)
		var vl := Label.new()
		vl.text = str(s)
		vl.add_theme_color_override("font_color", Color(0.65, 0.55, 0.85))
		vl.add_theme_font_size_override("font_size", 10)
		_dream_drivers.add_child(vl)
	# Weight adjustments in drivers grid too
	var weights : Dictionary = result.get("domain_weight_adjustments", {})
	for k in weights:
		var kl := Label.new()
		kl.text = str(k) + ":"
		kl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		kl.add_theme_font_size_override("font_size", 10)
		_dream_drivers.add_child(kl)
		var val : float = float(weights[k])
		var vl := Label.new()
		vl.text = "%+.4f" % val
		vl.add_theme_color_override("font_color",
			Color(0.45, 0.80, 0.45) if val > 0 else Color(0.80, 0.45, 0.45))
		vl.add_theme_font_size_override("font_size", 10)
		_dream_drivers.add_child(vl)

	# Recombination results
	_dream_recombo.clear()
	var found : int = result.get("candidates_found", 0)
	var processed : int = result.get("candidates_processed", 0)
	var novel : int = result.get("novel_connections", 0)
	_dream_recombo.append_text("[color=#9977cc]Candidates found: %d  |  Processed: %d  |  Novel connections: %d[/color]\n" % [found, processed, novel])
	_dream_recombo.append_text("[color=#8888aa]Processing time: %.2fs[/color]" %
		float(result.get("processing_time_s", 0.0)))

	# Journal
	_dream_journal.clear()
	_dream_journal.append_text("[color=#8888aa]Dream cycle complete. Check LTMM Journal for generated entries.[/color]")

	ZADOSClient.get_sleep_state()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

func _on_rem_run() -> void:
	_rem_run_btn.text = "Running REM..."
	_rem_run_btn.disabled = true
	_rem_phases = [false, false, false, false]
	for i in 4:
		var chip := _rem_phase_bar.get_child(i)
		if chip is PanelContainer:
			_update_phase_chip(chip, false)
	ZADOSClient.run_rem()


func _on_dream_run() -> void:
	_dream_run_btn.text = "Running Dream..."
	_dream_run_btn.disabled = true
	ZADOSClient.run_dream()


func _on_scene_shift() -> void:
	# Scene shift triggers a new dream run (which processes next batch)
	_dream_journal.clear()
	_dream_journal.append_text("[i]Scene shift — writing journal for current scene...[/i]")
	ZADOSClient.run_dream()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

func _section_label(title: String) -> Label:
	var lbl := Label.new()
	lbl.text = title
	lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.85))
	lbl.add_theme_font_size_override("font_size", 12)
	return lbl


func _make_phase_chip(title: String, done: bool) -> PanelContainer:
	var panel := PanelContainer.new()
	panel.size_flags_horizontal = SIZE_EXPAND_FILL
	var style := StyleBoxFlat.new()
	style.set_corner_radius_all(4)
	style.content_margin_left = 8
	style.content_margin_right = 8
	style.content_margin_top = 4
	style.content_margin_bottom = 4
	style.bg_color = Color(0.15, 0.25, 0.15) if done else Color(0.10, 0.10, 0.13)
	style.border_color = Color(0.30, 0.70, 0.35) if done else Color(0.25, 0.25, 0.30)
	style.border_width_bottom = 2
	panel.add_theme_stylebox_override("panel", style)
	var lbl := Label.new()
	lbl.text = ("✓ " if done else "○ ") + title
	lbl.add_theme_color_override("font_color",
		Color(0.50, 0.85, 0.55) if done else Color(0.45, 0.45, 0.50))
	lbl.add_theme_font_size_override("font_size", 10)
	panel.add_child(lbl)
	return panel


func _update_phase_chip(chip: PanelContainer, done: bool) -> void:
	var style := StyleBoxFlat.new()
	style.set_corner_radius_all(4)
	style.content_margin_left = 8
	style.content_margin_right = 8
	style.content_margin_top = 4
	style.content_margin_bottom = 4
	style.bg_color = Color(0.15, 0.25, 0.15) if done else Color(0.10, 0.10, 0.13)
	style.border_color = Color(0.30, 0.70, 0.35) if done else Color(0.25, 0.25, 0.30)
	style.border_width_bottom = 2
	chip.add_theme_stylebox_override("panel", style)
	var lbl := chip.get_child(0) as Label
	if lbl:
		var text : String = lbl.text
		if done and not text.begins_with("✓"):
			lbl.text = "✓" + text.substr(1)
			lbl.add_theme_color_override("font_color", Color(0.50, 0.85, 0.55))
		elif not done and not text.begins_with("○"):
			lbl.text = "○" + text.substr(1)
			lbl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))


func _make_candidate_card(c: Dictionary) -> PanelContainer:
	var is_dream : bool = c.get("dream_candidate", false)
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.10, 0.10, 0.13)
	style.set_corner_radius_all(5)
	style.content_margin_left = 10
	style.content_margin_right = 10
	style.content_margin_top = 6
	style.content_margin_bottom = 6
	if is_dream:
		style.border_width_left = 2
		style.border_color = Color(0.65, 0.35, 0.85)
	panel.add_theme_stylebox_override("panel", style)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 3)
	panel.add_child(vbox)

	var hdr := HBoxContainer.new()
	var source_lbl := Label.new()
	source_lbl.text = str(c.get("source_engine", "—"))
	source_lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.80))
	source_lbl.add_theme_font_size_override("font_size", 10)
	hdr.add_child(source_lbl)
	var spacer := Control.new()
	spacer.size_flags_horizontal = SIZE_EXPAND_FILL
	hdr.add_child(spacer)
	var stag := Label.new()
	stag.text = "%d cycles stagnated" % c.get("stagnation_cycles", 0)
	stag.add_theme_color_override("font_color", Color(0.80, 0.55, 0.25))
	stag.add_theme_font_size_override("font_size", 9)
	hdr.add_child(stag)
	vbox.add_child(hdr)

	var ql := Label.new()
	ql.text = str(c.get("concept_formulation", "")).left(200)
	ql.add_theme_color_override("font_color", Color(0.85, 0.88, 0.92))
	ql.add_theme_font_size_override("font_size", 11)
	ql.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	vbox.add_child(ql)

	return panel


func _extract_float(val) -> float:
	if val is float or val is int:
		return float(val)
	if val is Dictionary:
		for k in ["tonic", "level", "value", "concentration"]:
			if val.has(k):
				return float(val[k])
	return 0.0

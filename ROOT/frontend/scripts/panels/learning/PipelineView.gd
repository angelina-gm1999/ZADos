##
## PipelineView — right panel of the Learning Workspace.
## Shows mode-specific controls, LTMM data, and pipeline results.
##
extends ScrollContainer

# ---------------------------------------------------------------------------
# Mode configuration data (from LearningModeConfig in the ZADOS backend)
# ---------------------------------------------------------------------------

const MODE_INFO := {
	"Normal": {
		"title":       "Regular Input",
		"description": "Standard conversation mode. Full AnswerPipeline runs each turn.\nAll 29 cognitive engines available. NT feedback loop active.",
		"color":       Color(0.55, 0.55, 0.55),
		"config":      {},
	},
	"M1": {
		"title":       "M1 — Teach Me",
		"description": "You teach ZADOS something. High ACh encoding, mild DA-D1 salience.\nOXT receptivity boost. GABA noise suppression. Low NE.",
		"color":       Color(0.30, 0.55, 0.95),
		"config": {
			"Semantic hops":     "2",
			"Pattern depth":     "2",
			"Max questions/turn":"2",
			"Response depth":    "full",
			"Contradiction mode":"learning",
			"Engine budget":     "14 (T1+T2)",
		},
		"risks": ["frustrated", "defensiveness", "overwhelmed"],
	},
	"M2": {
		"title":       "M2 — Review",
		"description": "ZADOS reviews / quizzes you. High NE vigilance, high ACh attention.\n5-HT1A buffering, mild cortisol. Two-pass retroactive memory contrast.",
		"color":       Color(0.90, 0.60, 0.20),
		"config": {
			"Semantic hops":       "3",
			"Pattern depth":       "3",
			"Max questions/turn":  "0 (no questions — quiz only)",
			"Response depth":      "full",
			"Retroactive contrast":"yes",
			"Contradiction mode":  "soft",
			"Engine budget":       "16 (T1+T2)",
		},
		"risks": ["ashamed", "contempt", "dismissiveness"],
	},
	"M3": {
		"title":       "M3 — Explore",
		"description": "Full Socratic / dialectic mode. Max DA-D3 exploration, CB1 flexibility.\n5-HT2A symbolic processing. ZADOS actively challenges claims.",
		"color":       Color(0.25, 0.75, 0.72),
		"config": {
			"Semantic hops":     "unlimited",
			"Pattern depth":     "unlimited",
			"Max questions/turn":"unlimited",
			"Response depth":    "full",
			"Contradiction mode":"learning",
			"Stochastic path":   "active",
			"Engine budget":     "18 (T1+T2, all clusters)",
		},
		"risks": ["confused", "overwhelmed", "frustrated"],
	},
	"M4": {
		"title":       "M4 — Questions",
		"description": "Driven by the Unsolved Questions buffer. Max DA-D3 curiosity.\n5-HT2A abstract, ACh attention. Sub-modes: automatic / prompted / clustered.",
		"color":       Color(0.65, 0.35, 0.85),
		"config": {
			"Semantic hops":     "3",
			"Pattern depth":     "2",
			"Max questions/turn":"1 (one focused question)",
			"Response depth":    "abbreviated",
			"Contradiction mode":"learning",
			"Engine budget":     "12 (T1+T2)",
		},
		"risks": ["rumination", "apathy", "stagnation"],
	},
	"M5": {
		"title":       "M5 — Independent",
		"description": "Fully autonomous. No response output — ZADOS studies internally.\nMax ACh-alpha7/M1 attention, DA-D1 goal salience. NT-boredom detection.",
		"color":       Color(0.80, 0.45, 0.25),
		"config": {
			"Semantic hops":     "3",
			"Pattern depth":     "3",
			"Max questions/turn":"2",
			"Response depth":    "none  (autonomous — no output)",
			"E28 emotion input": "off",
			"Engine budget":     "14 (T1+T2)",
		},
		"risks": ["boredom", "apathy", "confused"],
	},
	"Homework": {
		"title":       "Homework Pipeline",
		"description": "6-phase autonomous processing of accumulated learning log entries.\nNo user present during run. NT layer is read-only.",
		"color":       Color(0.90, 0.65, 0.20),
		"config":      {},
	},
	"Reflective": {
		"title":       "Reflective Synthesis",
		"description": "E31 meta-learning analysis + E32 identity coherence analysis.\nCorrelates learning patterns with identity contradictions.",
		"color":       Color(0.45, 0.40, 0.85),
		"config":      {},
	},
	"SelfReflective": {
		"title":       "Self-Reflective Query",
		"description": "Auto-activated when self-ref markers are present AND the Unsolved\nBuffer is non-empty. Routes via M3 (dialectic) engine tier.\n\nTrigger phrases: 'what do I think', 'reflect on', 'how do I feel about',\n'what have I learned', 'introspect', 'examine my thinking'.",
		"color":       Color(0.30, 0.80, 0.60),
		"config":      {},
	},
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

var _current_mode : String = "Normal"
var _content      : VBoxContainer
var _is_running   : bool   = false

# Homework result sections (kept across refreshes)
var _hw_phases_vbox  : VBoxContainer = null
var _hw_summary_vbox : VBoxContainer = null
var _ref_result_vbox : VBoxContainer = null

# ---------------------------------------------------------------------------

func _ready() -> void:
	horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	var margin := MarginContainer.new()
	margin.size_flags_horizontal = SIZE_EXPAND_FILL
	margin.size_flags_vertical = SIZE_EXPAND_FILL
	margin.add_theme_constant_override("margin_left",  12)
	margin.add_theme_constant_override("margin_right", 12)
	margin.add_theme_constant_override("margin_top",   10)
	add_child(margin)
	_content = VBoxContainer.new()
	_content.size_flags_horizontal = SIZE_EXPAND_FILL
	_content.add_theme_constant_override("separation", 8)
	margin.add_child(_content)

	ZADOSClient.homework_complete.connect(_on_homework_done)
	ZADOSClient.reflective_complete.connect(_on_reflective_done)
	ZADOSClient.memory_data_received.connect(_on_memory_data)
	ZADOSClient.turn_complete.connect(_on_turn_complete)

	show_mode("Normal")


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

func show_mode(mode: String) -> void:
	_current_mode = mode
	_rebuild()


# ---------------------------------------------------------------------------
# Rebuild UI for current mode
# ---------------------------------------------------------------------------

func _rebuild() -> void:
	for child in _content.get_children():
		child.queue_free()
	_hw_phases_vbox  = null
	_hw_summary_vbox = null
	_ref_result_vbox = null

	var info : Dictionary = MODE_INFO.get(_current_mode, MODE_INFO["Normal"])
	_header(info)

	match _current_mode:
		"Normal":
			_build_normal_view()
		"M1", "M2", "M3", "M4", "M5":
			_build_learning_view(info)
		"Homework":
			_build_homework_view()
		"Reflective":
			_build_reflective_view()
		"SelfReflective":
			_build_self_ref_view()


# ---------------------------------------------------------------------------
# Mode views
# ---------------------------------------------------------------------------

func _build_normal_view() -> void:
	_section("Tips")
	var tips := [
		"Send a message in the Conversation tab to begin.",
		"ZADOS will route to M1–M5 automatically when it detects learning markers.",
		"Use the mode selector on the left to pre-set the routing.",
	]
	for tip in tips:
		_bullet(tip, Color(0.65, 0.68, 0.72))


func _build_learning_view(info: Dictionary) -> void:
	# Config table
	var config : Dictionary = info.get("config", {})
	if not config.is_empty():
		_section("Mode Configuration")
		_fields_card(config, Color(0.10, 0.10, 0.13))

	# Risk emotions
	var risks : Array = info.get("risks", [])
	if not risks.is_empty():
		_section("Watch-for Emotional Risks")
		var risk_row := HFlowContainer.new()
		risk_row.add_theme_constant_override("h_separation", 6)
		for r in risks:
			risk_row.add_child(_badge(r, Color(0.85, 0.35, 0.25)))
		_content.add_child(risk_row)

	# Switch to Regular button
	var btn := Button.new()
	btn.text = "Switch to Regular"
	btn.flat = false
	btn.focus_mode = Control.FOCUS_NONE
	btn.add_theme_font_size_override("font_size", 11)
	btn.pressed.connect(func():
		ZADOSClient.set_session_mode("Normal"))
	_content.add_child(btn)

	# LTMM: Lessons generated this session
	_section("Lessons in LTMM  (from learning turns)")
	var lessons_hdr := HBoxContainer.new()
	var lessons_lbl := Label.new()
	lessons_lbl.text = "Fetching…"
	lessons_lbl.size_flags_horizontal = SIZE_EXPAND_FILL
	lessons_lbl.add_theme_color_override("font_color", Color(0.5, 0.5, 0.5))
	lessons_lbl.add_theme_font_size_override("font_size", 11)
	lessons_hdr.add_child(lessons_lbl)
	var refresh_btn := Button.new()
	refresh_btn.text = "↺"
	refresh_btn.flat = true
	refresh_btn.focus_mode = Control.FOCUS_NONE
	refresh_btn.add_theme_font_size_override("font_size", 11)
	refresh_btn.pressed.connect(func(): ZADOSClient.get_memory("ltmm/knowledge/lessons"))
	lessons_hdr.add_child(refresh_btn)
	_content.add_child(lessons_hdr)

	# Placeholder list — filled when memory_data_received fires
	var lessons_list := VBoxContainer.new()
	lessons_list.name = "LessonsList"
	lessons_list.add_theme_constant_override("separation", 4)
	_content.add_child(lessons_list)

	ZADOSClient.get_memory("ltmm/knowledge/lessons")

	# Unsolved questions count
	_section("Unsolved Questions (LTMM)")
	var unsolved_row := HBoxContainer.new()
	var unsolved_lbl := Label.new()
	unsolved_lbl.name = "UnsolvedLabel"
	unsolved_lbl.text = "Fetching…"
	unsolved_lbl.add_theme_color_override("font_color", Color(0.65, 0.55, 0.85))
	unsolved_lbl.add_theme_font_size_override("font_size", 11)
	unsolved_row.add_child(unsolved_lbl)
	_content.add_child(unsolved_row)
	ZADOSClient.get_memory("ltmm/unsolved")


func _build_homework_view() -> void:
	# Explanation
	_section("How Homework Works")
	var phases := [
		"Phase 0 — Input Assembly & Triage  (batch learning logs, NT deficit bias)",
		"Phase 1 — Analysis  (decomposition, memory contrast, novel patterns)",
		"Phase 2 — Processing  (contradiction resolution, fallacy/bias sweep)",
		"Phase 3 — Question Resolution  (unsolved buffer update, dream candidate flagging)",
		"Phase 4 — Synthesis  (lessons finalized, knowledge maps, core memory gating)",
		"Phase 5 — Output  (LTMM writes, overview log, journal entry)",
	]
	for ph in phases:
		_bullet(ph, Color(0.70, 0.72, 0.76))

	# Run button
	var run_btn := Button.new()
	run_btn.name = "RunButton"
	run_btn.text = "▶  Run Homework Pipeline"
	run_btn.focus_mode = Control.FOCUS_NONE
	run_btn.add_theme_font_size_override("font_size", 13)
	run_btn.pressed.connect(_on_run_homework)
	_content.add_child(run_btn)

	# Status label
	var status := Label.new()
	status.name = "StatusLabel"
	status.text = "Not yet run this session."
	status.add_theme_color_override("font_color", Color(0.5, 0.5, 0.5))
	status.add_theme_font_size_override("font_size", 11)
	_content.add_child(status)

	# Phase progress (populated after run)
	_section("Phase Progress")
	_hw_phases_vbox = VBoxContainer.new()
	_hw_phases_vbox.add_theme_constant_override("separation", 4)
	_add_phase_stubs(_hw_phases_vbox)
	_content.add_child(_hw_phases_vbox)

	# Run summary (populated after run)
	_section("Last Run Summary")
	_hw_summary_vbox = VBoxContainer.new()
	_hw_summary_vbox.add_theme_constant_override("separation", 4)
	var placeholder := Label.new()
	placeholder.text = "Run the pipeline to see summary."
	placeholder.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
	placeholder.add_theme_font_size_override("font_size", 11)
	_hw_summary_vbox.add_child(placeholder)
	_content.add_child(_hw_summary_vbox)

	# LTMM links
	_section("LTMM Data  (refresh after run)")
	_ltmm_links()


func _build_reflective_view() -> void:
	_section("What Reflective Does")
	var steps := [
		"Phase 0 — Input Assembly  (learning logs, identity stores, pending updates)",
		"Phase 1 — E31 Meta-Learning Analysis  (patterns, failures, mode effectiveness)",
		"Phase 2 — E32 Identity Coherence  (contradictions, fragility, alignment issues)",
		"Phase 3 — Cross-Reference  (E31 patterns × E32 contradictions)",
		"Phase 4 — Identity Store Mutations  (reinforce conclusions, journal writes)",
		"Phase 5 — Summary Output",
	]
	for s in steps:
		_bullet(s, Color(0.70, 0.72, 0.76))

	var run_btn := Button.new()
	run_btn.name = "RunButton"
	run_btn.text = "▶  Run Reflective Synthesis"
	run_btn.focus_mode = Control.FOCUS_NONE
	run_btn.add_theme_font_size_override("font_size", 13)
	run_btn.pressed.connect(_on_run_reflective)
	_content.add_child(run_btn)

	var status := Label.new()
	status.name = "StatusLabel"
	status.text = "Not yet run this session."
	status.add_theme_color_override("font_color", Color(0.5, 0.5, 0.5))
	status.add_theme_font_size_override("font_size", 11)
	_content.add_child(status)

	_section("Synthesis Result")
	_ref_result_vbox = VBoxContainer.new()
	_ref_result_vbox.add_theme_constant_override("separation", 4)
	var placeholder := Label.new()
	placeholder.text = "Run synthesis to see results."
	placeholder.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
	placeholder.add_theme_font_size_override("font_size", 11)
	_ref_result_vbox.add_child(placeholder)
	_content.add_child(_ref_result_vbox)


func _build_self_ref_view() -> void:
	_section("How Self-Reflective Activates")
	var info := Label.new()
	info.text = (
		"This mode is auto-detected — it cannot be forced via the mode selector.\n\n"
		+ "Conditions for activation:\n"
		+ "  1. Input contains self-reflective markers  AND\n"
		+ "  2. The Unsolved Questions buffer is non-empty\n\n"
		+ "Trigger phrases:\n"
		+ "  • 'what do I think about'\n"
		+ "  • 'how do I feel about'\n"
		+ "  • 'reflect on [topic]'\n"
		+ "  • 'what have I learned'\n"
		+ "  • 'introspect on'\n"
		+ "  • 'examine my thinking about'\n\n"
		+ "Routes via M3 (dialectic) engine tier on the selected unsolved question."
	)
	info.add_theme_color_override("font_color", Color(0.70, 0.72, 0.76))
	info.add_theme_font_size_override("font_size", 11)
	info.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_content.add_child(info)

	_section("Unsolved Buffer Status")
	var unsolved_lbl := Label.new()
	unsolved_lbl.name = "UnsolvedLabel"
	unsolved_lbl.text = "Fetching…"
	unsolved_lbl.add_theme_color_override("font_color", Color(0.65, 0.55, 0.85))
	unsolved_lbl.add_theme_font_size_override("font_size", 12)
	_content.add_child(unsolved_lbl)
	ZADOSClient.get_memory("ltmm/unsolved")


# ---------------------------------------------------------------------------
# Pipeline run handlers
# ---------------------------------------------------------------------------

func _on_run_homework() -> void:
	if _is_running:
		return
	_is_running = true
	var btn := _content.find_child("RunButton", true, false)
	if btn:
		(btn as Button).disabled = true
		(btn as Button).text = "⏳  Running…"
	var status := _content.find_child("StatusLabel", true, false)
	if status:
		(status as Label).text = "Pipeline running…"
		(status as Label).add_theme_color_override("font_color", Color(0.90, 0.75, 0.25))
	_animate_phases_running()
	ZADOSClient.run_homework()


func _on_run_reflective() -> void:
	if _is_running:
		return
	_is_running = true
	var btn := _content.find_child("RunButton", true, false)
	if btn:
		(btn as Button).disabled = true
		(btn as Button).text = "⏳  Running…"
	var status := _content.find_child("StatusLabel", true, false)
	if status:
		(status as Label).text = "Synthesis running…"
		(status as Label).add_theme_color_override("font_color", Color(0.90, 0.75, 0.25))
	ZADOSClient.run_reflective()


# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

func _on_turn_complete(_r: Dictionary) -> void:
	# After any turn in M1-M5, refresh LTMM lessons
	if _current_mode in ["M1","M2","M3","M4","M5"]:
		ZADOSClient.get_memory("ltmm/knowledge/lessons")
		ZADOSClient.get_memory("ltmm/unsolved")


func _on_homework_done(result: Dictionary) -> void:
	_is_running = false
	var btn := _content.find_child("RunButton", true, false)
	if btn:
		(btn as Button).disabled = false
		(btn as Button).text = "▶  Run Homework Pipeline"
	var status := _content.find_child("StatusLabel", true, false)
	if status:
		(status as Label).text = "Completed."
		(status as Label).add_theme_color_override("font_color", Color(0.25, 0.85, 0.45))
	_populate_hw_phases(result)
	_populate_hw_summary(result)


func _on_reflective_done(result: Dictionary) -> void:
	_is_running = false
	var btn := _content.find_child("RunButton", true, false)
	if btn:
		(btn as Button).disabled = false
		(btn as Button).text = "▶  Run Reflective Synthesis"
	var status := _content.find_child("StatusLabel", true, false)
	if status:
		(status as Label).text = "Synthesis complete."
		(status as Label).add_theme_color_override("font_color", Color(0.25, 0.85, 0.45))
	_populate_reflective_result(result)


func _on_memory_data(key: String, data: Dictionary) -> void:
	match key:
		"ltmm/knowledge/lessons":
			_update_lessons_list(data)
		"ltmm/unsolved":
			_update_unsolved_count(data)


# ---------------------------------------------------------------------------
# Homework phase display
# ---------------------------------------------------------------------------

const PHASE_NAMES := [
	"Phase 0 — Input Assembly & Triage",
	"Phase 1 — Analysis",
	"Phase 2 — Processing",
	"Phase 3 — Question Resolution",
	"Phase 4 — Synthesis & Integration",
	"Phase 5 — Output & Storage",
]

func _add_phase_stubs(container: VBoxContainer) -> void:
	for i in range(6):
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 8)
		var dot := Label.new()
		dot.text = "[ ]"
		dot.add_theme_color_override("font_color", Color(0.35, 0.35, 0.40))
		dot.add_theme_font_size_override("font_size", 11)
		dot.custom_minimum_size = Vector2(24, 0)
		row.add_child(dot)
		var lbl := Label.new()
		lbl.text = PHASE_NAMES[i]
		lbl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		lbl.add_theme_font_size_override("font_size", 11)
		row.add_child(lbl)
		container.add_child(row)


func _animate_phases_running() -> void:
	if _hw_phases_vbox == null:
		return
	# Just mark all as "running" - we'll update when done
	for i in range(_hw_phases_vbox.get_child_count()):
		var row = _hw_phases_vbox.get_child(i)
		if row is HBoxContainer and row.get_child_count() > 0:
			var dot = row.get_child(0)
			if dot is Label:
				dot.text = "[~]"
				dot.add_theme_color_override("font_color", Color(0.90, 0.75, 0.25))


func _populate_hw_phases(result: Dictionary) -> void:
	if _hw_phases_vbox == null:
		return
	for child in _hw_phases_vbox.get_children():
		child.queue_free()
	# After homework completes, mark all phases done (we ran synchronously)
	# Extract phase-level data from result if available
	var phases_done : int = result.get("batches_processed", 0)
	var all_done : bool   = phases_done > 0 or result.get("status","") == "completed"
	for i in range(6):
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 8)
		var dot := Label.new()
		dot.text = "[●]" if all_done else "[ ]"
		dot.add_theme_color_override("font_color",
			Color(0.25, 0.85, 0.45) if all_done else Color(0.35, 0.35, 0.40))
		dot.add_theme_font_size_override("font_size", 11)
		dot.custom_minimum_size = Vector2(24, 0)
		row.add_child(dot)
		var lbl := Label.new()
		lbl.text = PHASE_NAMES[i]
		lbl.add_theme_color_override("font_color",
			Color(0.75, 0.78, 0.82) if all_done else Color(0.45, 0.45, 0.50))
		lbl.add_theme_font_size_override("font_size", 11)
		row.add_child(lbl)
		_hw_phases_vbox.add_child(row)


func _populate_hw_summary(result: Dictionary) -> void:
	if _hw_summary_vbox == null:
		return
	for child in _hw_summary_vbox.get_children():
		child.queue_free()

	# Core summary fields
	var summary_fields := {}
	var field_map := [
		["batches_processed",       "Batches processed"],
		["lessons_validated",        "Lessons validated"],
		["lessons_pending",          "Lessons pending"],
		["contradictions_resolved",  "Contradictions resolved"],
		["contradictions_unresolved","Contradictions unresolved"],
		["questions_resolved",       "Questions resolved"],
		["questions_new",            "New questions added"],
		["dream_candidates_flagged", "Dream candidates flagged"],
		["core_memory_updates_applied","Core memory updates"],
	]
	for pair in field_map:
		if pair[0] in result:
			summary_fields[pair[1]] = str(result[pair[0]])

	if not summary_fields.is_empty():
		_hw_summary_vbox.add_child(_make_fields_card(summary_fields))
	else:
		var lbl := Label.new()
		lbl.text = "No summary data — check if learning log has entries first."
		lbl.add_theme_color_override("font_color", Color(0.5, 0.5, 0.5))
		lbl.add_theme_font_size_override("font_size", 11)
		lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		_hw_summary_vbox.add_child(lbl)

	# Refresh LTMM data after homework
	ZADOSClient.get_memory("ltmm/knowledge/lessons")
	ZADOSClient.get_memory("ltmm/knowledge/academic_buffer")
	ZADOSClient.get_memory("ltmm/unsolved")
	ZADOSClient.get_memory("ltmm/journal")


func _populate_reflective_result(result: Dictionary) -> void:
	if _ref_result_vbox == null:
		return
	for child in _ref_result_vbox.get_children():
		child.queue_free()

	# E31 meta-learning
	var e31_fields := {}
	var patterns : Array = result.get("learning_patterns", [])
	e31_fields["Learning patterns found"] = str(patterns.size())
	var failures : Array = result.get("recurring_failures", [])
	e31_fields["Recurring failures"]      = str(failures.size())
	var mode_eff : Dictionary = result.get("mode_effectiveness", {})
	e31_fields["Modes analyzed"]          = str(mode_eff.size())
	var recs : Array = result.get("learning_recommendations", [])
	e31_fields["Recommendations"]         = str(recs.size())

	# E32 identity
	var coherence : float = float(result.get("coherence_score", -1.0))
	if coherence >= 0.0:
		e31_fields["Identity coherence"]  = "%.2f" % coherence
	var contradictions : Array = result.get("core_contradictions", [])
	e31_fields["Core contradictions"]    = str(contradictions.size())

	# Mutations
	var conclusions_reinforced : int = result.get("conclusions_reinforced", 0)
	var conclusions_created    : int = result.get("conclusions_created",    0)
	var journal_entries        : int = result.get("journal_entries_created", 0)
	e31_fields["Conclusions reinforced"] = str(conclusions_reinforced)
	e31_fields["Conclusions created"]    = str(conclusions_created)
	e31_fields["Journal entries written"]= str(journal_entries)

	if not e31_fields.is_empty() and e31_fields.values().any(func(v): return v != "0"):
		_ref_result_vbox.add_child(_make_fields_card(e31_fields))
	else:
		var lbl := Label.new()
		lbl.text = "Synthesis completed. Check Memory workspace for LTMM updates."
		lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.80))
		lbl.add_theme_font_size_override("font_size", 11)
		lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		_ref_result_vbox.add_child(lbl)

	# Show top recommendation if any
	var recs2 : Array = result.get("learning_recommendations", [])
	if not recs2.is_empty():
		_ref_result_vbox.add_child(_section_lbl("Top Recommendations"))
		for i in range(mini(3, recs2.size())):
			_ref_result_vbox.add_child(_bullet_lbl("• " + str(recs2[i]), Color(0.80, 0.82, 0.50)))


# ---------------------------------------------------------------------------
# LTMM lesson / unsolved list updates
# ---------------------------------------------------------------------------

func _update_lessons_list(data: Dictionary) -> void:
	var lessons_list : VBoxContainer = _content.find_child("LessonsList", true, false)
	if lessons_list == null:
		return
	for child in lessons_list.get_children():
		child.queue_free()
	var items : Array = data.get("items", [])
	if items.is_empty():
		var lbl := Label.new()
		lbl.text = "No lessons yet. Complete a learning turn to generate lessons."
		lbl.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		lbl.add_theme_font_size_override("font_size", 11)
		lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		lessons_list.add_child(lbl)
		return
	# Show count + most recent 5
	var count_lbl := Label.new()
	count_lbl.text = "%d lesson(s) in LTMM" % items.size()
	count_lbl.add_theme_color_override("font_color", Color(0.30, 0.75, 0.55))
	count_lbl.add_theme_font_size_override("font_size", 11)
	lessons_list.add_child(count_lbl)
	var recent := items.slice(maxi(0, items.size() - 5))
	for i in range(recent.size() - 1, -1, -1):
		var l : Dictionary = recent[i]
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 8)
		var dot := Label.new()
		dot.text = "●"
		dot.add_theme_color_override("font_color", Color(0.30, 0.60, 0.80))
		dot.add_theme_font_size_override("font_size", 10)
		row.add_child(dot)
		var lbl := Label.new()
		lbl.text = (l.get("content","") as String).left(120)
		lbl.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
		lbl.add_theme_font_size_override("font_size", 11)
		lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		lbl.size_flags_horizontal = SIZE_EXPAND_FILL
		row.add_child(lbl)
		var subject := Label.new()
		subject.text = l.get("subject_category","")
		subject.add_theme_color_override("font_color", Color(0.50, 0.52, 0.56))
		subject.add_theme_font_size_override("font_size", 10)
		row.add_child(subject)
		lessons_list.add_child(row)


func _update_unsolved_count(data: Dictionary) -> void:
	var unsolved_lbl : Label = _content.find_child("UnsolvedLabel", true, false)
	if unsolved_lbl == null:
		return
	var items : Array = data.get("items", [])
	var dream_count : int = 0
	for item in items:
		if item.get("dream_candidate", false):
			dream_count += 1
	unsolved_lbl.text = "%d unsolved question(s)  (%d dream candidates)" % [items.size(), dream_count]


# ---------------------------------------------------------------------------
# LTMM links panel (Homework view)
# ---------------------------------------------------------------------------

func _ltmm_links() -> void:
	var grid := GridContainer.new()
	grid.columns = 3
	grid.add_theme_constant_override("h_separation", 8)
	grid.add_theme_constant_override("v_separation", 4)
	var link_defs := [
		["Lessons",       "ltmm/knowledge/lessons"],
		["Academic Buffer","ltmm/knowledge/academic_buffer"],
		["Unsolved",      "ltmm/unsolved"],
		["Journal",       "ltmm/journal"],
	]
	for ld in link_defs:
		var btn := Button.new()
		btn.text = "→ " + ld[0]
		btn.flat = true
		btn.focus_mode = Control.FOCUS_NONE
		btn.add_theme_font_size_override("font_size", 11)
		btn.add_theme_color_override("font_color", Color(0.40, 0.65, 0.90))
		var path : String = ld[1]
		btn.pressed.connect(func(): ZADOSClient.get_memory(path))
		grid.add_child(btn)
	_content.add_child(grid)


# ---------------------------------------------------------------------------
# Shared UI builders
# ---------------------------------------------------------------------------

func _header(info: Dictionary) -> void:
	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 4)
	_content.add_child(vbox)

	var title := Label.new()
	title.text = info.get("title","")
	title.add_theme_color_override("font_color", info.get("color", Color(0.7, 0.8, 1.0)))
	title.add_theme_font_size_override("font_size", 16)
	vbox.add_child(title)

	var desc := Label.new()
	desc.text = info.get("description","")
	desc.add_theme_color_override("font_color", Color(0.60, 0.63, 0.68))
	desc.add_theme_font_size_override("font_size", 11)
	desc.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	vbox.add_child(desc)

	var sep := HSeparator.new()
	sep.add_theme_color_override("color", Color(0.20, 0.20, 0.25))
	_content.add_child(sep)


func _section(title: String) -> void:
	var lbl := Label.new()
	lbl.text = title
	lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.80))
	lbl.add_theme_font_size_override("font_size", 11)
	_content.add_child(lbl)


func _section_lbl(title: String) -> Label:
	var lbl := Label.new()
	lbl.text = title
	lbl.add_theme_color_override("font_color", Color(0.55, 0.65, 0.80))
	lbl.add_theme_font_size_override("font_size", 11)
	return lbl


func _bullet(text: String, color: Color) -> void:
	_content.add_child(_bullet_lbl(text, color))


func _bullet_lbl(text: String, color: Color) -> Label:
	var lbl := Label.new()
	lbl.text = text
	lbl.add_theme_color_override("font_color", color)
	lbl.add_theme_font_size_override("font_size", 11)
	lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	return lbl


func _fields_card(data: Dictionary, bg: Color) -> PanelContainer:
	var panel := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = bg
	style.corner_radius_top_left = 4; style.corner_radius_top_right = 4
	style.corner_radius_bottom_left = 4; style.corner_radius_bottom_right = 4
	style.content_margin_left = 10; style.content_margin_right = 10
	style.content_margin_top = 6; style.content_margin_bottom = 6
	panel.add_theme_stylebox_override("panel", style)
	var grid := GridContainer.new()
	grid.columns = 2
	grid.add_theme_constant_override("h_separation", 16)
	grid.add_theme_constant_override("v_separation", 3)
	panel.add_child(grid)
	for k in data:
		var kl := Label.new()
		kl.text = str(k) + ":"
		kl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		kl.add_theme_font_size_override("font_size", 11)
		kl.custom_minimum_size = Vector2(160, 0)
		grid.add_child(kl)
		var vl := Label.new()
		vl.text = str(data[k])
		vl.add_theme_color_override("font_color", Color(0.85, 0.88, 0.92))
		vl.add_theme_font_size_override("font_size", 11)
		vl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		grid.add_child(vl)
	return panel


func _make_fields_card(data: Dictionary) -> PanelContainer:
	return _fields_card(data, Color(0.10, 0.10, 0.13))


func _badge(text: String, color: Color) -> PanelContainer:
	var bg := StyleBoxFlat.new()
	bg.bg_color = Color(color.r * 0.18, color.g * 0.18, color.b * 0.18)
	bg.corner_radius_top_left = 3; bg.corner_radius_top_right = 3
	bg.corner_radius_bottom_left = 3; bg.corner_radius_bottom_right = 3
	bg.content_margin_left = 6; bg.content_margin_right = 6
	bg.content_margin_top = 2; bg.content_margin_bottom = 2
	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", bg)
	var lbl := Label.new()
	lbl.text = text
	lbl.add_theme_color_override("font_color", color)
	lbl.add_theme_font_size_override("font_size", 10)
	panel.add_child(lbl)
	return panel

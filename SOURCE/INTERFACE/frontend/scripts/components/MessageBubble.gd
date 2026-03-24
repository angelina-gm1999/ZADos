##
## MessageBubble — a single message in the conversation history.
##
## Modes:
##   USER      — dark right-aligned bubble
##   AI        — left-aligned bubble with streaming support + detail tray
##   GENERATING — AI bubble in progress; shows phase progress bar
##                transitions automatically to streaming text on Phase 6 tokens
##
class_name MessageBubble
extends MarginContainer

enum Role { USER, AI, GENERATING }

# Phase labels shown during generation
const PHASE_NAMES := {
	1: "Perceive", 2: "Modulate", 3: "Dispatch",
	4: "Think",    5: "Reward",   6: "Respond"
}

# Color for ALLOW / SUPPRESS / ABSTAIN directive
const DIRECTIVE_COLORS := {
	"allow":    Color(0.20, 0.85, 0.40),
	"suppress": Color(0.90, 0.25, 0.25),
	"abstain":  Color(0.95, 0.75, 0.15),
}

var _role           : Role
var _body_label     : RichTextLabel    # main text / streaming target
var _phase_row      : HBoxContainer    # shown during GENERATING state
var _phase_badges   : Array[Label]     # one per phase 1-6
var _detail_toggle  : Button
var _detail_tray    : VBoxContainer
var _detail_data    : Dictionary = {}  # populated as phases complete

var _streaming_started : bool = false

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

## Initialise this bubble after .new(). Call _build_ui() via this method.
func initialize(role: Role, text: String = "") -> void:
	_role = role
	_build_ui()
	if not text.is_empty():
		_body_label.text = text

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

func _build_ui() -> void:
	# Outer margins
	add_theme_constant_override("margin_left",  8)
	add_theme_constant_override("margin_right", 8)
	add_theme_constant_override("margin_top",   4)
	add_theme_constant_override("margin_bottom",4)

	var outer := HBoxContainer.new()
	outer.size_flags_horizontal = SIZE_EXPAND_FILL
	add_child(outer)

	# ── Alignment spacer ────────────────────────────────────────────────
	if _role == Role.USER:
		var spacer := _make_spacer()
		outer.add_child(spacer)

	# ── Bubble body ─────────────────────────────────────────────────────
	var bubble := PanelContainer.new()
	bubble.size_flags_horizontal = SIZE_EXPAND_FILL if _role == Role.GENERATING else 0
	if _role == Role.USER:
		bubble.size_flags_horizontal = 0
		bubble.custom_minimum_size   = Vector2(120, 0)
	_style_bubble(bubble)

	var inner := VBoxContainer.new()
	bubble.add_child(inner)

	# Phase progress row (GENERATING only — hidden for USER/AI)
	_phase_row = HBoxContainer.new()
	_phase_row.visible = (_role == Role.GENERATING)
	_phase_badges.resize(7)   # index 1-6

	for i in range(1, 7):
		if i > 1:
			var arrow := Label.new()
			arrow.text = "→"
			arrow.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
			_phase_row.add_child(arrow)

		var badge := Label.new()
		badge.text = "[ ] %s" % PHASE_NAMES[i]
		badge.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		badge.add_theme_font_size_override("font_size", 11)
		_phase_row.add_child(badge)
		_phase_badges[i] = badge

	inner.add_child(_phase_row)

	# Main text label
	_body_label = RichTextLabel.new()
	_body_label.bbcode_enabled       = true
	_body_label.fit_content          = true
	_body_label.scroll_active        = false
	_body_label.autowrap_mode        = TextServer.AUTOWRAP_WORD_SMART
	_body_label.custom_minimum_size  = Vector2(80, 0)
	_body_label.size_flags_horizontal = SIZE_EXPAND_FILL
	_body_label.visible = (_role != Role.GENERATING)

	if _role == Role.USER:
		_body_label.add_theme_color_override("default_color", Color(0.95, 0.95, 0.95))
	else:
		_body_label.add_theme_color_override("default_color", Color(0.85, 0.88, 0.92))

	inner.add_child(_body_label)

	# Detail toggle (AI only) — sits below the text label
	if _role in [Role.AI, Role.GENERATING]:
		var footer := HBoxContainer.new()
		inner.add_child(footer)

		_detail_toggle = Button.new()
		_detail_toggle.text      = "▶  details"
		_detail_toggle.flat      = true
		_detail_toggle.focus_mode = Control.FOCUS_NONE
		_detail_toggle.add_theme_font_size_override("font_size", 10)
		_detail_toggle.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
		_detail_toggle.visible   = false
		_detail_toggle.pressed.connect(_toggle_detail_tray)
		footer.add_child(_detail_toggle)

		# Detail tray
		_detail_tray = VBoxContainer.new()
		_detail_tray.visible = false
		_detail_tray.add_theme_constant_override("separation", 2)
		inner.add_child(_detail_tray)

	outer.add_child(bubble)

	if _role == Role.AI or _role == Role.GENERATING:
		var spacer := _make_spacer()
		outer.add_child(spacer)


func _make_spacer() -> Control:
	var s := Control.new()
	s.custom_minimum_size    = Vector2(80, 0)
	s.size_flags_horizontal  = SIZE_EXPAND_FILL
	return s


func _style_bubble(panel: PanelContainer) -> void:
	var style := StyleBoxFlat.new()
	style.corner_radius_top_left     = 10
	style.corner_radius_top_right    = 10
	style.corner_radius_bottom_left  = 10
	style.corner_radius_bottom_right = 10
	style.content_margin_left   = 12
	style.content_margin_right  = 12
	style.content_margin_top    = 8
	style.content_margin_bottom = 8
	match _role:
		Role.USER:
			style.bg_color = Color(0.16, 0.22, 0.34)
		Role.AI, Role.GENERATING:
			style.bg_color = Color(0.13, 0.14, 0.17)
	panel.add_theme_stylebox_override("panel", style)

# ---------------------------------------------------------------------------
# Public API — called by ConversationWorkspace
# ---------------------------------------------------------------------------

## Mark a phase as complete and store its data.
func phase_done(phase: int, data: Dictionary) -> void:
	if phase < 1 or phase > 6:
		return
	var badge : Label = _phase_badges[phase]
	if badge == null:
		return
	badge.text = "[●] %s" % PHASE_NAMES[phase]
	badge.add_theme_color_override("font_color", Color(0.35, 0.75, 0.45))

	# Harvest detail tray data from relevant phases
	match phase:
		1:
			_detail_data["intent_archetype"] = data.get("intent_archetype", "—")
		2:
			_detail_data["selected_mode"]      = data.get("mode_token", "—")
			_detail_data["reward_profile_name"] = data.get("reward_profile_name", "—")
		3:
			# Dominant emotion from E28
			var e28 = data.get("e28_result", {})
			if e28 is Dictionary:
				var emotions : Dictionary = e28.get("emotion_vector", {})
				if not emotions.is_empty():
					var dominant = emotions.keys()[0]
					var max_val  = 0.0
					for k in emotions:
						if (emotions[k] as float) > max_val:
							max_val  = emotions[k]
							dominant = k
					_detail_data["dominant_emotion"] = dominant


## Append streaming text to the bubble body (Phase 6 tokens).
## First call transitions the bubble from phase-progress to text mode.
func append_token(text: String) -> void:
	if not _streaming_started:
		_streaming_started = true
		_phase_row.visible  = false
		_body_label.visible = true

	_body_label.append_text(text)


## Finalise the bubble once the turn completes.
func finalise(directive: String) -> void:
	_detail_data["directive"] = directive
	_phase_row.visible        = false
	_body_label.visible       = true

	if _detail_toggle != null:
		_detail_toggle.visible = true
	_build_detail_tray()


## Replace phase-progress state with a plain text body (non-streaming path).
func set_text(text: String) -> void:
	_phase_row.visible  = false
	_body_label.visible = true
	_body_label.text    = text

# ---------------------------------------------------------------------------
# Detail tray
# ---------------------------------------------------------------------------

func _build_detail_tray() -> void:
	if _detail_tray == null:
		return
	for child in _detail_tray.get_children():
		child.queue_free()

	var fields := [
		["Intent",     _detail_data.get("intent_archetype",   "—")],
		["Directive",  _detail_data.get("directive",          "—")],
		["Mode",       _detail_data.get("selected_mode",      "—")],
		["Profile",    _detail_data.get("reward_profile_name","—")],
		["Emotion",    _detail_data.get("dominant_emotion",   "—")],
	]

	for pair in fields:
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 6)
		_detail_tray.add_child(row)

		var key := Label.new()
		key.text = pair[0] + ":"
		key.custom_minimum_size = Vector2(68, 0)
		key.add_theme_font_size_override("font_size", 10)
		key.add_theme_color_override("font_color", Color(0.45, 0.45, 0.45))
		row.add_child(key)

		var val := Label.new()
		val.text = str(pair[1])
		val.add_theme_font_size_override("font_size", 10)
		# Colour the directive field
		if pair[0] == "Directive":
			val.add_theme_color_override(
				"font_color",
				DIRECTIVE_COLORS.get(str(pair[1]).to_lower(), Color(0.7, 0.7, 0.7))
			)
		else:
			val.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
		row.add_child(val)


func _toggle_detail_tray() -> void:
	if _detail_tray == null:
		return
	_detail_tray.visible = not _detail_tray.visible
	_detail_toggle.text  = ("▼  details" if _detail_tray.visible else "▶  details")

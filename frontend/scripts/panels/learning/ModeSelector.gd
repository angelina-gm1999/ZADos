##
## ModeSelector — left panel of the Learning Workspace.
## Radio-style button list for all ZADOS modes.
## Emits mode_selected(mode_key) when a mode is chosen.
##
## Addendum B.3.1: disable buttons while generation is in progress.
##
extends VBoxContainer

signal mode_selected(mode_key: String)

const MODES := [
	["Normal",         "Regular",            "Standard conversation mode"],
	["---", "", ""],
	["M1",             "M1 — Teach me",      "Teach me something  (receptive, deep encoding)"],
	["M2",             "M2 — Review",        "Review / quiz me  (critical, retroactive contrast)"],
	["M3",             "M3 — Explore",       "Explore / Socratic  (full dialectic, unlimited depth)"],
	["M4",             "M4 — Questions",     "Questions I have  (question-driven, unsolved buffer)"],
	["M5",             "M5 — Independent",   "Independent study  (autonomous, no response output)"],
	["---", "", ""],
	["Homework",       "Homework",           "Multi-step structured processing  (no user present)"],
	["Reflective",     "Reflective",         "Post-session synthesis  (E31 meta-learning + E32 identity)"],
	["---", "", ""],
	["SelfReflective", "Self-Reflective",    "Introspection on unsolved questions  (auto-activated)"],
]

var _active_key : String = "Normal"
var _buttons    : Dictionary = {}   # key → Button
var _group      : ButtonGroup

func _ready() -> void:
	add_theme_constant_override("separation", 2)
	_group = ButtonGroup.new()
	_build_ui()
	ZADOSClient.session_mode_set.connect(_on_mode_set)
	ZADOSClient.session_state_received.connect(_on_state)
	ZADOSClient.generation_started.connect(_on_generation_started)
	ZADOSClient.turn_complete.connect(_on_turn_complete)
	ZADOSClient.generation_cancelled.connect(_on_generation_ended)
	ZADOSClient.get_session_state()


func _build_ui() -> void:
	var header := Label.new()
	header.text = "Mode"
	header.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	add_child(header)

	var sep0 := HSeparator.new()
	sep0.add_theme_color_override("color", Color(0.2, 0.2, 0.25))
	add_child(sep0)

	for entry in MODES:
		var key : String    = entry[0]
		var label : String  = entry[1]
		var tooltip : String = entry[2]

		if key == "---":
			var sep := HSeparator.new()
			sep.add_theme_color_override("color", Color(0.18, 0.18, 0.22))
			add_child(sep)
			continue

		var btn := Button.new()
		btn.text            = label
		btn.tooltip_text    = tooltip
		btn.toggle_mode     = true
		btn.button_group    = _group
		btn.flat            = true
		btn.focus_mode      = Control.FOCUS_NONE
		btn.alignment       = HORIZONTAL_ALIGNMENT_LEFT
		btn.add_theme_font_size_override("font_size", 12)
		btn.add_theme_color_override("font_color",          Color(0.70, 0.72, 0.76))
		btn.add_theme_color_override("font_color_hover",    Color(0.90, 0.92, 0.96))
		btn.add_theme_color_override("font_color_pressed",  Color(0.30, 0.75, 0.55))
		btn.pressed.connect(func(): _on_btn_pressed(key))
		add_child(btn)
		_buttons[key] = btn

	_set_active("Normal")


func _on_btn_pressed(key: String) -> void:
	if key == _active_key:
		return
	_active_key = key
	ZADOSClient.set_session_mode(key)
	mode_selected.emit(key)


func _set_active(key: String) -> void:
	_active_key = key
	if key in _buttons:
		_buttons[key].button_pressed = true


func _on_mode_set(mode: String) -> void:
	_set_active(mode)


func _on_state(state: Dictionary) -> void:
	var mode : String = state.get("active_mode", "Normal")
	_set_active(mode)


## B.3.1 — disable all mode buttons while generation is running.
func _set_buttons_disabled(disabled: bool) -> void:
	for key in _buttons:
		(_buttons[key] as Button).disabled = disabled


func _on_generation_started() -> void:
	_set_buttons_disabled(true)


func _on_turn_complete(_r: Dictionary) -> void:
	_set_buttons_disabled(false)


func _on_generation_ended() -> void:
	_set_buttons_disabled(false)

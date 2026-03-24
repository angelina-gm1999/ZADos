extends HBoxContainer

## Emitted when the user clicks a nav button.
signal workspace_selected(key: String)

# Tab definitions: [key, display label]
const TABS := [
	["conversation", "Conversation"],
	["memory",       "Memory"],
	["learning",     "Learning"],
	["dev",          "Dev"],
	["map",          "Map"],
]

var _buttons : Dictionary = {}
var _active  : String     = ""

# ---------------------------------------------------------------------------

func _ready() -> void:
	_build_buttons()
	_build_status_area()
	# Select default without emitting (Main._ready handles the first switch).
	_set_active("conversation", false)


func _build_buttons() -> void:
	for entry in TABS:
		var key   : String = entry[0]
		var label : String = entry[1]

		var btn := Button.new()
		btn.text             = label
		btn.flat             = true
		btn.toggle_mode      = true
		btn.focus_mode       = Control.FOCUS_NONE
		btn.custom_minimum_size = Vector2(110, 0)
		btn.pressed.connect(func(): _select(key))
		add_child(btn)
		_buttons[key] = btn

	# Vertical separator before status area.
	var sep := VSeparator.new()
	sep.custom_minimum_size = Vector2(2, 0)
	add_child(sep)


func _build_status_area() -> void:
	# Flexible spacer keeps status right-aligned.
	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND
	add_child(spacer)

	# Session / mode badge — updated by Main via update_status().
	var lbl := Label.new()
	lbl.name              = "StatusLabel"
	lbl.text              = "—"
	lbl.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	lbl.add_theme_color_override("font_color", Color(0.6, 0.6, 0.6))
	add_child(lbl)

	var padding := Control.new()
	padding.custom_minimum_size = Vector2(12, 0)
	add_child(padding)


## Called by Main when session / turn data arrives.
func update_status(text: String) -> void:
	var lbl := get_node_or_null("StatusLabel")
	if lbl:
		lbl.text = text

# ---------------------------------------------------------------------------

func _select(key: String) -> void:
	_set_active(key, true)


func _set_active(key: String, emit: bool = true) -> void:
	if key == _active:
		return
	for k in _buttons:
		_buttons[k].button_pressed = (k == key)
	_active = key
	if emit:
		workspace_selected.emit(key)

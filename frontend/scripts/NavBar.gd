##
## NavBar — top navigation with workspace buttons, generation badges,
## and clickable session status indicator.
##
## Addendum A.5 (generation badges), A.8 (session status popover).
##
extends HBoxContainer

signal workspace_selected(key: String)

const TABS := [
	["conversation", "Conversation"],
	["memory",       "Memory"],
	["learning",     "Learning"],
	["dev",          "Dev"],
	["map",          "Map"],
]

const MODE_COLORS := {
	"Normal":          Color(0.55, 0.55, 0.55),
	"LearningMode":   Color(0.35, 0.55, 0.95),
	"SleepMode":      Color(0.35, 0.30, 0.70),
	"SleepMode_REM":  Color(0.35, 0.30, 0.70),
	"SleepMode_Dream":Color(0.55, 0.30, 0.70),
	"Dream":          Color(0.55, 0.30, 0.70),
	"Homework":       Color(0.85, 0.70, 0.20),
	"SelfReflective": Color(0.20, 0.70, 0.65),
	"Reflective":     Color(0.20, 0.65, 0.60),
}

var _buttons : Dictionary = {}        # key → Button
var _badges  : Dictionary = {}        # key → ColorRect (generation dot)
var _active  : String     = ""
var _status_btn : Button              # clickable session status
var _popover : Control = null         # session info popover

# Session data cache for popover
var _session_data : Dictionary = {}

# ---------------------------------------------------------------------------

func _ready() -> void:
	_build_buttons()
	_build_status_area()
	_set_active("conversation", false)

	ZADOSClient.session_opened.connect(_on_session_data)
	ZADOSClient.turn_complete.connect(func(r): _on_session_data(r.get("session", {})))
	ZADOSClient.mode_changed.connect(_on_mode_changed)


func _build_buttons() -> void:
	for entry in TABS:
		var key   : String = entry[0]
		var label : String = entry[1]

		# Container for button + badge
		var wrapper := Control.new()
		wrapper.custom_minimum_size = Vector2(110, 0)
		wrapper.size_flags_vertical = Control.SIZE_EXPAND_FILL

		var btn := Button.new()
		btn.text             = label
		btn.flat             = true
		btn.toggle_mode      = true
		btn.focus_mode       = Control.FOCUS_NONE
		btn.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
		btn.pressed.connect(func(): _select(key))
		wrapper.add_child(btn)
		_buttons[key] = btn

		# Generation badge (small animated dot, top-right corner)
		var badge := ColorRect.new()
		badge.custom_minimum_size = Vector2(8, 8)
		badge.size = Vector2(8, 8)
		badge.position = Vector2(96, 4)
		badge.color = Color(0.3, 0.75, 1.0)
		badge.visible = false
		wrapper.add_child(badge)
		_badges[key] = badge

		add_child(wrapper)

	# Vertical separator
	var sep := VSeparator.new()
	sep.custom_minimum_size = Vector2(2, 0)
	add_child(sep)


func _build_status_area() -> void:
	# Flexible spacer
	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND
	add_child(spacer)

	# Session status — clickable button styled as label
	_status_btn = Button.new()
	_status_btn.text = "—"
	_status_btn.flat = true
	_status_btn.focus_mode = Control.FOCUS_NONE
	_status_btn.add_theme_color_override("font_color", Color(0.6, 0.6, 0.6))
	_status_btn.add_theme_font_size_override("font_size", 12)
	_status_btn.pressed.connect(_toggle_popover)
	add_child(_status_btn)

	var padding := Control.new()
	padding.custom_minimum_size = Vector2(12, 0)
	add_child(padding)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

func update_status(text: String) -> void:
	if _status_btn:
		_status_btn.text = text


## Show animated generation badge on a workspace tab.
func set_badge(key: String, active: bool) -> void:
	if _badges.has(key):
		_badges[key].visible = active
		if active:
			_animate_badge(_badges[key])


## Clear the generation badge.
func clear_badge(key: String) -> void:
	if _badges.has(key):
		_badges[key].visible = false


func _animate_badge(badge: ColorRect) -> void:
	# Pulsing animation
	var tw := create_tween().set_loops()
	tw.tween_property(badge, "color:a", 0.3, 0.6)
	tw.tween_property(badge, "color:a", 1.0, 0.6)


# ---------------------------------------------------------------------------
# Session status popover  (addendum A.8)
# ---------------------------------------------------------------------------

func _on_session_data(data: Dictionary) -> void:
	_session_data = data
	var sid    : String = (data.get("session_id", "") as String).left(8)
	var branch : String = data.get("branch", "") as String
	var turns  : int    = data.get("turn_count", 0) as int
	var mode   : String = data.get("active_mode", data.get("initial_mode", "Normal")) as String

	# Update label with mode color
	var mode_color := _get_mode_color(mode)
	_status_btn.text = "Branch %s | Turn %d | %s" % [branch, turns, mode]
	_status_btn.add_theme_color_override("font_color", mode_color)


func _on_mode_changed(_old: String, new_mode: String) -> void:
	_status_btn.add_theme_color_override("font_color", _get_mode_color(new_mode))


func _get_mode_color(mode: String) -> Color:
	for key in MODE_COLORS:
		if mode.begins_with(key) or mode == key:
			return MODE_COLORS[key]
	return Color(0.55, 0.55, 0.55)


func _toggle_popover() -> void:
	if _popover:
		_popover.queue_free()
		_popover = null
		return
	_build_popover()


func _build_popover() -> void:
	_popover = PanelContainer.new()
	var sb := StyleBoxFlat.new()
	sb.bg_color = Color(0.12, 0.12, 0.15, 0.97)
	sb.border_color = Color(0.22, 0.22, 0.28)
	sb.border_width_top = 1
	sb.border_width_bottom = 1
	sb.border_width_left = 1
	sb.border_width_right = 1
	sb.corner_radius_top_left = 6
	sb.corner_radius_top_right = 6
	sb.corner_radius_bottom_left = 6
	sb.corner_radius_bottom_right = 6
	sb.content_margin_left = 16
	sb.content_margin_right = 16
	sb.content_margin_top = 14
	sb.content_margin_bottom = 14
	_popover.add_theme_stylebox_override("panel", sb)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 8)
	_popover.add_child(vbox)

	# Session ID (copyable)
	var sid := _session_data.get("session_id", "—")
	_add_popover_row(vbox, "Session ID", sid)
	_add_popover_row(vbox, "Branch", _session_data.get("branch", "—"))
	_add_popover_row(vbox, "Turns", str(_session_data.get("turn_count", 0)))
	_add_popover_row(vbox, "Mode", _session_data.get("active_mode", "Normal"))

	# Separator
	var sep := HSeparator.new()
	vbox.add_child(sep)

	# End Session button
	var end_btn := Button.new()
	end_btn.text = "End Session"
	end_btn.add_theme_color_override("font_color", Color(0.85, 0.4, 0.4))
	end_btn.add_theme_font_size_override("font_size", 13)
	end_btn.flat = true
	end_btn.pressed.connect(func():
		# TODO: wire to actual session end
		_popover.queue_free()
		_popover = null
	)
	vbox.add_child(end_btn)

	# New Branch button (disabled, future feature)
	var branch_btn := Button.new()
	branch_btn.text = "New Branch"
	branch_btn.disabled = true
	branch_btn.flat = true
	branch_btn.add_theme_font_size_override("font_size", 13)
	branch_btn.add_theme_color_override("font_color", Color(0.4, 0.4, 0.45))
	branch_btn.tooltip_text = "Coming soon"
	vbox.add_child(branch_btn)

	# Position below the status button
	_popover.position = Vector2(_status_btn.global_position.x - 100, 48)
	_popover.custom_minimum_size = Vector2(240, 0)

	# Add to tree above NavBar
	var root := get_tree().get_root().get_node_or_null("Main")
	if root:
		root.add_child(_popover)
	else:
		add_child(_popover)


func _add_popover_row(parent: VBoxContainer, label_text: String, value_text: String) -> void:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)

	var lbl := Label.new()
	lbl.text = label_text + ":"
	lbl.add_theme_font_size_override("font_size", 12)
	lbl.add_theme_color_override("font_color", Color(0.5, 0.5, 0.55))
	lbl.custom_minimum_size = Vector2(80, 0)
	row.add_child(lbl)

	var val := Label.new()
	val.text = value_text
	val.add_theme_font_size_override("font_size", 12)
	val.add_theme_color_override("font_color", Color(0.8, 0.82, 0.88))
	val.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
	row.add_child(val)

	parent.add_child(row)


# ---------------------------------------------------------------------------
# Workspace selection
# ---------------------------------------------------------------------------

func _select(key: String) -> void:
	_set_active(key, true)
	# Close popover on workspace switch
	if _popover:
		_popover.queue_free()
		_popover = null


func _set_active(key: String, emit: bool = true) -> void:
	if key == _active:
		return
	for k in _buttons:
		_buttons[k].button_pressed = (k == key)
	_active = key
	if emit:
		workspace_selected.emit(key)

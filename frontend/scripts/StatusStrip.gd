##
## StatusStrip — persistent bottom bar (NeurochemPulseStrip).
##
## Left side: session info (id, branch, turn count, mode).
## Right side: NT pulse — 5 colored segments with proportional width,
##   hover tooltips, click-to-expand (6px→40px), sleep dimming.
##
## Addendum A.7: full behavioural spec.
##
extends HBoxContainer

const NT_COLORS := {
	"da":   Color(0.94, 0.75, 0.13),   # gold
	"5ht":  Color(0.18, 0.75, 0.72),   # teal
	"ne":   Color(0.95, 0.52, 0.12),   # orange
	"gaba": Color(0.26, 0.55, 0.95),   # blue
	"glu":  Color(0.90, 0.25, 0.25),   # red
}

const NT_NAMES := {
	"da": "Dopamine", "5ht": "Serotonin", "ne": "Norepinephrine",
	"gaba": "GABA", "glu": "Glutamate",
}

const COLLAPSED_HEIGHT := 6.0
const EXPANDED_HEIGHT  := 40.0
const ANIM_SECONDS     := 0.3
const GLOW_SECONDS     := 0.5

var _session_label : Label
var _mode_label    : Label
var _strip_container : HBoxContainer
var _nt_bars       : Dictionary = {}   # nt_key → ColorRect
var _nt_values     : Dictionary = {}   # nt_key → float
var _nt_labels     : Dictionary = {}   # nt_key → Label (shown in expanded mode)
var _expanded      : bool = false
var _in_sleep      : bool = false

# ---------------------------------------------------------------------------

func _ready() -> void:
	custom_minimum_size = Vector2(0, 22)
	_build_ui()
	ZADOSClient.metrics_updated.connect(_on_metrics)
	ZADOSClient.turn_complete.connect(_on_turn_complete)
	ZADOSClient.sleep_activated.connect(func(_t): _set_sleep_mode(true))
	ZADOSClient.sleep_exited.connect(func(): _set_sleep_mode(false))


func _build_ui() -> void:
	# Left: session info.
	_session_label = Label.new()
	_session_label.text = "No session"
	_session_label.add_theme_color_override("font_color", Color(0.5, 0.5, 0.5))
	_session_label.add_theme_font_size_override("font_size", 11)
	_session_label.custom_minimum_size = Vector2(180, 0)
	add_child(_session_label)

	_mode_label = Label.new()
	_mode_label.text = ""
	_mode_label.add_theme_color_override("font_color", Color(0.8, 0.8, 0.4))
	_mode_label.add_theme_font_size_override("font_size", 11)
	add_child(_mode_label)

	# Flexible spacer.
	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND
	add_child(spacer)

	# Right: NT pulse strip label
	var strip_label := Label.new()
	strip_label.text = "NT  "
	strip_label.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
	strip_label.add_theme_font_size_override("font_size", 10)
	strip_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	add_child(strip_label)

	# Strip container (clickable for expand/collapse)
	_strip_container = HBoxContainer.new()
	_strip_container.add_theme_constant_override("separation", 1)
	_strip_container.custom_minimum_size = Vector2(200, 0)
	_strip_container.mouse_filter = Control.MOUSE_FILTER_STOP
	_strip_container.gui_input.connect(_on_strip_input)
	add_child(_strip_container)

	for nt_key in NT_COLORS:
		_nt_values[nt_key] = 0.5

		var wrapper := VBoxContainer.new()
		wrapper.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		wrapper.add_theme_constant_override("separation", 1)
		wrapper.tooltip_text = "%s: 0.50" % NT_NAMES.get(nt_key, nt_key)
		_strip_container.add_child(wrapper)

		var seg := ColorRect.new()
		seg.color = NT_COLORS[nt_key]
		seg.color.a = 0.35
		seg.custom_minimum_size = Vector2(0, COLLAPSED_HEIGHT)
		seg.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		seg.size_flags_vertical = Control.SIZE_SHRINK_END
		wrapper.add_child(seg)
		_nt_bars[nt_key] = seg

		# Label (for expanded mode, initially hidden)
		var lbl := Label.new()
		lbl.text = "0.50"
		lbl.add_theme_font_size_override("font_size", 9)
		lbl.add_theme_color_override("font_color", NT_COLORS[nt_key])
		lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		lbl.visible = false
		wrapper.add_child(lbl)
		_nt_labels[nt_key] = lbl

	var pad := Control.new()
	pad.custom_minimum_size = Vector2(8, 0)
	add_child(pad)


## Called by Main with session data from session_opened or turn_complete.
func refresh(data: Dictionary) -> void:
	var sid    : String = (data.get("session_id", "") as String).left(8)
	var branch : String = data.get("branch", "") as String
	var turns  : int    = data.get("turn_count", 0) as int
	var mode   : String = (data.get("active_mode", data.get("initial_mode", "")) as String)

	if _session_label:
		_session_label.text = "  %s  [%s]  turn %d" % [sid, branch, turns]
	if _mode_label:
		_mode_label.text = mode


func _on_turn_complete(result: Dictionary) -> void:
	refresh(result.get("session", {}))
	ZADOSClient.get_metrics()


func _on_metrics(metrics: Dictionary) -> void:
	var prev_values := _nt_values.duplicate()
	var max_change_key := ""
	var max_change_val := 0.0

	for nt_key in _nt_bars:
		var val: float = metrics.get(nt_key, 0.5)
		_nt_values[nt_key] = val
		var seg: ColorRect = _nt_bars[nt_key]

		# Proportional width based on NT level (A.7)
		var width_ratio := clampf(val, 0.1, 1.0)
		seg.custom_minimum_size.x = width_ratio * 50.0

		# Alpha based on value
		var target_alpha: float = lerp(0.15, 1.0, clampf(val, 0.0, 1.0))
		if _in_sleep:
			target_alpha *= 0.55   # Muted during sleep (A.7)

		var tween := create_tween()
		tween.tween_property(seg, "color:a", target_alpha, ANIM_SECONDS)

		# Update tooltip
		seg.get_parent().tooltip_text = "%s: %.2f" % [NT_NAMES.get(nt_key, nt_key), val]

		# Update expanded label
		if _nt_labels.has(nt_key):
			_nt_labels[nt_key].text = "%.2f" % val

		# Track most-changed segment for glow
		var delta := absf(val - prev_values.get(nt_key, 0.5))
		if delta > max_change_val:
			max_change_val = delta
			max_change_key = nt_key

	# Brief glow pulse on most-changed segment (A.7)
	if max_change_key != "" and max_change_val > 0.02:
		_glow_segment(max_change_key)


func _glow_segment(nt_key: String) -> void:
	if not _nt_bars.has(nt_key):
		return
	var seg : ColorRect = _nt_bars[nt_key]
	var original_color := seg.color
	var glow_color := NT_COLORS[nt_key]
	glow_color.a = 1.0
	var tw := create_tween()
	tw.tween_property(seg, "color", glow_color, GLOW_SECONDS * 0.3)
	tw.tween_property(seg, "color:a", original_color.a, GLOW_SECONDS * 0.7)


## Click strip to expand/collapse (A.7)
func _on_strip_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.pressed and mb.button_index == MOUSE_BUTTON_LEFT:
			_toggle_expanded()


func _toggle_expanded() -> void:
	_expanded = not _expanded
	var target_h := EXPANDED_HEIGHT if _expanded else COLLAPSED_HEIGHT
	for nt_key in _nt_bars:
		var seg : ColorRect = _nt_bars[nt_key]
		var tw := create_tween()
		tw.tween_property(seg, "custom_minimum_size:y", target_h, ANIM_SECONDS)
		_nt_labels[nt_key].visible = _expanded


## Sleep mode visual shift (A.7)
func _set_sleep_mode(active: bool) -> void:
	_in_sleep = active
	# Re-trigger metrics display with current values
	var fake_metrics := {}
	for nt_key in _nt_values:
		fake_metrics[nt_key] = _nt_values[nt_key]
	_on_metrics(fake_metrics)

##
## StatusStrip — persistent bottom bar.
##
## Left side: session info (id, branch, turn count, mode).
## Right side: NT pulse — 5 colored segments (DA/5HT/NE/GABA/GLU) that
##             animate softly each turn.  Values polled from /metrics.
##
extends HBoxContainer

# NT color map matching the spec.
const NT_COLORS := {
	"da":   Color(0.94, 0.75, 0.13),   # gold
	"5ht":  Color(0.18, 0.75, 0.72),   # teal
	"ne":   Color(0.95, 0.52, 0.12),   # orange
	"gaba": Color(0.26, 0.55, 0.95),   # blue
	"glu":  Color(0.90, 0.25, 0.25),   # red
}

var _session_label : Label
var _mode_label    : Label
var _nt_bars       : Dictionary = {}   # nt_key → ColorRect

# ---------------------------------------------------------------------------

func _ready() -> void:
	_build_ui()
	ZADOSClient.metrics_updated.connect(_on_metrics)
	ZADOSClient.turn_complete.connect(_on_turn_complete)


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

	# Right: NT pulse strip.
	var strip_label := Label.new()
	strip_label.text = "NT  "
	strip_label.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
	strip_label.add_theme_font_size_override("font_size", 10)
	strip_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	add_child(strip_label)

	for nt_key in NT_COLORS:
		var seg := ColorRect.new()
		seg.color = NT_COLORS[nt_key]
		seg.color.a = 0.35          # dim until data arrives
		seg.custom_minimum_size = Vector2(36, 0)
		seg.size_flags_vertical = Control.SIZE_EXPAND_FILL
		add_child(seg)
		_nt_bars[nt_key] = seg

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
	# Trigger a metrics poll after each turn to update the NT strip.
	ZADOSClient.get_metrics()


func _on_metrics(metrics: Dictionary) -> void:
	for nt_key in _nt_bars:
		var val: float = metrics.get(nt_key, 0.5)
		var seg: ColorRect = _nt_bars[nt_key]
		# Map 0.0–1.0 NT concentration to alpha 0.15–1.0.
		var target_alpha: float = lerp(0.15, 1.0, clampf(val, 0.0, 1.0))
		# Tween the alpha for a subtle pulse effect.
		var tween := create_tween()
		tween.tween_property(seg, "color:a", target_alpha, 0.4)

##
## Toast — auto-dismissing notification bar.
##
## Usage:
##   var t := Toast.new()
##   t.show_message("Saved!", Toast.Level.INFO)
##   parent.add_child(t)
##
## Levels control the left-edge accent colour.
## After `dismiss_seconds` the toast fades out and frees itself.
##
extends PanelContainer

enum Level { INFO, SUCCESS, WARNING, ERROR }

const LEVEL_COLORS := {
	Level.INFO:    Color(0.45, 0.65, 0.95),   # blue
	Level.SUCCESS: Color(0.25, 0.82, 0.45),   # green
	Level.WARNING: Color(0.95, 0.75, 0.15),   # amber
	Level.ERROR:   Color(0.92, 0.28, 0.28),   # red
}

const DISMISS_SECONDS := 8.0
const FADE_SECONDS    := 0.35

signal dismissed()
signal retry_requested()

var _label   : RichTextLabel
var _retry_btn : Button
var _dismiss_btn : Button
var _accent  : ColorRect
var _timer   : Timer
var _has_retry : bool = false

# ---------------------------------------------------------------------------

func show_message(text: String, level: int = Level.INFO, retryable: bool = false) -> void:
	_has_retry = retryable
	_build_ui(text, level)


func _build_ui(text: String, level: int) -> void:
	# Panel styling
	custom_minimum_size = Vector2(320, 0)
	size_flags_horizontal = Control.SIZE_SHRINK_END

	var sb := StyleBoxFlat.new()
	sb.bg_color = Color(0.12, 0.12, 0.15, 0.95)
	sb.border_color = LEVEL_COLORS.get(level, LEVEL_COLORS[Level.INFO])
	sb.border_width_left = 4
	sb.corner_radius_top_left = 4
	sb.corner_radius_top_right = 4
	sb.corner_radius_bottom_left = 4
	sb.corner_radius_bottom_right = 4
	sb.content_margin_left = 14
	sb.content_margin_right = 10
	sb.content_margin_top = 10
	sb.content_margin_bottom = 10
	add_theme_stylebox_override("panel", sb)

	var hbox := HBoxContainer.new()
	hbox.add_theme_constant_override("separation", 8)
	add_child(hbox)

	_label = RichTextLabel.new()
	_label.bbcode_enabled = true
	_label.fit_content = true
	_label.scroll_active = false
	_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_label.add_theme_color_override("default_color", Color(0.85, 0.87, 0.92))
	_label.add_theme_font_size_override("normal_font_size", 13)
	_label.text = text
	hbox.add_child(_label)

	if _has_retry:
		_retry_btn = Button.new()
		_retry_btn.text = "Retry"
		_retry_btn.flat = true
		_retry_btn.add_theme_color_override("font_color", Color(0.55, 0.75, 1.0))
		_retry_btn.add_theme_font_size_override("font_size", 12)
		_retry_btn.pressed.connect(func(): retry_requested.emit(); _fade_out())
		hbox.add_child(_retry_btn)

	_dismiss_btn = Button.new()
	_dismiss_btn.text = "✕"
	_dismiss_btn.flat = true
	_dismiss_btn.add_theme_color_override("font_color", Color(0.5, 0.5, 0.5))
	_dismiss_btn.add_theme_font_size_override("font_size", 12)
	_dismiss_btn.pressed.connect(_fade_out)
	hbox.add_child(_dismiss_btn)

	# Auto-dismiss timer.
	_timer = Timer.new()
	_timer.wait_time = DISMISS_SECONDS
	_timer.one_shot = true
	_timer.timeout.connect(_fade_out)
	add_child(_timer)
	_timer.start()


func _fade_out() -> void:
	if _timer and not _timer.is_stopped():
		_timer.stop()
	var tw := create_tween()
	tw.tween_property(self, "modulate:a", 0.0, FADE_SECONDS)
	tw.tween_callback(func():
		dismissed.emit()
		queue_free()
	)

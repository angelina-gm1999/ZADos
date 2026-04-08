##
## ErrorDisplay — inline error panel with retry button.
##
## Replaces the content of a parent panel when a data fetch fails.
##
## Usage:
##   var err := ErrorDisplay.new()
##   err.show_error("Memory Packets", "HTTP 500: Internal server error")
##   parent.add_child(err)
##   err.retry_pressed.connect(func(): parent.refresh())
##
extends VBoxContainer

signal retry_pressed()

var _icon_label : Label
var _title_label : Label
var _detail_label : Label
var _retry_btn : Button

# ---------------------------------------------------------------------------

func show_error(section_name: String, error_message: String = "", show_retry: bool = true) -> void:
	_build_ui(section_name, error_message, show_retry)


## Convenience for 404 / empty-state (not a real error).
func show_empty(section_name: String, hint: String = "No data available yet") -> void:
	_build_ui(section_name, hint, false, true)


func _build_ui(section_name: String, error_message: String, show_retry: bool, is_empty_state: bool = false) -> void:
	alignment = BoxContainer.ALIGNMENT_CENTER
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	size_flags_vertical = Control.SIZE_EXPAND_FILL
	add_theme_constant_override("separation", 8)

	_icon_label = Label.new()
	_icon_label.text = "○" if is_empty_state else "⚠"
	_icon_label.add_theme_font_size_override("font_size", 28)
	_icon_label.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50) if is_empty_state else Color(0.9, 0.55, 0.2))
	_icon_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	add_child(_icon_label)

	_title_label = Label.new()
	_title_label.text = ("No data yet" if is_empty_state else "Failed to load %s" % section_name)
	_title_label.add_theme_font_size_override("font_size", 14)
	_title_label.add_theme_color_override("font_color", Color(0.7, 0.72, 0.78))
	_title_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	add_child(_title_label)

	if not error_message.is_empty():
		_detail_label = Label.new()
		_detail_label.text = error_message
		_detail_label.add_theme_font_size_override("font_size", 12)
		_detail_label.add_theme_color_override("font_color", Color(0.5, 0.5, 0.55))
		_detail_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		_detail_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		add_child(_detail_label)

	if show_retry:
		_retry_btn = Button.new()
		_retry_btn.text = "Retry"
		_retry_btn.flat = true
		_retry_btn.add_theme_color_override("font_color", Color(0.5, 0.7, 1.0))
		_retry_btn.add_theme_font_size_override("font_size", 13)
		_retry_btn.pressed.connect(func(): retry_pressed.emit())
		# Center the button
		var center := HBoxContainer.new()
		center.alignment = BoxContainer.ALIGNMENT_CENTER
		center.add_child(_retry_btn)
		add_child(center)

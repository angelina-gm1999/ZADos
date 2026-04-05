##
## ConfirmationDialog — reusable centered modal with message + confirm/cancel.
##
## Usage:
##   var dlg := ConfirmationDialog.new()
##   dlg.show_dialog("Delete item?", "This cannot be undone.", "Delete", "Cancel")
##   parent.add_child(dlg)
##   var confirmed : bool = await dlg.result
##   # dlg has already freed itself at this point
##
## For dialogs that need extra input (e.g. resolution note), use
## `show_dialog_with_input()` which adds a TextEdit field.
##
extends Control

signal result(confirmed: bool)
signal result_with_text(confirmed: bool, text: String)

var _overlay   : ColorRect
var _panel     : PanelContainer
var _msg_label : Label
var _detail_label : RichTextLabel
var _input_field  : TextEdit = null
var _input_min_chars : int = 0
var _confirm_btn : Button
var _cancel_btn  : Button

# ---------------------------------------------------------------------------

func show_dialog(title: String, detail: String = "", confirm_text: String = "Confirm", cancel_text: String = "Cancel") -> void:
	_build_ui(title, detail, confirm_text, cancel_text, false, 0)


func show_dialog_with_input(title: String, detail: String = "", confirm_text: String = "Confirm", cancel_text: String = "Cancel", placeholder: String = "", min_chars: int = 0) -> void:
	_input_min_chars = min_chars
	_build_ui(title, detail, confirm_text, cancel_text, true, min_chars)
	if _input_field and not placeholder.is_empty():
		_input_field.placeholder_text = placeholder


func _build_ui(title: String, detail: String, confirm_text: String, cancel_text: String, with_input: bool, min_chars: int) -> void:
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	mouse_filter = Control.MOUSE_FILTER_STOP

	# Dim overlay
	_overlay = ColorRect.new()
	_overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_overlay.color = Color(0.0, 0.0, 0.0, 0.55)
	_overlay.mouse_filter = Control.MOUSE_FILTER_STOP
	add_child(_overlay)

	# Center panel
	_panel = PanelContainer.new()
	_panel.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
	_panel.custom_minimum_size = Vector2(420, 0)

	var sb := StyleBoxFlat.new()
	sb.bg_color = Color(0.13, 0.13, 0.16)
	sb.corner_radius_top_left = 8
	sb.corner_radius_top_right = 8
	sb.corner_radius_bottom_left = 8
	sb.corner_radius_bottom_right = 8
	sb.border_color = Color(0.22, 0.22, 0.28)
	sb.border_width_top = 1
	sb.border_width_bottom = 1
	sb.border_width_left = 1
	sb.border_width_right = 1
	sb.content_margin_left = 24
	sb.content_margin_right = 24
	sb.content_margin_top = 20
	sb.content_margin_bottom = 20
	_panel.add_theme_stylebox_override("panel", sb)
	add_child(_panel)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 14)
	_panel.add_child(vbox)

	# Title
	_msg_label = Label.new()
	_msg_label.text = title
	_msg_label.add_theme_font_size_override("font_size", 16)
	_msg_label.add_theme_color_override("font_color", Color(0.9, 0.9, 0.95))
	_msg_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	vbox.add_child(_msg_label)

	# Detail
	if not detail.is_empty():
		_detail_label = RichTextLabel.new()
		_detail_label.bbcode_enabled = true
		_detail_label.fit_content = true
		_detail_label.scroll_active = false
		_detail_label.add_theme_color_override("default_color", Color(0.6, 0.62, 0.68))
		_detail_label.add_theme_font_size_override("normal_font_size", 13)
		_detail_label.text = detail
		vbox.add_child(_detail_label)

	# Optional text input
	if with_input:
		_input_field = TextEdit.new()
		_input_field.custom_minimum_size = Vector2(0, 80)
		_input_field.add_theme_font_size_override("font_size", 13)
		_input_field.add_theme_color_override("font_color", Color(0.85, 0.87, 0.92))
		var input_sb := StyleBoxFlat.new()
		input_sb.bg_color = Color(0.09, 0.09, 0.11)
		input_sb.border_color = Color(0.25, 0.25, 0.30)
		input_sb.border_width_top = 1
		input_sb.border_width_bottom = 1
		input_sb.border_width_left = 1
		input_sb.border_width_right = 1
		input_sb.corner_radius_top_left = 4
		input_sb.corner_radius_top_right = 4
		input_sb.corner_radius_bottom_left = 4
		input_sb.corner_radius_bottom_right = 4
		input_sb.content_margin_left = 8
		input_sb.content_margin_right = 8
		input_sb.content_margin_top = 6
		input_sb.content_margin_bottom = 6
		_input_field.add_theme_stylebox_override("normal", input_sb)
		vbox.add_child(_input_field)

		if min_chars > 0:
			var hint := Label.new()
			hint.text = "Minimum %d characters required" % min_chars
			hint.add_theme_font_size_override("font_size", 11)
			hint.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
			vbox.add_child(hint)

	# Button row
	var btn_row := HBoxContainer.new()
	btn_row.add_theme_constant_override("separation", 10)
	btn_row.alignment = BoxContainer.ALIGNMENT_END
	vbox.add_child(btn_row)

	_cancel_btn = Button.new()
	_cancel_btn.text = cancel_text
	_cancel_btn.flat = true
	_cancel_btn.add_theme_color_override("font_color", Color(0.55, 0.55, 0.60))
	_cancel_btn.add_theme_font_size_override("font_size", 14)
	_cancel_btn.pressed.connect(_on_cancel)
	btn_row.add_child(_cancel_btn)

	_confirm_btn = Button.new()
	_confirm_btn.text = confirm_text
	_confirm_btn.add_theme_font_size_override("font_size", 14)
	var btn_sb := StyleBoxFlat.new()
	btn_sb.bg_color = Color(0.22, 0.45, 0.85)
	btn_sb.corner_radius_top_left = 4
	btn_sb.corner_radius_top_right = 4
	btn_sb.corner_radius_bottom_left = 4
	btn_sb.corner_radius_bottom_right = 4
	btn_sb.content_margin_left = 16
	btn_sb.content_margin_right = 16
	btn_sb.content_margin_top = 6
	btn_sb.content_margin_bottom = 6
	_confirm_btn.add_theme_stylebox_override("normal", btn_sb)
	_confirm_btn.pressed.connect(_on_confirm)
	btn_row.add_child(_confirm_btn)

	# Center the panel properly
	_panel.anchor_left = 0.5
	_panel.anchor_top = 0.5
	_panel.anchor_right = 0.5
	_panel.anchor_bottom = 0.5
	_panel.grow_horizontal = Control.GROW_DIRECTION_BOTH
	_panel.grow_vertical = Control.GROW_DIRECTION_BOTH


func _on_confirm() -> void:
	if _input_field != null:
		var txt := _input_field.text.strip_edges()
		if _input_min_chars > 0 and txt.length() < _input_min_chars:
			# Flash the input border red briefly
			return
		result_with_text.emit(true, txt)
	result.emit(true)
	queue_free()


func _on_cancel() -> void:
	if _input_field != null:
		result_with_text.emit(false, "")
	result.emit(false)
	queue_free()


func _unhandled_key_input(event: InputEvent) -> void:
	if event is InputEventKey:
		var key := event as InputEventKey
		if key.pressed and key.keycode == KEY_ESCAPE:
			_on_cancel()
			get_viewport().set_input_as_handled()

##
## SleepDreamPanel — NT state, sleep phase, and sleep trigger.
##
## Addendum B.4.2: confirmation dialog before sleep activation.
##
extends VBoxContainer

const _ConfirmDialog = preload("res://scripts/components/ConfirmationDialog.gd")
const _Toast = preload("res://scripts/components/Toast.gd")
const _ErrorDisplay = preload("res://scripts/components/ErrorDisplay.gd")

const _PHASE_COLORS := {
	"WAKING":         Color(0.4, 0.8, 0.4),
	"TRIAGE":         Color(0.8, 0.8, 0.3),
	"REM_PROCESSING": Color(0.3, 0.6, 1.0),
	"DREAM":          Color(0.65, 0.35, 0.85),
}

const _NT_NAMES := ["DA", "NE", "5HT", "ACh", "GABA", "GLU", "His",
					"cortisol", "CRH", "OXT"]

var _phase_lbl  : Label
var _nt_grid    : VBoxContainer
var _raw_grid   : GridContainer
var _trigger_btn: Button

func _ready() -> void:
	add_theme_constant_override("separation", 6)
	add_child(_make_header())
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical    = SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	add_child(scroll)
	var inner := VBoxContainer.new()
	inner.size_flags_horizontal = SIZE_EXPAND_FILL
	inner.add_theme_constant_override("separation", 10)
	scroll.add_child(inner)
	_build_body(inner)
	ZADOSClient.dev_data_received.connect(_on_dev_data)
	ZADOSClient.sleep_triggered.connect(_on_sleep_triggered)
	ZADOSClient.turn_complete.connect(func(_r): refresh())
	ZADOSClient.request_failed.connect(_on_request_failed)


func refresh() -> void:
	ZADOSClient.get_dev("neurochem")


func _on_dev_data(key: String, data: Dictionary) -> void:
	if key != "neurochem":
		return
	_populate(data)


func _on_request_failed(path: String, error: Dictionary) -> void:
	if "/dev/neurochem" not in path:
		return
	_show_toast("Failed to load neurochem data: HTTP %s" % str(error.get("http_code", "?")),
		_Toast.Level.ERROR)


func _on_sleep_triggered(_result: Dictionary) -> void:
	_trigger_btn.text       = "⚡ Trigger Sleep Cycle"
	_trigger_btn.disabled   = false
	_show_toast("Sleep cycle activated", _Toast.Level.SUCCESS)
	refresh()


func _build_body(parent: VBoxContainer) -> void:
	# Phase badge row
	var phase_row := HBoxContainer.new()
	phase_row.add_theme_constant_override("separation", 8)
	parent.add_child(phase_row)
	var phase_title := Label.new()
	phase_title.text = "Estimated Phase:"
	phase_title.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
	phase_title.add_theme_font_size_override("font_size", 11)
	phase_row.add_child(phase_title)
	_phase_lbl = Label.new()
	_phase_lbl.text = "WAKING"
	_phase_lbl.add_theme_color_override("font_color", Color(0.4, 0.8, 0.4))
	_phase_lbl.add_theme_font_size_override("font_size", 11)
	phase_row.add_child(_phase_lbl)

	# NT bars
	var nt_title := Label.new()
	nt_title.text = "Neurotransmitter Concentrations"
	nt_title.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	nt_title.add_theme_font_size_override("font_size", 11)
	parent.add_child(nt_title)
	_nt_grid = VBoxContainer.new()
	_nt_grid.add_theme_constant_override("separation", 4)
	parent.add_child(_nt_grid)

	# Other readout fields
	var other_title := Label.new()
	other_title.text = "Full Readout"
	other_title.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	other_title.add_theme_font_size_override("font_size", 11)
	parent.add_child(other_title)
	_raw_grid = GridContainer.new()
	_raw_grid.columns = 2
	_raw_grid.add_theme_constant_override("h_separation", 12)
	_raw_grid.add_theme_constant_override("v_separation", 2)
	parent.add_child(_raw_grid)

	# Sleep trigger button
	var sep := HSeparator.new()
	parent.add_child(sep)
	_trigger_btn = Button.new()
	_trigger_btn.text = "⚡ Trigger Sleep Cycle"
	_trigger_btn.focus_mode = Control.FOCUS_NONE
	_trigger_btn.add_theme_font_size_override("font_size", 12)
	_trigger_btn.pressed.connect(_on_trigger_pressed)
	parent.add_child(_trigger_btn)
	var hint := Label.new()
	hint.text = "Runs /sleep through InputClassifier — activates TRIAGE → REM → DREAM processing."
	hint.add_theme_color_override("font_color", Color(0.4, 0.4, 0.4))
	hint.add_theme_font_size_override("font_size", 10)
	hint.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	parent.add_child(hint)

	# Open full sleep overlay button
	var overlay_btn := Button.new()
	overlay_btn.text = "Open Sleep Overlay"
	overlay_btn.focus_mode = Control.FOCUS_NONE
	overlay_btn.add_theme_font_size_override("font_size", 11)
	overlay_btn.add_theme_color_override("font_color", Color(0.55, 0.45, 0.85))
	overlay_btn.pressed.connect(_open_sleep_overlay)
	parent.add_child(overlay_btn)


## B.4.2 — confirmation dialog before triggering sleep.
func _on_trigger_pressed() -> void:
	var dlg := _ConfirmDialog.new()
	dlg.show_dialog(
		"Trigger Sleep Cycle?",
		"This will activate TRIAGE → REM → DREAM processing.\nThe system will be unavailable for normal conversation during sleep.",
		"Activate Sleep", "Cancel")
	var main := get_tree().get_root().get_node_or_null("Main/ModalContainer")
	if main:
		main.add_child(dlg)
	else:
		add_child(dlg)
	dlg.result.connect(func(confirmed: bool):
		if confirmed:
			_trigger_btn.text     = "Running…"
			_trigger_btn.disabled = true
			ZADOSClient.trigger_sleep()
	)


func _populate(d: Dictionary) -> void:
	# Sleep phase
	var phase : String = str(d.get("_sleep_phase", "WAKING"))
	_phase_lbl.text = phase
	_phase_lbl.add_theme_color_override("font_color",
		_PHASE_COLORS.get(phase, Color(0.75, 0.78, 0.82)))

	# Clear NT grid
	for child in _nt_grid.get_children():
		child.queue_free()

	# Try to render known NT names as bars
	var rendered_keys : Array = []
	for nt_name in _NT_NAMES:
		var val = d.get(nt_name, null)
		if val == null:
			continue
		var fval : float = _extract_float(val)
		rendered_keys.append(nt_name)
		var row := HBoxContainer.new()
		row.add_theme_constant_override("separation", 6)
		_nt_grid.add_child(row)
		var lbl := Label.new()
		lbl.text = nt_name
		lbl.custom_minimum_size = Vector2(60, 0)
		lbl.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))
		lbl.add_theme_font_size_override("font_size", 10)
		row.add_child(lbl)
		var bar := ProgressBar.new()
		bar.min_value = 0.0
		bar.max_value = 1.0
		bar.value     = clampf(fval, 0.0, 1.0)
		bar.size_flags_horizontal = SIZE_EXPAND_FILL
		bar.custom_minimum_size = Vector2(0, 13)
		row.add_child(bar)
		var val_lbl := Label.new()
		val_lbl.text = "%.3f" % fval
		val_lbl.add_theme_color_override("font_color", Color(0.50, 0.53, 0.57))
		val_lbl.add_theme_font_size_override("font_size", 10)
		val_lbl.custom_minimum_size = Vector2(38, 0)
		row.add_child(val_lbl)

	# Remaining fields as key-value table
	for child in _raw_grid.get_children():
		child.queue_free()
	for k in d:
		if k in rendered_keys or k == "_sleep_phase":
			continue
		var kl := Label.new()
		kl.text = str(k) + ":"
		kl.add_theme_color_override("font_color", Color(0.45, 0.45, 0.50))
		kl.add_theme_font_size_override("font_size", 10)
		kl.custom_minimum_size = Vector2(90, 0)
		_raw_grid.add_child(kl)
		var vl := Label.new()
		vl.text = (str(d[k])).left(120)
		vl.add_theme_color_override("font_color", Color(0.70, 0.73, 0.77))
		vl.add_theme_font_size_override("font_size", 10)
		vl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		vl.size_flags_horizontal = SIZE_EXPAND_FILL
		_raw_grid.add_child(vl)


func _extract_float(val) -> float:
	if val is float or val is int:
		return float(val)
	if val is Dictionary:
		for k in ["tonic", "level", "value", "concentration"]:
			if val.has(k):
				return float(val[k])
	return 0.0


func _open_sleep_overlay() -> void:
	var main := get_tree().get_root().get_node_or_null("Main")
	if main and main.has_method("show_sleep_overlay"):
		main.show_sleep_overlay()


func _show_toast(text: String, level: int) -> void:
	var tc = get_tree().get_root().get_node_or_null("Main/ToastContainer")
	if tc:
		tc.show_toast(text, level)


func _make_header() -> HBoxContainer:
	var hbox := HBoxContainer.new()
	var lbl  := Label.new()
	lbl.text = "Sleep / Dream"
	lbl.size_flags_horizontal = SIZE_EXPAND_FILL
	lbl.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	hbox.add_child(lbl)
	var btn := Button.new()
	btn.text = "↺ Refresh"
	btn.flat = true
	btn.focus_mode = Control.FOCUS_NONE
	btn.add_theme_font_size_override("font_size", 11)
	btn.pressed.connect(refresh)
	hbox.add_child(btn)
	return hbox

##
## PlumbingTestPanel — Runs server-side plumbing diagnostics.
## Verifies pipeline phase flow, memory tier I/O, NT→dispatch influence,
## and consolidation pathways.
##
extends VBoxContainer

var _run_btn    : Button
var _status_lbl : Label
var _summary_lbl: Label
var _results_box: VBoxContainer
var _scroll     : ScrollContainer


func _ready() -> void:
	add_theme_constant_override("separation", 6)
	_build_ui()
	ZADOSClient.dev_data_received.connect(_on_dev_data)
	ZADOSClient.request_failed.connect(func(path: String, error: Dictionary):
		if "/dev/plumbing" in path:
			_run_btn.disabled = false
			_status_lbl.text = "Error: HTTP %s" % str(error.get("http_code", "?"))
			_status_lbl.add_theme_color_override("font_color", Color(0.90, 0.40, 0.35)))


func refresh() -> void:
	pass   # no auto-refresh — user clicks the button


func _build_ui() -> void:
	# Header row
	var header := HBoxContainer.new()
	header.add_theme_constant_override("separation", 12)
	add_child(header)

	var title := Label.new()
	title.text = "Plumbing Diagnostics"
	title.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	title.add_theme_font_size_override("font_size", 14)
	header.add_child(title)

	_run_btn = Button.new()
	_run_btn.text = "Run Tests"
	_run_btn.pressed.connect(_on_run_pressed)
	header.add_child(_run_btn)

	_status_lbl = Label.new()
	_status_lbl.text = "Ready"
	_status_lbl.add_theme_font_size_override("font_size", 11)
	_status_lbl.add_theme_color_override("font_color", Color(0.6, 0.6, 0.6))
	header.add_child(_status_lbl)

	# Summary line
	_summary_lbl = Label.new()
	_summary_lbl.text = ""
	_summary_lbl.add_theme_font_size_override("font_size", 12)
	add_child(_summary_lbl)

	# Scroll + results
	_scroll = ScrollContainer.new()
	_scroll.size_flags_vertical = SIZE_EXPAND_FILL
	_scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	add_child(_scroll)

	_results_box = VBoxContainer.new()
	_results_box.size_flags_horizontal = SIZE_EXPAND_FILL
	_results_box.add_theme_constant_override("separation", 2)
	_scroll.add_child(_results_box)


func _on_run_pressed() -> void:
	_run_btn.disabled = true
	_status_lbl.text = "Running..."
	_status_lbl.add_theme_color_override("font_color", Color(1.0, 0.85, 0.3))
	_clear_results()
	ZADOSClient.post_dev("plumbing", {})


func _on_dev_data(key: String, data: Dictionary) -> void:
	if key != "plumbing/result":
		return
	_run_btn.disabled = false
	_populate(data)


func _populate(data: Dictionary) -> void:
	_clear_results()

	var all_passed: bool = data.get("all_passed", false)
	var passed: int      = data.get("passed", 0)
	var failed: int      = data.get("failed", 0)
	var errors: int      = data.get("errors", 0)
	var total_ms: float  = data.get("total_ms", 0.0)

	# Summary
	if all_passed:
		_summary_lbl.text = "ALL PASSED  (%d tests, %.0f ms)" % [passed, total_ms]
		_summary_lbl.add_theme_color_override("font_color", Color(0.3, 1.0, 0.3))
		_status_lbl.text = "Done"
		_status_lbl.add_theme_color_override("font_color", Color(0.3, 1.0, 0.3))
	else:
		_summary_lbl.text = "%d PASSED / %d FAILED / %d ERRORS  (%.0f ms)" % [
			passed, failed, errors, total_ms]
		_summary_lbl.add_theme_color_override("font_color", Color(1.0, 0.3, 0.3))
		_status_lbl.text = "Issues found"
		_status_lbl.add_theme_color_override("font_color", Color(1.0, 0.3, 0.3))

	# Per-test rows
	var results: Array = data.get("results", [])
	for r in results:
		_add_result_row(r)


func _add_result_row(r: Dictionary) -> void:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	_results_box.add_child(row)

	# Status icon
	var icon := Label.new()
	icon.add_theme_font_size_override("font_size", 12)
	if r.get("passed", false):
		icon.text = "PASS"
		icon.add_theme_color_override("font_color", Color(0.3, 1.0, 0.3))
	else:
		icon.text = "FAIL"
		icon.add_theme_color_override("font_color", Color(1.0, 0.3, 0.3))
	icon.custom_minimum_size.x = 40
	row.add_child(icon)

	# Test name
	var name_lbl := Label.new()
	name_lbl.text = r.get("name", "?")
	name_lbl.add_theme_font_size_override("font_size", 11)
	name_lbl.custom_minimum_size.x = 260
	row.add_child(name_lbl)

	# Elapsed
	var ms_lbl := Label.new()
	ms_lbl.text = "%.1f ms" % r.get("elapsed_ms", 0.0)
	ms_lbl.add_theme_font_size_override("font_size", 10)
	ms_lbl.add_theme_color_override("font_color", Color(0.5, 0.5, 0.5))
	ms_lbl.custom_minimum_size.x = 60
	row.add_child(ms_lbl)

	# Message (shown only on failure)
	if not r.get("passed", false):
		var msg_lbl := Label.new()
		msg_lbl.text = r.get("message", "")
		msg_lbl.add_theme_font_size_override("font_size", 10)
		msg_lbl.add_theme_color_override("font_color", Color(1.0, 0.5, 0.4))
		msg_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		msg_lbl.size_flags_horizontal = SIZE_EXPAND_FILL
		row.add_child(msg_lbl)


func _clear_results() -> void:
	for c in _results_box.get_children():
		c.queue_free()

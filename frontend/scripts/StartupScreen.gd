##
## StartupScreen — shown on launch while connecting to the ZADOS server.
##
## Displays:
##   - ZADOS branding / splash
##   - Connection status with retry
##   - Bootstrap progress (if knowledge seeds are loading)
##   - Error states with retry / settings buttons
##
## Addendum A.1: full startup & initialisation sequence.
##
extends Control

signal startup_complete()
signal settings_requested()

const _Toast = preload("res://scripts/components/Toast.gd")

var _status_label  : Label
var _progress_bar  : ProgressBar
var _progress_label: Label
var _retry_btn     : Button
var _settings_btn  : Button
var _error_detail  : RichTextLabel
var _countdown_label : Label

var _auto_retry_timer : Timer
var _auto_retry_sec   : float = 10.0
var _auto_retry_remaining : float = 0.0

# ---------------------------------------------------------------------------

func _ready() -> void:
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_build_ui()
	_run_startup()


func _build_ui() -> void:
	var bg := ColorRect.new()
	bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	bg.color = Color(0.08, 0.08, 0.10)
	add_child(bg)

	var center := VBoxContainer.new()
	center.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
	center.anchor_left = 0.5; center.anchor_top = 0.5
	center.anchor_right = 0.5; center.anchor_bottom = 0.5
	center.grow_horizontal = Control.GROW_DIRECTION_BOTH
	center.grow_vertical = Control.GROW_DIRECTION_BOTH
	center.custom_minimum_size = Vector2(480, 0)
	center.add_theme_constant_override("separation", 18)
	center.alignment = BoxContainer.ALIGNMENT_CENTER
	add_child(center)

	# Branding
	var title := Label.new()
	title.text = "ZADOS"
	title.add_theme_font_size_override("font_size", 36)
	title.add_theme_color_override("font_color", Color(0.85, 0.87, 0.92))
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	center.add_child(title)

	var subtitle := Label.new()
	subtitle.text = "Zonal Adaptive Dynamics Operating System"
	subtitle.add_theme_font_size_override("font_size", 13)
	subtitle.add_theme_color_override("font_color", Color(0.45, 0.48, 0.55))
	subtitle.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	center.add_child(subtitle)

	# Spacer
	var spacer := Control.new()
	spacer.custom_minimum_size = Vector2(0, 20)
	center.add_child(spacer)

	# Status
	_status_label = Label.new()
	_status_label.text = "Connecting..."
	_status_label.add_theme_font_size_override("font_size", 14)
	_status_label.add_theme_color_override("font_color", Color(0.65, 0.68, 0.75))
	_status_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	center.add_child(_status_label)

	# Progress bar (for bootstrap)
	_progress_bar = ProgressBar.new()
	_progress_bar.custom_minimum_size = Vector2(300, 8)
	_progress_bar.max_value = 100.0
	_progress_bar.value = 0.0
	_progress_bar.visible = false
	var bar_sb := StyleBoxFlat.new()
	bar_sb.bg_color = Color(0.15, 0.15, 0.20)
	bar_sb.corner_radius_top_left = 4
	bar_sb.corner_radius_top_right = 4
	bar_sb.corner_radius_bottom_left = 4
	bar_sb.corner_radius_bottom_right = 4
	_progress_bar.add_theme_stylebox_override("background", bar_sb)
	center.add_child(_progress_bar)

	_progress_label = Label.new()
	_progress_label.text = ""
	_progress_label.add_theme_font_size_override("font_size", 12)
	_progress_label.add_theme_color_override("font_color", Color(0.5, 0.52, 0.58))
	_progress_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_progress_label.visible = false
	center.add_child(_progress_label)

	# Error detail (initially hidden)
	_error_detail = RichTextLabel.new()
	_error_detail.bbcode_enabled = true
	_error_detail.fit_content = true
	_error_detail.scroll_active = false
	_error_detail.custom_minimum_size = Vector2(400, 0)
	_error_detail.add_theme_color_override("default_color", Color(0.55, 0.45, 0.45))
	_error_detail.add_theme_font_size_override("normal_font_size", 12)
	_error_detail.visible = false
	center.add_child(_error_detail)

	# Countdown label
	_countdown_label = Label.new()
	_countdown_label.text = ""
	_countdown_label.add_theme_font_size_override("font_size", 11)
	_countdown_label.add_theme_color_override("font_color", Color(0.4, 0.42, 0.48))
	_countdown_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_countdown_label.visible = false
	center.add_child(_countdown_label)

	# Button row
	var btn_row := HBoxContainer.new()
	btn_row.add_theme_constant_override("separation", 12)
	btn_row.alignment = BoxContainer.ALIGNMENT_CENTER
	center.add_child(btn_row)

	_retry_btn = Button.new()
	_retry_btn.text = "Retry"
	_retry_btn.visible = false
	_retry_btn.pressed.connect(func():
		_retry_btn.visible = false
		_settings_btn.visible = false
		_error_detail.visible = false
		_countdown_label.visible = false
		_run_startup()
	)
	btn_row.add_child(_retry_btn)

	_settings_btn = Button.new()
	_settings_btn.text = "Settings"
	_settings_btn.flat = true
	_settings_btn.visible = false
	_settings_btn.add_theme_color_override("font_color", Color(0.5, 0.6, 0.8))
	_settings_btn.pressed.connect(func(): settings_requested.emit())
	btn_row.add_child(_settings_btn)

	# Auto-retry timer
	_auto_retry_timer = Timer.new()
	_auto_retry_timer.wait_time = 1.0
	_auto_retry_timer.timeout.connect(_on_auto_retry_tick)
	add_child(_auto_retry_timer)


# ---------------------------------------------------------------------------
# Startup sequence
# ---------------------------------------------------------------------------

func _run_startup() -> void:
	_status_label.text = "Connecting to server..."

	# Step 2: Health check
	var health := await ZADOSClient.check_health(3, 2.0)
	if health.is_empty():
		_show_connection_error("ZADOS server is not running at %s" % ZADOSClient.BASE_URL)
		return

	# Step 4: Open session
	_status_label.text = "Opening session..."
	await ZADOSClient.open_session()
	if ZADOSClient.session_id.is_empty():
		_show_error("Session Creation Failed",
			"Could not create a new session. Check server logs.")
		return

	# Step 7: Check bootstrap status
	var bootstrap := await ZADOSClient.poll_bootstrap_status()
	var bs_status := bootstrap.get("status", "skipped") as String
	if bs_status == "running":
		_status_label.text = "Loading knowledge seeds..."
		_progress_bar.visible = true
		_progress_label.visible = true
		while bs_status == "running":
			var progress := bootstrap.get("progress", {}) as Dictionary
			var current_n : int = progress.get("current", 0)
			var total_n   : int = progress.get("total", 1)
			_progress_bar.value = (float(current_n) / float(max(total_n, 1))) * 100.0
			_progress_label.text = "Loading knowledge seeds... (%d/%d)" % [current_n, total_n]
			await get_tree().create_timer(2.0).timeout
			bootstrap = await ZADOSClient.poll_bootstrap_status()
			bs_status = bootstrap.get("status", "complete") as String
		_progress_bar.visible = false
		_progress_label.visible = false

	# Step 5: Fetch initial state
	_status_label.text = "Loading initial state..."
	ZADOSClient.get_session_state()
	ZADOSClient.get_metrics()

	# Brief pause for the fetches to land
	await get_tree().create_timer(0.5).timeout

	# Step 6: Finalise
	_status_label.text = "Ready"
	ZADOSClient.finalise_startup()
	startup_complete.emit()


func _show_connection_error(message: String) -> void:
	_status_label.text = "Server Unavailable"
	_status_label.add_theme_color_override("font_color", Color(0.9, 0.55, 0.35))
	_error_detail.text = message
	_error_detail.visible = true
	_retry_btn.visible = true
	_settings_btn.visible = true

	# Start auto-retry countdown
	_auto_retry_remaining = _auto_retry_sec
	_countdown_label.visible = true
	_countdown_label.text = "Auto-retry in %ds..." % int(_auto_retry_remaining)
	_auto_retry_timer.start()


func _show_error(title: String, detail: String) -> void:
	_status_label.text = title
	_status_label.add_theme_color_override("font_color", Color(0.9, 0.4, 0.35))
	_error_detail.text = detail
	_error_detail.visible = true
	_retry_btn.visible = true


func _on_auto_retry_tick() -> void:
	_auto_retry_remaining -= 1.0
	if _auto_retry_remaining <= 0:
		_auto_retry_timer.stop()
		_countdown_label.visible = false
		_retry_btn.visible = false
		_settings_btn.visible = false
		_error_detail.visible = false
		_run_startup()
	else:
		_countdown_label.text = "Auto-retry in %ds..." % int(_auto_retry_remaining)

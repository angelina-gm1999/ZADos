##
## ConversationWorkspace — the primary ZADOS interface.
##
## Layout (spec §1.1):
##   [ThinkingPanel] [❯] [message history + input bar] [❮] [StatsPanel]
##
## Addendum B.1: stop button, slash commands, mode selector popup,
## phase failure/timing, message history recall, pending result consumption.
##
extends Control

# ---------------------------------------------------------------------------
# Preloads
# ---------------------------------------------------------------------------

const _MessageBubble  = preload("res://scripts/components/MessageBubble.gd")
const _ThinkingPanel  = preload("res://scripts/panels/ThinkingPanel.gd")
const _StatsPanel     = preload("res://scripts/panels/StatsPanel.gd")
const _Toast          = preload("res://scripts/components/Toast.gd")
const _ConfirmDialog  = preload("res://scripts/components/ConfirmationDialog.gd")

# ---------------------------------------------------------------------------
# Node refs
# ---------------------------------------------------------------------------

@onready var _thinking_panel : Control         = $MainLayout/ThinkingPanel
@onready var _stats_panel    : Control         = $MainLayout/StatsPanel
@onready var _think_toggle   : Button          = $MainLayout/ThinkingToggle
@onready var _stats_toggle   : Button          = $MainLayout/StatsToggle
@onready var _message_scroll : ScrollContainer = $MainLayout/CenterArea/MessageScroll
@onready var _message_list   : VBoxContainer   = $MainLayout/CenterArea/MessageScroll/MessageList
@onready var _input_text     : TextEdit        = $MainLayout/CenterArea/InputBar/InputText
@onready var _send_btn       : Button          = $MainLayout/CenterArea/InputBar/SendButton
@onready var _mode_btn       : Button          = $MainLayout/CenterArea/InputBar/ModeButton

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

var _generating_bubble : Control = null
var _is_generating     : bool    = false
var _gen_start_ms      : int     = 0
var _phase_times       : Dictionary = {}   # phase → elapsed_ms
var _phase_start_ms    : int     = 0

# Message history recall (B.1.1)
var _sent_history      : Array[String] = []
var _history_index     : int = -1
var _history_draft     : String = ""

# Mode selector popup (B.1.3)
var _mode_popup        : Control = null

# Prefill label
var _prefill_label     : Label = null

# ---------------------------------------------------------------------------

func _ready() -> void:
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	$MainLayout.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)

	_think_toggle.pressed.connect(_toggle_thinking)
	_stats_toggle.pressed.connect(_toggle_stats)

	_send_btn.pressed.connect(_on_send)
	_input_text.gui_input.connect(_on_input_key)
	_mode_btn.pressed.connect(_on_mode_btn)

	ZADOSClient.session_opened.connect(_on_session_opened)
	ZADOSClient.turn_phase_updated.connect(_on_phase_updated)
	ZADOSClient.turn_token.connect(_on_token)
	ZADOSClient.turn_complete.connect(_on_turn_complete)
	ZADOSClient.turn_error.connect(_on_turn_error)
	ZADOSClient.generation_cancelled.connect(_on_generation_cancelled)

	set_process_unhandled_key_input(true)

	# Consume prefill text from other workspaces (e.g. Unsolved → Send to Chat)
	if not ZADOSClient.prefill_text.is_empty():
		_input_text.text = ZADOSClient.prefill_text
		_show_prefill_label("From Unsolved Question")
		ZADOSClient.prefill_text = ""
		_input_text.grab_focus()

	# Consume pending result if generation completed while user was away (A.5)
	var pending := ZADOSClient.consume_pending_result()
	if not pending.is_empty():
		_append_completed_result(pending)

	# If generation is in progress (user navigated away and back), show phase bar
	if ZADOSClient.is_generating:
		_is_generating = true
		_send_btn.text = "■"
		_send_btn.disabled = false
		_input_text.editable = false
		_generating_bubble = _add_bubble(_MessageBubble.Role.GENERATING)
		# Mark phases already completed
		for i in range(1, ZADOSClient.current_phase + 1):
			if _generating_bubble:
				_generating_bubble.phase_done(i, {})


func _unhandled_key_input(event: InputEvent) -> void:
	if not event is InputEventKey:
		return
	if _input_text.has_focus():
		return
	var key := event as InputEventKey
	if not key.pressed:
		return
	match key.keycode:
		KEY_T: _toggle_thinking()
		KEY_S: _toggle_stats()

# ---------------------------------------------------------------------------
# Send / receive  (addendum B.1.1)
# ---------------------------------------------------------------------------

func _on_send() -> void:
	# Stop button during generation
	if _is_generating:
		ZADOSClient.cancel_generation()
		return

	var text := _input_text.text.strip_edges()
	if text.is_empty():
		return

	# Clear prefill label
	_clear_prefill_label()

	# Check for slash commands (B.1.1)
	if text.begins_with("/"):
		_handle_slash_command(text)
		_input_text.text = ""
		return

	# Save to history for recall
	_sent_history.append(text)
	_history_index = -1

	_input_text.text = ""
	_is_generating   = true
	_gen_start_ms    = Time.get_ticks_msec()
	_phase_times.clear()
	_phase_start_ms  = _gen_start_ms

	# Input bar → stop mode
	_send_btn.text = "■"
	_send_btn.disabled = false
	_input_text.editable = false

	_add_bubble(_MessageBubble.Role.USER, text)
	_thinking_panel.clear()
	_generating_bubble = _add_bubble(_MessageBubble.Role.GENERATING)

	ZADOSClient.send_message(text, "conversation")


## Slash command routing (B.1.1)
func _handle_slash_command(cmd: String) -> void:
	var parts := cmd.strip_edges().split(" ", false)
	var base  := parts[0].to_lower()

	# System message bubble for command
	_add_system_bubble(cmd)

	match base:
		"/sleep":
			if parts.size() > 1:
				var mode := parts[1].to_lower()
				if mode in ["rem", "dream", "triage"]:
					ZADOSClient.activate_sleep(mode)
				else:
					_show_toast("Unknown sleep mode: %s" % mode, _Toast.Level.WARNING)
			else:
				_show_toast("Usage: /sleep rem | /sleep dream | /sleep triage", _Toast.Level.INFO)
		"/homework":
			ZADOSClient.run_homework()
			_show_toast("Homework pipeline started", _Toast.Level.INFO)
		_:
			_show_toast("Unknown command: %s" % base, _Toast.Level.WARNING)


func _on_input_key(event: InputEvent) -> void:
	if not event is InputEventKey:
		return
	var key := event as InputEventKey
	if not key.pressed:
		return

	# Enter without Shift → send (or Ctrl+Enter from anywhere)
	if key.keycode == KEY_ENTER and not key.shift_pressed:
		_on_send()
		get_viewport().set_input_as_handled()
		return

	# Ctrl+Up/Down: message history recall (B.1.1)
	if key.ctrl_pressed and key.keycode == KEY_UP:
		_recall_history(-1)
		get_viewport().set_input_as_handled()
		return
	if key.ctrl_pressed and key.keycode == KEY_DOWN:
		_recall_history(1)
		get_viewport().set_input_as_handled()
		return

	# Ctrl+M: open mode selector
	if key.ctrl_pressed and key.keycode == KEY_M:
		_on_mode_btn()
		get_viewport().set_input_as_handled()
		return


## Navigate through previously sent messages (B.1.1)
func _recall_history(direction: int) -> void:
	if _sent_history.is_empty():
		return
	if _history_index == -1:
		_history_draft = _input_text.text
		_history_index = _sent_history.size()
	_history_index += direction
	_history_index = clampi(_history_index, 0, _sent_history.size())
	if _history_index == _sent_history.size():
		_input_text.text = _history_draft
		_history_index = -1
	else:
		_input_text.text = _sent_history[_history_index]


## Mode selector popup (B.1.3)
func _on_mode_btn() -> void:
	if _is_generating:
		return   # Disabled during generation (B.1.3)
	if _mode_popup:
		_mode_popup.queue_free()
		_mode_popup = null
		return
	_build_mode_popup()


func _build_mode_popup() -> void:
	var modes := [
		["Normal",          "Regular input"],
		["LearningMode_M1", "M1 — Teach me something"],
		["LearningMode_M2", "M2 — Review / quiz me"],
		["LearningMode_M3", "M3 — Explore / socratic"],
		["LearningMode_M4", "M4 — Questions I have"],
		["LearningMode_M5", "M5 — Independent study"],
		["_sep", ""],
		["SelfReflective",  "Introspection on unsolved questions"],
	]

	_mode_popup = PanelContainer.new()
	var sb := StyleBoxFlat.new()
	sb.bg_color = Color(0.11, 0.11, 0.14, 0.97)
	sb.border_color = Color(0.22, 0.22, 0.28)
	sb.border_width_top = 1
	sb.border_width_bottom = 1
	sb.border_width_left = 1
	sb.border_width_right = 1
	sb.corner_radius_top_left = 6
	sb.corner_radius_top_right = 6
	sb.corner_radius_bottom_left = 6
	sb.corner_radius_bottom_right = 6
	sb.content_margin_left = 8
	sb.content_margin_right = 8
	sb.content_margin_top = 8
	sb.content_margin_bottom = 8
	_mode_popup.add_theme_stylebox_override("panel", sb)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 2)
	_mode_popup.add_child(vbox)

	for entry in modes:
		var key  : String = entry[0]
		var desc : String = entry[1]

		if key == "_sep":
			var sep := HSeparator.new()
			sep.add_theme_constant_override("separation", 4)
			vbox.add_child(sep)
			continue

		var btn := Button.new()
		btn.text = desc
		btn.flat = true
		btn.alignment = HORIZONTAL_ALIGNMENT_LEFT
		btn.add_theme_font_size_override("font_size", 13)

		if key == ZADOSClient.current_mode:
			btn.add_theme_color_override("font_color", Color(0.4, 0.8, 1.0))
			btn.text = "✓ " + btn.text
		else:
			btn.add_theme_color_override("font_color", Color(0.75, 0.78, 0.82))

		btn.pressed.connect(func bound_key=key:
			ZADOSClient.set_session_mode(bound_key)
			_mode_popup.queue_free()
			_mode_popup = null
		)
		vbox.add_child(btn)

	# Position above the input bar
	_mode_popup.custom_minimum_size = Vector2(280, 0)
	var popup_pos := _mode_btn.global_position
	_mode_popup.position = Vector2(popup_pos.x, popup_pos.y - 280)

	var main := get_tree().get_root().get_node_or_null("Main")
	if main:
		main.add_child(_mode_popup)


# ---------------------------------------------------------------------------
# ZADOSClient signal handlers
# ---------------------------------------------------------------------------

func _on_session_opened(data: Dictionary) -> void:
	_mode_btn.text = data.get("initial_mode", "Normal")


func _on_phase_updated(phase: int, data: Dictionary) -> void:
	if _generating_bubble == null:
		return

	# Record phase timing (B.1.2)
	var now := Time.get_ticks_msec()
	_phase_times[phase] = now - _phase_start_ms
	_phase_start_ms = now

	_generating_bubble.phase_done(phase, data)

	# Show timing on the phase badge
	_generating_bubble.set_phase_timing(phase, _phase_times[phase])


func _on_token(phase: int, text: String) -> void:
	if phase == 6 and _generating_bubble != null:
		_generating_bubble.append_token(text)
		_scroll_to_bottom()


func _on_turn_complete(result: Dictionary) -> void:
	_finish_generation(result)


func _on_turn_error(error: Dictionary) -> void:
	_is_generating = false
	_send_btn.text = "Send"
	_send_btn.disabled = false
	_input_text.editable = true

	if _generating_bubble:
		_generating_bubble.show_error(
			error.get("phase", 0),
			error.get("reason", "Unknown error"))
		_generating_bubble = null

	# Keep user message in input bar for retry
	_show_toast(
		"Generation failed at Phase %d: %s" % [
			error.get("phase", 0),
			error.get("reason", "unknown")],
		_Toast.Level.ERROR)


func _on_generation_cancelled() -> void:
	_is_generating = false
	_send_btn.text = "Send"
	_send_btn.disabled = false
	_input_text.editable = true

	if _generating_bubble:
		_generating_bubble.show_cancelled()
		_generating_bubble = null


func _finish_generation(result: Dictionary) -> void:
	_is_generating = false
	_send_btn.text = "Send"
	_send_btn.disabled = false
	_input_text.editable = true

	if _generating_bubble != null:
		var directive : String = result.get("directive", "allow")
		var final_answer : String = result.get("final_answer", "")
		if not final_answer.is_empty() and not _generating_bubble._streaming_started:
			_generating_bubble.set_text(final_answer)

		# Total generation time (B.1.2)
		var total_ms := Time.get_ticks_msec() - _gen_start_ms
		_generating_bubble.set_total_time(total_ms)

		_generating_bubble.finalise(directive)
		_generating_bubble = null

	# Update mode badge
	var session_data : Dictionary = result.get("session", {})
	var mode : String = session_data.get("active_mode", "")
	if not mode.is_empty():
		_mode_btn.text = mode

	_scroll_to_bottom()


## Append a completed result that arrived while user was on another workspace.
func _append_completed_result(result: Dictionary) -> void:
	var answer : String = result.get("final_answer", "")
	if answer.is_empty():
		return
	var bubble := _add_bubble(_MessageBubble.Role.AI, answer)
	var directive : String = result.get("directive", "allow")
	bubble.finalise(directive)

	var session_data := result.get("session", {}) as Dictionary
	var mode : String = session_data.get("active_mode", "")
	if not mode.is_empty():
		_mode_btn.text = mode

	_scroll_to_bottom()


# ---------------------------------------------------------------------------
# Panel toggles
# ---------------------------------------------------------------------------

func _toggle_thinking() -> void:
	_thinking_panel.toggle()
	_think_toggle.text = "❮" if _thinking_panel._expanded else "❯"


func _toggle_stats() -> void:
	_stats_panel.toggle()
	_stats_toggle.text = "❯" if _stats_panel._expanded else "❮"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

func _add_bubble(role: int, text: String = "") -> Control:
	var bubble : Control = _MessageBubble.new()
	bubble.initialize(role, text)
	_message_list.add_child(bubble)
	_scroll_to_bottom()
	return bubble


## Add a system-style message (for slash commands).
func _add_system_bubble(text: String) -> void:
	var lbl := Label.new()
	lbl.text = text
	lbl.add_theme_font_size_override("font_size", 12)
	lbl.add_theme_color_override("font_color", Color(0.45, 0.48, 0.55))
	lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	lbl.add_theme_constant_override("margin_top", 4)
	lbl.add_theme_constant_override("margin_bottom", 4)
	_message_list.add_child(lbl)
	_scroll_to_bottom()


func _show_prefill_label(text: String) -> void:
	_clear_prefill_label()
	_prefill_label = Label.new()
	_prefill_label.text = text
	_prefill_label.add_theme_font_size_override("font_size", 11)
	_prefill_label.add_theme_color_override("font_color", Color(0.45, 0.55, 0.75))
	# Insert above input bar
	var input_bar = $MainLayout/CenterArea/InputBar
	input_bar.get_parent().add_child(_prefill_label)
	input_bar.get_parent().move_child(_prefill_label, input_bar.get_index())


func _clear_prefill_label() -> void:
	if _prefill_label:
		_prefill_label.queue_free()
		_prefill_label = null


func _show_toast(text: String, level: int) -> void:
	var tc = get_tree().get_root().get_node_or_null("Main/ToastContainer")
	if tc:
		tc.show_toast(text, level)


func _scroll_to_bottom() -> void:
	await get_tree().process_frame
	var vbar := _message_scroll.get_v_scroll_bar()
	_message_scroll.scroll_vertical = int(vbar.max_value)

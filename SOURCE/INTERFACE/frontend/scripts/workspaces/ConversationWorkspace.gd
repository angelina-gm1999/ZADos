##
## ConversationWorkspace — the primary ZADOS interface.
##
## Layout (spec §1.1):
##   [ThinkingPanel] [❯] [message history + input bar] [❮] [StatsPanel]
##
## All ZADOSClient signals are connected here and routed to the appropriate
## child panels / message bubbles.
##
extends Control

# ---------------------------------------------------------------------------
# Preloads (avoids class_name resolution issues in headless parse)
# ---------------------------------------------------------------------------

const _MessageBubble  = preload("res://scripts/components/MessageBubble.gd")
const _ThinkingPanel  = preload("res://scripts/panels/ThinkingPanel.gd")
const _StatsPanel     = preload("res://scripts/panels/StatsPanel.gd")

# ---------------------------------------------------------------------------
# Node refs (matched to ConversationWorkspace.tscn)
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
var _is_generating     : bool          = false

# ---------------------------------------------------------------------------

func _ready() -> void:
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	$MainLayout.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)

	# Toggle buttons
	_think_toggle.pressed.connect(_toggle_thinking)
	_stats_toggle.pressed.connect(_toggle_stats)

	# Input actions
	_send_btn.pressed.connect(_on_send)
	_input_text.gui_input.connect(_on_input_key)
	_mode_btn.pressed.connect(_on_mode_btn)

	# ZADOSClient signals
	ZADOSClient.session_opened.connect(_on_session_opened)
	ZADOSClient.turn_phase_updated.connect(_on_phase_updated)
	ZADOSClient.turn_token.connect(_on_token)
	ZADOSClient.turn_complete.connect(_on_turn_complete)

	# Keyboard shortcuts
	set_process_unhandled_key_input(true)

	# Check for prefill text from other workspaces (e.g. Unsolved → Send to Chat)
	if not ZADOSClient.prefill_text.is_empty():
		_input_text.text = ZADOSClient.prefill_text
		ZADOSClient.prefill_text = ""
		_input_text.grab_focus()


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
# Send / receive
# ---------------------------------------------------------------------------

func _on_send() -> void:
	var text := _input_text.text.strip_edges()
	if text.is_empty() or _is_generating:
		return

	_input_text.text = ""
	_is_generating   = true
	_send_btn.disabled = true

	_add_bubble(_MessageBubble.Role.USER, text)

	_thinking_panel.clear()
	_generating_bubble = _add_bubble(_MessageBubble.Role.GENERATING)

	ZADOSClient.stream_turn(text)


func _on_input_key(event: InputEvent) -> void:
	if not event is InputEventKey:
		return
	var key := event as InputEventKey
	# Enter without Shift → send
	if key.pressed and key.keycode == KEY_ENTER and not key.shift_pressed:
		_on_send()
		get_viewport().set_input_as_handled()


func _on_mode_btn() -> void:
	# Switch to Learning Workspace so the user can pick a mode
	var main := get_tree().get_root().get_node_or_null("Main")
	if main and main.has_method("_switch_to"):
		main._switch_to("learning")

# ---------------------------------------------------------------------------
# ZADOSClient signal handlers
# ---------------------------------------------------------------------------

func _on_session_opened(data: Dictionary) -> void:
	_mode_btn.text = data.get("initial_mode", "Normal")


func _on_phase_updated(phase: int, data: Dictionary) -> void:
	if _generating_bubble == null:
		return
	_generating_bubble.phase_done(phase, data)


func _on_token(phase: int, text: String) -> void:
	# Phase 4 tokens → ThinkingPanel (handled there via its own signal connection)
	# Phase 6 tokens → the generating bubble
	if phase == 6 and _generating_bubble != null:
		_generating_bubble.append_token(text)
		_scroll_to_bottom()


func _on_turn_complete(result: Dictionary) -> void:
	_is_generating   = false
	_send_btn.disabled = false

	if _generating_bubble != null:
		var directive : String = result.get("directive", "allow")

		# If no streaming tokens arrived for phase 6 (sync path), set text now.
		var final_answer : String = result.get("final_answer", "")
		if not final_answer.is_empty() and not _generating_bubble._streaming_started:
			_generating_bubble.set_text(final_answer)

		_generating_bubble.finalise(directive)
		_generating_bubble = null

	# Update mode badge from result
	var session_data : Dictionary = result.get("session", {})
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


func _scroll_to_bottom() -> void:
	# Defer one frame so the new node has laid out before scrolling.
	await get_tree().process_frame
	var vbar := _message_scroll.get_v_scroll_bar()
	_message_scroll.scroll_vertical = int(vbar.max_value)

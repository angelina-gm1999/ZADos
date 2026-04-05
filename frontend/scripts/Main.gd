##
## Main — root orchestrator.
##
## Manages: startup sequence, workspace navigation, sleep overlay,
## toast/modal containers, generation badges, global keyboard shortcuts.
##
## Addendum A.1 (startup), A.5 (generation resilience), A.9 (shortcuts).
##
extends Control

# ---------------------------------------------------------------------------
# Workspace registry
# ---------------------------------------------------------------------------

const WORKSPACE_SCENES := {
	"conversation": "res://scenes/workspaces/ConversationWorkspace.tscn",
	"memory":       "res://scenes/workspaces/MemoryWorkspace.tscn",
	"learning":     "res://scenes/workspaces/LearningWorkspace.tscn",
	"dev":          "res://scenes/workspaces/DevWorkspace.tscn",
	"map":          "res://scenes/workspaces/MapWorkspace.tscn",
}

const WORKSPACE_KEYS := ["conversation", "memory", "learning", "dev", "map"]

const _StartupScreen    = preload("res://scripts/StartupScreen.gd")
const _ToastContainer   = preload("res://scripts/components/ToastContainer.gd")
const _ConfirmDialog    = preload("res://scripts/components/ConfirmationDialog.gd")

# ---------------------------------------------------------------------------
# Node refs
# ---------------------------------------------------------------------------

@onready var _container : Control      = $UI/WorkspaceContainer
@onready var _nav       : HBoxContainer = $UI/NavBar
@onready var _status    : HBoxContainer = $UI/StatusStrip

var _active_workspace : Node   = null
var _active_key       : String = ""
var _sleep_overlay    : Control = null
var _toast_container  : Control = null
var _modal_container  : Control = null
var _startup_screen   : Control = null
var _minimized_sleep  : Control = null

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

func _ready() -> void:
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	$UI.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)

	# NavBar + StatusStrip start hidden until startup completes
	_nav.visible = false
	_status.visible = false

	_nav.workspace_selected.connect(_switch_to)

	# Toast container (bottom-left, persistent)
	_toast_container = _ToastContainer.new()
	_toast_container.name = "ToastContainer"
	add_child(_toast_container)

	# Modal container (centered overlay layer)
	_modal_container = Control.new()
	_modal_container.name = "ModalContainer"
	_modal_container.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_modal_container.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(_modal_container)

	# Preload sleep overlay (hidden by default)
	var sleep_scene : PackedScene = load("res://scenes/overlays/SleepOverlay.tscn")
	if sleep_scene:
		_sleep_overlay = sleep_scene.instantiate()
		_sleep_overlay.visible = false
		_sleep_overlay.close_requested.connect(_hide_sleep_overlay)

	# Build minimized sleep bar (hidden by default)
	_build_minimized_sleep_bar()

	# ZADOSClient signals
	ZADOSClient.session_opened.connect(_on_session_opened)
	ZADOSClient.turn_complete.connect(_on_turn_complete)
	ZADOSClient.turn_error.connect(_on_turn_error)
	ZADOSClient.sleep_triggered.connect(_on_sleep_triggered)
	ZADOSClient.sleep_activated.connect(_on_sleep_activated)
	ZADOSClient.sleep_exited.connect(_on_sleep_exited)
	ZADOSClient.generation_started.connect(_on_generation_started)
	ZADOSClient.generation_cancelled.connect(_on_generation_cancelled)
	ZADOSClient.connection_lost.connect(_on_connection_lost)
	ZADOSClient.connection_restored.connect(_on_connection_restored)
	ZADOSClient.homework_complete.connect(_on_homework_complete)

	# Global keyboard shortcuts
	set_process_unhandled_key_input(true)

	# Run startup sequence (addendum A.1)
	_show_startup_screen()


# ---------------------------------------------------------------------------
# Startup  (addendum A.1)
# ---------------------------------------------------------------------------

func _show_startup_screen() -> void:
	_startup_screen = _StartupScreen.new()
	_startup_screen.startup_complete.connect(_on_startup_complete)
	_container.add_child(_startup_screen)


func _on_startup_complete() -> void:
	if _startup_screen:
		_startup_screen.queue_free()
		_startup_screen = null

	# Show UI
	_nav.visible = true
	_status.visible = true

	# Add sleep overlay to container now
	if _sleep_overlay:
		_container.add_child(_sleep_overlay)

	# Load default workspace
	_switch_to("conversation")


# ---------------------------------------------------------------------------
# ZADOSClient callbacks
# ---------------------------------------------------------------------------

func _on_session_opened(data: Dictionary) -> void:
	_status.refresh(data)


func _on_turn_complete(result: Dictionary) -> void:
	_status.refresh(result.get("session", {}))
	_nav.clear_badge("conversation")


func _on_turn_error(error: Dictionary) -> void:
	_nav.clear_badge("conversation")
	_toast_container.show_toast(
		"Generation failed: %s" % error.get("reason", "unknown"),
		1,   # Toast.Level.ERROR = 3, but use index
		true,
		Callable()
	)


func _on_generation_started() -> void:
	# Show animated badge on the source workspace tab
	_nav.set_badge(ZADOSClient.generation_target_workspace, true)


func _on_generation_cancelled() -> void:
	_nav.clear_badge(ZADOSClient.generation_target_workspace)


func _on_homework_complete(_result: Dictionary) -> void:
	_nav.clear_badge("learning")
	_toast_container.show_toast("Homework pipeline complete", 1)


func _on_sleep_triggered(_result: Dictionary) -> void:
	show_sleep_overlay()


func _on_sleep_activated(sleep_type: String) -> void:
	show_sleep_overlay()


func _on_sleep_exited() -> void:
	_hide_sleep_overlay()
	_hide_minimized_sleep()


func _on_connection_lost() -> void:
	_toast_container.show_toast("Connection to server lost. Reconnecting...", 3)


func _on_connection_restored() -> void:
	_toast_container.show_toast("Connection restored", 1)


# ---------------------------------------------------------------------------
# Sleep overlay  (addendum B.4.2)
# ---------------------------------------------------------------------------

func show_sleep_overlay() -> void:
	if _sleep_overlay:
		_sleep_overlay.visible = true
		_sleep_overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
		_container.move_child(_sleep_overlay, _container.get_child_count() - 1)
		_hide_minimized_sleep()


func _hide_sleep_overlay() -> void:
	if _sleep_overlay:
		_sleep_overlay.visible = false


func minimize_sleep_overlay() -> void:
	_hide_sleep_overlay()
	_show_minimized_sleep()


func _build_minimized_sleep_bar() -> void:
	_minimized_sleep = HBoxContainer.new()
	_minimized_sleep.name = "MinimizedSleepBar"
	_minimized_sleep.visible = false
	_minimized_sleep.custom_minimum_size = Vector2(0, 28)
	_minimized_sleep.set_anchors_and_offsets_preset(Control.PRESET_TOP_WIDE)

	var bg := StyleBoxFlat.new()
	bg.bg_color = Color(0.10, 0.08, 0.18, 0.92)
	bg.content_margin_left = 12
	bg.content_margin_right = 12

	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", bg)
	panel.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_minimized_sleep.add_child(panel)

	var inner := HBoxContainer.new()
	inner.add_theme_constant_override("separation", 12)
	panel.add_child(inner)

	var lbl := Label.new()
	lbl.text = "Sleep Mode Active"
	lbl.add_theme_font_size_override("font_size", 12)
	lbl.add_theme_color_override("font_color", Color(0.65, 0.55, 0.85))
	inner.add_child(lbl)

	var mode_lbl := Label.new()
	mode_lbl.name = "SleepModeLabel"
	mode_lbl.text = "[REM]"
	mode_lbl.add_theme_font_size_override("font_size", 12)
	mode_lbl.add_theme_color_override("font_color", Color(0.4, 0.6, 1.0))
	inner.add_child(mode_lbl)

	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND
	inner.add_child(spacer)

	var expand_btn := Button.new()
	expand_btn.text = "Expand"
	expand_btn.flat = true
	expand_btn.add_theme_font_size_override("font_size", 12)
	expand_btn.add_theme_color_override("font_color", Color(0.5, 0.7, 1.0))
	expand_btn.pressed.connect(show_sleep_overlay)
	inner.add_child(expand_btn)

	var exit_btn := Button.new()
	exit_btn.text = "Exit"
	exit_btn.flat = true
	exit_btn.add_theme_font_size_override("font_size", 12)
	exit_btn.add_theme_color_override("font_color", Color(0.8, 0.5, 0.5))
	exit_btn.pressed.connect(func(): ZADOSClient.exit_sleep())
	inner.add_child(exit_btn)

	add_child(_minimized_sleep)


func _show_minimized_sleep() -> void:
	if _minimized_sleep:
		_minimized_sleep.visible = true


func _hide_minimized_sleep() -> void:
	if _minimized_sleep:
		_minimized_sleep.visible = false


# ---------------------------------------------------------------------------
# Workspace swap  (addendum A.5 + A.6)
# ---------------------------------------------------------------------------

func _switch_to(key: String) -> void:
	if key == _active_key:
		return

	# Tear down current workspace.
	if _active_workspace != null:
		_container.remove_child(_active_workspace)
		_active_workspace.queue_free()
		_active_workspace = null

	# Instantiate next workspace.
	var path: String = WORKSPACE_SCENES.get(key, "")
	if path.is_empty():
		push_error("Main: unknown workspace key '%s'" % key)
		return

	var scene: PackedScene = load(path)
	if scene == null:
		push_error("Main: failed to load scene at '%s'" % path)
		return

	_active_workspace = scene.instantiate()
	_container.add_child(_active_workspace)

	if _active_workspace is Control:
		_active_workspace.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)

	# Keep sleep overlay on top
	if _sleep_overlay and _sleep_overlay.get_parent() == _container:
		_container.move_child(_sleep_overlay, _container.get_child_count() - 1)

	_active_key = key

	# Notify ZADOSClient of workspace change (metrics polling, staleness)
	ZADOSClient.set_active_workspace(key)

	# Check for stale data — workspace.refresh() is called if stale
	if ZADOSClient.is_stale(key) or ZADOSClient.is_stale(_stale_key(key)):
		if _active_workspace.has_method("refresh"):
			_active_workspace.refresh()
		ZADOSClient.clear_stale(key)
		ZADOSClient.clear_stale(_stale_key(key))


## Map workspace key → staleness key (some workspaces have sub-keys).
func _stale_key(ws_key: String) -> String:
	match ws_key:
		"memory": return "memory_stmm"   # any memory sub-tab
		_: return ws_key


# ---------------------------------------------------------------------------
# Global keyboard shortcuts  (addendum A.9)
# ---------------------------------------------------------------------------

func _unhandled_key_input(event: InputEvent) -> void:
	if not event is InputEventKey:
		return
	var key := event as InputEventKey
	if not key.pressed:
		return

	# Ctrl+1 through Ctrl+5: switch workspace
	if key.ctrl_pressed and not key.shift_pressed:
		var idx := -1
		match key.keycode:
			KEY_1: idx = 0
			KEY_2: idx = 1
			KEY_3: idx = 2
			KEY_4: idx = 3
			KEY_5: idx = 4
		if idx >= 0 and idx < WORKSPACE_KEYS.size():
			_switch_to(WORKSPACE_KEYS[idx])
			_nav._set_active(WORKSPACE_KEYS[idx], false)
			get_viewport().set_input_as_handled()
			return

	# Ctrl+Shift+D: quick toggle Dev workspace
	if key.ctrl_pressed and key.shift_pressed and key.keycode == KEY_D:
		if _active_key == "dev":
			_switch_to("conversation")
			_nav._set_active("conversation", false)
		else:
			_switch_to("dev")
			_nav._set_active("dev", false)
		get_viewport().set_input_as_handled()
		return

	# Ctrl+Shift+S: open NerdStats (switches to conversation + opens stats)
	if key.ctrl_pressed and key.shift_pressed and key.keycode == KEY_S:
		_switch_to("conversation")
		_nav._set_active("conversation", false)
		if _active_workspace and _active_workspace.has_method("_toggle_stats"):
			_active_workspace._toggle_stats()
		get_viewport().set_input_as_handled()
		return

	# Ctrl+Shift+T: open Thinking panel
	if key.ctrl_pressed and key.shift_pressed and key.keycode == KEY_T:
		_switch_to("conversation")
		_nav._set_active("conversation", false)
		if _active_workspace and _active_workspace.has_method("_toggle_thinking"):
			_active_workspace._toggle_thinking()
		get_viewport().set_input_as_handled()
		return

	# Escape: close any open modal, popover, or expanded panel
	if key.keycode == KEY_ESCAPE:
		# Modals handle their own Escape; this catches leftover cases
		if _sleep_overlay and _sleep_overlay.visible:
			minimize_sleep_overlay()
			get_viewport().set_input_as_handled()
			return

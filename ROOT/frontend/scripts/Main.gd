extends Control

# ---------------------------------------------------------------------------
# Workspace registry — maps nav key → scene path.
# Add new workspaces here; the swap logic handles the rest.
# ---------------------------------------------------------------------------

const WORKSPACE_SCENES := {
	"conversation": "res://scenes/workspaces/ConversationWorkspace.tscn",
	"memory":       "res://scenes/workspaces/MemoryWorkspace.tscn",
	"learning":     "res://scenes/workspaces/LearningWorkspace.tscn",
	"dev":          "res://scenes/workspaces/DevWorkspace.tscn",
	"map":          "res://scenes/workspaces/MapWorkspace.tscn",
}

@onready var _container : Control      = $UI/WorkspaceContainer
@onready var _nav       : HBoxContainer = $UI/NavBar
@onready var _status    : HBoxContainer = $UI/StatusStrip

var _active_workspace : Node   = null
var _active_key       : String = ""
var _sleep_overlay    : Control = null

# ---------------------------------------------------------------------------

func _ready() -> void:
	# Fill the viewport.
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	$UI.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)

	_nav.workspace_selected.connect(_switch_to)

	ZADOSClient.session_opened.connect(_on_session_opened)
	ZADOSClient.turn_complete.connect(_on_turn_complete)
	ZADOSClient.sleep_triggered.connect(_on_sleep_triggered)
	ZADOSClient.open_session()

	# Preload sleep overlay (hidden by default)
	var sleep_scene : PackedScene = load("res://scenes/overlays/SleepOverlay.tscn")
	if sleep_scene:
		_sleep_overlay = sleep_scene.instantiate()
		_sleep_overlay.visible = false
		_sleep_overlay.close_requested.connect(_hide_sleep_overlay)
		_container.add_child(_sleep_overlay)

	_switch_to("conversation")


# ---------------------------------------------------------------------------
# ZADOSClient callbacks
# ---------------------------------------------------------------------------

func _on_session_opened(data: Dictionary) -> void:
	_status.refresh(data)


func _on_turn_complete(result: Dictionary) -> void:
	_status.refresh(result.get("session", {}))


func _on_sleep_triggered(_result: Dictionary) -> void:
	show_sleep_overlay()


func show_sleep_overlay() -> void:
	if _sleep_overlay:
		_sleep_overlay.visible = true
		_sleep_overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
		_container.move_child(_sleep_overlay, _container.get_child_count() - 1)


func _hide_sleep_overlay() -> void:
	if _sleep_overlay:
		_sleep_overlay.visible = false

# ---------------------------------------------------------------------------
# Workspace swap
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

	_active_key = key

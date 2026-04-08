##
## ToastContainer — persistent bottom-left stack of Toast notifications.
##
## Added as a direct child of Main. Provides the global `show_toast()` API
## that any panel can call via:
##   get_tree().get_root().get_node("Main/ToastContainer").show_toast(...)
##
## Or more conveniently through the helper: ToastContainer.toast(tree, ...)
##
extends VBoxContainer

const _Toast = preload("res://scripts/components/Toast.gd")

const MAX_VISIBLE := 5

# ---------------------------------------------------------------------------

func _ready() -> void:
	set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_LEFT)
	anchor_left   = 0.0
	anchor_right  = 0.0
	anchor_bottom = 1.0
	anchor_top    = 1.0
	offset_left   = 16
	offset_bottom = -32
	offset_top    = -300
	offset_right  = 380
	size_flags_vertical = Control.SIZE_SHRINK_END
	add_theme_constant_override("separation", 6)
	mouse_filter = Control.MOUSE_FILTER_IGNORE


## Create and display a toast notification.
func show_toast(text: String, level: int = _Toast.Level.INFO, retryable: bool = false, retry_callback: Callable = Callable()) -> void:
	# Cap the visible stack
	while get_child_count() >= MAX_VISIBLE:
		var oldest := get_child(0)
		oldest.queue_free()

	var t := _Toast.new()
	t.show_message(text, level, retryable)
	if retryable and retry_callback.is_valid():
		t.retry_requested.connect(retry_callback)
	add_child(t)


## Static helper — any node can call this without caching a reference.
static func toast(tree: SceneTree, text: String, level: int = 0, retryable: bool = false, retry_callback: Callable = Callable()) -> void:
	var container = tree.get_root().get_node_or_null("Main/ToastContainer")
	if container:
		container.show_toast(text, level, retryable, retry_callback)

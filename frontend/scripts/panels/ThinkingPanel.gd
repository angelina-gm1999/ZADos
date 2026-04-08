##
## ThinkingPanel — collapsible left panel showing the VT thinking trace.
##
## Width animates between 0 (collapsed) and EXPANDED_W (open).
## Connects to ZADOSClient.turn_token(4, text) for live streaming.
##
class_name ThinkingPanel
extends Control

const EXPANDED_W   := 280.0
const ANIM_SECONDS := 0.18

var _expanded  : bool  = false
var _label     : RichTextLabel
var _save_btn  : Button
var _last_trace: String = ""

# ---------------------------------------------------------------------------

func _ready() -> void:
	_build_ui()
	ZADOSClient.turn_token.connect(_on_token)
	ZADOSClient.turn_complete.connect(_on_turn_complete)


func _build_ui() -> void:
	clip_contents = true

	var vbox := VBoxContainer.new()
	vbox.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	vbox.add_theme_constant_override("separation", 0)
	add_child(vbox)

	# Header
	var header := HBoxContainer.new()
	header.custom_minimum_size = Vector2(0, 32)
	vbox.add_child(header)

	var title := Label.new()
	title.text = "  Thinking"
	title.size_flags_horizontal = SIZE_EXPAND_FILL
	title.add_theme_color_override("font_color", Color(0.55, 0.55, 0.60))
	title.add_theme_font_size_override("font_size", 11)
	header.add_child(title)

	_save_btn = Button.new()
	_save_btn.text       = "Save"
	_save_btn.flat       = true
	_save_btn.focus_mode = Control.FOCUS_NONE
	_save_btn.add_theme_font_size_override("font_size", 10)
	_save_btn.pressed.connect(_on_save)
	header.add_child(_save_btn)

	# Separator
	var sep := HSeparator.new()
	vbox.add_child(sep)

	# Scrollable thinking text
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical       = SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode    = ScrollContainer.SCROLL_MODE_DISABLED
	vbox.add_child(scroll)

	_label = RichTextLabel.new()
	_label.bbcode_enabled         = true
	_label.scroll_active          = false
	_label.size_flags_horizontal  = SIZE_EXPAND_FILL
	_label.autowrap_mode          = TextServer.AUTOWRAP_WORD_SMART
	_label.add_theme_color_override("default_color", Color(0.55, 0.58, 0.62))
	_label.add_theme_font_size_override("font_size", 12)
	scroll.add_child(_label)

	# Style the panel background
	var bg := StyleBoxFlat.new()
	bg.bg_color = Color(0.07, 0.07, 0.09)
	bg.border_width_right = 1
	bg.border_color = Color(0.18, 0.18, 0.22)
	add_theme_stylebox_override("panel", bg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

func toggle() -> void:
	_expanded = not _expanded
	var target := EXPANDED_W if _expanded else 0.0
	var tween  := create_tween()
	tween.set_ease(Tween.EASE_OUT)
	tween.set_trans(Tween.TRANS_QUART)
	tween.tween_property(self, "custom_minimum_size:x", target, ANIM_SECONDS)


func clear() -> void:
	if _label:
		_label.clear()
	_last_trace = ""

# ---------------------------------------------------------------------------
# Signal handlers
# ---------------------------------------------------------------------------

func _on_token(phase: int, text: String) -> void:
	if phase != 4:
		return
	_last_trace += text
	_label.append_text("[i]%s[/i]" % text)


func _on_turn_complete(_result: Dictionary) -> void:
	# Ensure thinking panel is populated even on sync path
	var result: Dictionary = _result
	var state : Dictionary = result.get("state", {})
	var thinking: Dictionary = state.get("thinking", {})
	var trace : String = thinking.get("thinking_trace", "")

	if _last_trace.is_empty() and not trace.is_empty():
		_label.clear()
		_label.append_text("[i]%s[/i]" % trace)
		_last_trace = trace


func _on_save() -> void:
	if _last_trace.strip_edges().is_empty():
		return
	var body := {
		"thought_fragment": _last_trace,
		"emotion_tag": "",
		"context_summary": "",
		"pipeline_phase": "phase4_thinking",
	}
	ZADOSClient.post_memory("ltmm/thoughts/held_blocks", body)

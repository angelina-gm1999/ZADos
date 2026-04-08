##
## WorkspacePlaceholder — temporary stand-in for unbuilt workspaces.
## Each workspace scene exports its name; this script displays it centered.
##
extends Control

@export var workspace_name: String = "Workspace"

func _ready() -> void:
	var vbox := VBoxContainer.new()
	vbox.alignment = BoxContainer.ALIGNMENT_CENTER
	vbox.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(vbox)

	var label := Label.new()
	label.text                 = workspace_name
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	label.add_theme_font_size_override("font_size", 28)
	label.add_theme_color_override("font_color", Color(0.5, 0.5, 0.5))
	vbox.add_child(label)

	var sub := Label.new()
	sub.text                 = "— coming soon —"
	sub.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	sub.add_theme_font_size_override("font_size", 14)
	sub.add_theme_color_override("font_color", Color(0.35, 0.35, 0.35))
	vbox.add_child(sub)

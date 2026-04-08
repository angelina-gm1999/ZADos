##
## LoadingSkeleton — placeholder UI for panels that are loading data.
##
## Renders animated gray rectangles matching the expected content layout.
## Call `show_skeleton(row_count)` to display, then `queue_free()` when data
## arrives and the real content replaces it.
##
extends VBoxContainer

var _rows : Array[ColorRect] = []
var _time : float = 0.0

# ---------------------------------------------------------------------------

func show_skeleton(row_count: int = 5, row_height: float = 18.0) -> void:
	add_theme_constant_override("separation", 10)
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	size_flags_vertical = Control.SIZE_EXPAND_FILL

	for i in row_count:
		var row := ColorRect.new()
		var width_pct := randf_range(0.45, 0.95)
		row.custom_minimum_size = Vector2(0, row_height)
		row.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		# Start with varying widths to look organic
		row.anchor_right = width_pct
		row.color = Color(0.16, 0.16, 0.20, 0.6)
		add_child(row)
		_rows.append(row)

	set_process(true)


func _process(delta: float) -> void:
	_time += delta
	# Subtle pulse animation
	for i in _rows.size():
		var phase : float = _time * 1.8 + float(i) * 0.4
		var alpha : float = 0.3 + 0.25 * sin(phase)
		_rows[i].color.a = alpha

##
## GraphCanvas — Force-directed graph renderer.
## Pan: middle-mouse drag.  Zoom: scroll wheel.  Select: left-click node.
##
## Addendum B.5.1: 500-atom cap with warning banner.
##
extends Control

signal node_selected(node_id: String, node_data: Dictionary)

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
const REPULSION  := 7000.0
const SPRING_K   := 0.035
const SPRING_LEN := 110.0
const DAMPING    := 0.80
const GRAVITY    := 0.007
const MIN_DIST   := 18.0
const NODE_R     := 11.0
const SIM_TICKS  := 180

const MAX_ATOMS  := 500   # B.5.1

# ---------------------------------------------------------------------------
# Type colours (node type → Color)
# ---------------------------------------------------------------------------
const _NODE_COLOR := {
	"CONCEPT_NODE":   Color(0.40, 0.70, 1.00),
	"PREDICATE_NODE": Color(0.45, 0.90, 0.55),
	"NUMBER_NODE":    Color(1.00, 0.85, 0.30),
	"VARIABLE_NODE":  Color(0.80, 0.50, 0.90),
	"SCHEMA_NODE":    Color(1.00, 0.60, 0.30),
	"GROUNDED_NODE":  Color(0.30, 0.90, 0.70),
	"concept":        Color(0.40, 0.70, 1.00),
	"principle":      Color(0.45, 0.90, 0.55),
	"fact":           Color(1.00, 0.85, 0.30),
	"open_question":  Color(1.00, 0.40, 0.40),
}
const _EDGE_COLOR := {
	"INHERITANCE_LINK": Color(0.40, 0.70, 1.00),
	"SIMILARITY_LINK":  Color(0.45, 0.90, 0.55),
	"HEBBIAN_LINK":     Color(0.80, 0.50, 0.90),
	"IMPLICATION_LINK": Color(1.00, 0.60, 0.30),
	"EVALUATION_LINK":  Color(0.90, 0.75, 0.35),
	"supports":         Color(0.30, 0.80, 0.30),
	"contradicts":      Color(0.90, 0.30, 0.30),
	"extends":          Color(0.40, 0.70, 1.00),
	"requires":         Color(1.00, 0.75, 0.25),
	"exemplifies":      Color(0.70, 0.40, 0.90),
}
const _DEFAULT_NODE_COLOR := Color(0.60, 0.65, 0.72)
const _DEFAULT_EDGE_COLOR := Color(0.30, 0.30, 0.35, 0.70)

# Node shape: type → "circle" | "diamond" | "hexagon" | "square"
const _NODE_SHAPE := {
	"CONCEPT_NODE":   "circle",
	"PREDICATE_NODE": "diamond",
	"NUMBER_NODE":    "square",
	"VARIABLE_NODE":  "hexagon",
	"SCHEMA_NODE":    "diamond",
	"GROUNDED_NODE":  "hexagon",
	"concept":        "circle",
	"principle":      "diamond",
	"fact":           "square",
	"open_question":  "hexagon",
}

# Edge style: type → "solid" | "dashed" | "dotted"
const _EDGE_STYLE := {
	"INHERITANCE_LINK": "solid",
	"SIMILARITY_LINK":  "dashed",
	"HEBBIAN_LINK":     "dotted",
	"IMPLICATION_LINK": "solid",
	"supports":         "solid",
	"contradicts":      "dashed",
	"extends":          "solid",
	"requires":         "dotted",
}

# Fixed world centre — nodes live in world-space; camera transforms to screen.
const _WORLD_CX := 400.0
const _WORLD_CY := 300.0

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
var _nodes       : Dictionary = {}   # id → {pos, vel, label, type, strength, confidence, sti, metadata}
var _edges       : Array      = []   # [{source, target, label, weight}]
var _spring_map  : Dictionary = {}   # id → [neighbour_ids]

var _zoom        : float  = 1.0
var _cam_offset  : Vector2 = Vector2.ZERO
var _dragging    : bool   = false
var _drag_start  : Vector2 = Vector2.ZERO
var _cam_start   : Vector2 = Vector2.ZERO

var _selected_id : String = ""

var _sim_tick    : int  = 0
var _simulating  : bool = false

# B.5.1 — warning banner
var _warning_banner : PanelContainer = null
var _was_capped     : bool = false
var _total_atom_count : int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

func load_graph(data: Dictionary) -> void:
	_nodes.clear()
	_edges.clear()
	_spring_map.clear()
	_selected_id = ""
	_sim_tick    = 0
	_simulating  = true
	_was_capped  = false

	var node_list : Array = data.get("nodes", [])
	_total_atom_count = node_list.size()

	# B.5.1 — cap at MAX_ATOMS
	if node_list.size() > MAX_ATOMS:
		_was_capped = true
		node_list = node_list.slice(0, MAX_ATOMS)

	var count     : int   = node_list.size()
	var angle_step: float = TAU / maxf(1.0, float(count))
	var r_base    : float = minf(50.0 + count * 12.0, 280.0)

	for i in count:
		var n   : Dictionary = node_list[i]
		var nid : String     = str(n.get("id", ""))
		var ang : float      = i * angle_step
		var r   : float      = r_base + randf() * 40.0
		_nodes[nid] = {
			"pos":        Vector2(_WORLD_CX + cos(ang) * r, _WORLD_CY + sin(ang) * r),
			"vel":        Vector2.ZERO,
			"label":      str(n.get("label", nid.left(8))),
			"type":       str(n.get("type",  "concept")),
			"strength":   float(n.get("strength",   0.5)),
			"confidence": float(n.get("confidence", 0.5)),
			"sti":        float(n.get("sti",         0.0)),
			"metadata":   n.get("metadata", {}),
		}

	for e in (data.get("edges", []) as Array):
		var src : String = str(e.get("source", ""))
		var tgt : String = str(e.get("target", ""))
		# Only include edges whose endpoints survived the cap
		if _nodes.has(src) and _nodes.has(tgt):
			_edges.append({
				"source": src,
				"target": tgt,
				"label":  str(e.get("label",  "")),
				"weight": float(e.get("weight", 0.5)),
			})

	_build_spring_map()
	_zoom       = 1.0
	_cam_offset = Vector2.ZERO
	_update_warning_banner()
	queue_redraw()


func restart_simulation() -> void:
	_sim_tick   = 0
	_simulating = true


func clear() -> void:
	_nodes.clear()
	_edges.clear()
	_spring_map.clear()
	_selected_id = ""
	_simulating  = false
	_was_capped  = false
	_total_atom_count = 0
	_update_warning_banner()
	queue_redraw()


func get_node_count() -> int:
	return _nodes.size()


func get_edge_count() -> int:
	return _edges.size()


func get_total_atom_count() -> int:
	return _total_atom_count


func is_capped() -> bool:
	return _was_capped


# ---------------------------------------------------------------------------
# Warning banner (B.5.1)
# ---------------------------------------------------------------------------

func _update_warning_banner() -> void:
	if _warning_banner != null:
		_warning_banner.queue_free()
		_warning_banner = null

	if not _was_capped:
		return

	_warning_banner = PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.35, 0.25, 0.05, 0.90)
	style.set_corner_radius_all(4)
	style.content_margin_left = 10; style.content_margin_right = 10
	style.content_margin_top = 6; style.content_margin_bottom = 6
	_warning_banner.add_theme_stylebox_override("panel", style)

	var lbl := Label.new()
	lbl.text = "Graph capped at %d / %d atoms. Increase filter specificity or use search to narrow results." % [MAX_ATOMS, _total_atom_count]
	lbl.add_theme_color_override("font_color", Color(0.95, 0.80, 0.25))
	lbl.add_theme_font_size_override("font_size", 11)
	lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_warning_banner.add_child(lbl)

	# Position at top of canvas
	_warning_banner.set_anchors_and_offsets_preset(Control.PRESET_TOP_WIDE)
	_warning_banner.offset_top = 4
	_warning_banner.offset_left = 8
	_warning_banner.offset_right = -8
	add_child(_warning_banner)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

func _build_spring_map() -> void:
	_spring_map.clear()
	for e in _edges:
		var s : String = str(e["source"])
		var t : String = str(e["target"])
		if not _spring_map.has(s):
			_spring_map[s] = []
		if not _spring_map.has(t):
			_spring_map[t] = []
		if t not in _spring_map[s]:
			(_spring_map[s] as Array).append(t)
		if s not in _spring_map[t]:
			(_spring_map[t] as Array).append(s)


func _process(_delta: float) -> void:
	if _simulating:
		_simulate_step()
		_sim_tick += 1
		if _sim_tick >= SIM_TICKS:
			_simulating = false
		queue_redraw()


func _simulate_step() -> void:
	var ids   : Array = _nodes.keys()
	var center: Vector2 = Vector2(_WORLD_CX, _WORLD_CY)

	for id_a in ids:
		var na : Dictionary = _nodes[id_a]
		var force := Vector2.ZERO

		# Repulsion
		for id_b in ids:
			if id_a == id_b:
				continue
			var nb    : Dictionary = _nodes[id_b]
			var diff  : Vector2   = (na["pos"] as Vector2) - (nb["pos"] as Vector2)
			var dist  : float     = maxf(diff.length(), MIN_DIST)
			force += diff.normalized() * (REPULSION / (dist * dist))

		# Springs
		if _spring_map.has(id_a):
			for id_b in (_spring_map[id_a] as Array):
				if not _nodes.has(id_b):
					continue
				var nb   : Dictionary = _nodes[id_b]
				var diff : Vector2   = (nb["pos"] as Vector2) - (na["pos"] as Vector2)
				var dist : float     = maxf(diff.length(), MIN_DIST)
				force += diff.normalized() * ((dist - SPRING_LEN) * SPRING_K)

		# Gravity toward world centre
		force += (center - (na["pos"] as Vector2)) * GRAVITY

		na["vel"] = ((na["vel"] as Vector2) + force) * DAMPING
		na["pos"] = (na["pos"] as Vector2) + (na["vel"] as Vector2)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

func _draw() -> void:
	var font := ThemeDB.fallback_font

	if _nodes.is_empty():
		draw_string(font, size / 2.0 - Vector2(80, 6), "No graph loaded",
		            HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.35, 0.35, 0.40))
		return

	# Edges
	for e in _edges:
		var src : String = str(e["source"])
		var tgt : String = str(e["target"])
		if not (_nodes.has(src) and _nodes.has(tgt)):
			continue
		var sp    : Vector2 = _w2s(_nodes[src]["pos"] as Vector2)
		var tp    : Vector2 = _w2s(_nodes[tgt]["pos"] as Vector2)
		var w     : float   = float(e["weight"])
		var lbl   : String  = str(e["label"])
		var color : Color   = _EDGE_COLOR.get(lbl, _DEFAULT_EDGE_COLOR)
		color.a = 0.45 + w * 0.45
		var line_w : float = maxf(0.8, w * 1.8) * _zoom
		var edge_style : String = _EDGE_STYLE.get(lbl, "solid")
		if edge_style == "dashed":
			_draw_dashed_line(sp, tp, color, line_w, 8.0 * _zoom)
		elif edge_style == "dotted":
			_draw_dashed_line(sp, tp, color, line_w, 3.0 * _zoom)
		else:
			draw_line(sp, tp, color, line_w)

	# Nodes
	for nid in _nodes:
		var n      : Dictionary = _nodes[nid]
		var pos    : Vector2   = _w2s(n["pos"] as Vector2)
		var conf   : float     = float(n["confidence"])
		var radius : float     = (NODE_R + conf * 5.0) * _zoom
		var color  : Color     = _NODE_COLOR.get(n["type"] as String, _DEFAULT_NODE_COLOR)

		# Selection ring
		if nid == _selected_id:
			draw_circle(pos, radius + 4.0 * _zoom, Color(1.0, 1.0, 1.0, 0.25))

		var shape : String = _NODE_SHAPE.get(n["type"] as String, "circle")
		_draw_node_shape(pos, radius, color, shape)

		# STI glow for high-attention nodes
		if float(n["sti"]) > 0.5:
			draw_circle(pos, radius * 1.35, Color(color.r, color.g, color.b, 0.12))

		# Label
		if _zoom >= 0.45:
			var lbl    : String = (n["label"] as String).left(18)
			var fsize  : int    = clamp(int(9.5 * _zoom), 8, 13)
			draw_string(font, pos + Vector2(radius + 3.0 * _zoom, 4.0),
			            lbl, HORIZONTAL_ALIGNMENT_LEFT, -1, fsize,
			            Color(0.85, 0.88, 0.92, 0.90))


# ---------------------------------------------------------------------------
# Input — pan, zoom, select
# ---------------------------------------------------------------------------

func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		var mb : InputEventMouseButton = event as InputEventMouseButton
		match mb.button_index:
			MOUSE_BUTTON_WHEEL_UP:
				if mb.pressed:
					_zoom = minf(_zoom * 1.12, 6.0)
					queue_redraw()
			MOUSE_BUTTON_WHEEL_DOWN:
				if mb.pressed:
					_zoom = maxf(_zoom / 1.12, 0.08)
					queue_redraw()
			MOUSE_BUTTON_MIDDLE:
				_dragging  = mb.pressed
				_drag_start = mb.position
				_cam_start  = _cam_offset
			MOUSE_BUTTON_LEFT:
				if mb.pressed:
					_try_select(mb.position)

	elif event is InputEventMouseMotion:
		if _dragging:
			var mm : InputEventMouseMotion = event as InputEventMouseMotion
			_cam_offset = _cam_start + (mm.position - _drag_start)
			queue_redraw()


func _try_select(screen_pos: Vector2) -> void:
	var world_pos : Vector2 = _s2w(screen_pos)
	var best_id   : String  = ""
	var best_dist : float   = 9999.0

	for nid in _nodes:
		var n      : Dictionary = _nodes[nid]
		var radius : float      = NODE_R + float(n["confidence"]) * 5.0
		var dist   : float      = (n["pos"] as Vector2).distance_to(world_pos)
		if dist <= radius + 5.0 and dist < best_dist:
			best_dist = dist
			best_id   = nid

	_selected_id = best_id
	if best_id != "":
		node_selected.emit(best_id, _nodes[best_id])
	queue_redraw()


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

func _w2s(world_pos: Vector2) -> Vector2:
	return (world_pos - Vector2(_WORLD_CX, _WORLD_CY)) * _zoom + size * 0.5 + _cam_offset


func _s2w(screen_pos: Vector2) -> Vector2:
	return (screen_pos - size * 0.5 - _cam_offset) / _zoom + Vector2(_WORLD_CX, _WORLD_CY)


# ---------------------------------------------------------------------------
# Shape drawing helpers
# ---------------------------------------------------------------------------

func _draw_node_shape(pos: Vector2, radius: float, color: Color, shape: String) -> void:
	match shape:
		"diamond":
			var pts := PackedVector2Array([
				pos + Vector2(0, -radius),
				pos + Vector2(radius, 0),
				pos + Vector2(0, radius),
				pos + Vector2(-radius, 0),
			])
			draw_colored_polygon(pts, color.darkened(0.40))
			for i in pts.size():
				draw_line(pts[i], pts[(i + 1) % pts.size()], color, maxf(1.0, _zoom * 0.8))
		"square":
			var half := radius * 0.85
			var pts := PackedVector2Array([
				pos + Vector2(-half, -half),
				pos + Vector2(half, -half),
				pos + Vector2(half, half),
				pos + Vector2(-half, half),
			])
			draw_colored_polygon(pts, color.darkened(0.40))
			for i in pts.size():
				draw_line(pts[i], pts[(i + 1) % pts.size()], color, maxf(1.0, _zoom * 0.8))
		"hexagon":
			var pts := PackedVector2Array()
			for i in 6:
				var angle := i * TAU / 6.0 - PI / 6.0
				pts.append(pos + Vector2(cos(angle), sin(angle)) * radius)
			draw_colored_polygon(pts, color.darkened(0.40))
			for i in pts.size():
				draw_line(pts[i], pts[(i + 1) % pts.size()], color, maxf(1.0, _zoom * 0.8))
		_:  # circle (default)
			draw_circle(pos, radius, color.darkened(0.40))
			draw_arc(pos, radius, 0.0, TAU, 28, color, maxf(1.0, _zoom * 0.8))


func _draw_dashed_line(from: Vector2, to: Vector2, color: Color, width: float, dash_len: float) -> void:
	var dir   := (to - from)
	var total := dir.length()
	if total < 1.0:
		return
	var norm  := dir / total
	var drawn := 0.0
	var on    := true
	while drawn < total:
		var segment := minf(dash_len, total - drawn)
		if on:
			draw_line(from + norm * drawn, from + norm * (drawn + segment), color, width)
		drawn += segment
		on = not on

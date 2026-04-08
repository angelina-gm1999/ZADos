##
## NeurochemTab — Tab 2: rolling turn-by-turn neurochem log.
##
## Each turn entry shows:
##   - turn index + mode token
##   - NT snapshot as a color-gradient strip (blue=low → white=mid → red=high)
##   - Oscillatory bands as a mini bar row
##   - Key metrics as labeled value pairs
##   Clicking an entry expands the full NT value list.
##
class_name NeurochemTab
extends ScrollContainer

# NT groups in display order (spec §1.3 Tab 2)
const NT_GROUPS := [
	["da",  "DA",  Color(0.94, 0.75, 0.13)],
	["5ht", "5HT", Color(0.18, 0.75, 0.72)],
	["ne",  "NE",  Color(0.95, 0.52, 0.12)],
	["ach", "ACh", Color(0.55, 0.85, 0.55)],
	["mor", "MOR", Color(0.90, 0.45, 0.65)],
	["cb1", "CB1", Color(0.60, 0.80, 0.30)],
	["crh", "CRH", Color(0.80, 0.35, 0.35)],
	["cor", "COR", Color(0.75, 0.40, 0.20)],
	["gaba","GABA",Color(0.26, 0.55, 0.95)],
	["glu", "GLU", Color(0.90, 0.25, 0.25)],
]

const METRIC_KEYS := [
	"motivation", "empathy", "cognitive_rigidity",
	"fatigue", "precision", "openness", "anxiety", "social_engagement",
]

const SLEEP_METRIC_KEYS := [
	"dream_permissiveness", "consolidation_depth", "narrative_plasticity",
]

var _list   : VBoxContainer
var _entries: Array = []   # Array of Dictionaries for each turn

# ---------------------------------------------------------------------------

func _ready() -> void:
	horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	_list = VBoxContainer.new()
	_list.size_flags_horizontal = SIZE_EXPAND_FILL
	_list.add_theme_constant_override("separation", 4)
	add_child(_list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

func add_turn(turn_index: int, modulation: Dictionary) -> void:
	_entries.push_back({"turn": turn_index, "modulation": modulation})
	_build_entry(turn_index, modulation)


# ---------------------------------------------------------------------------
# Build a single turn entry
# ---------------------------------------------------------------------------

func _build_entry(turn_index: int, mod: Dictionary) -> void:
	var entry := VBoxContainer.new()
	entry.add_theme_constant_override("separation", 3)

	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.10, 0.10, 0.13)
	style.set_corner_radius_all(4)
	style.content_margin_left   = 6
	style.content_margin_right  = 6
	style.content_margin_top    = 4
	style.content_margin_bottom = 4

	var panel := PanelContainer.new()
	panel.add_theme_stylebox_override("panel", style)
	panel.add_child(entry)
	_list.add_child(panel)

	# Header row: turn # + mode token
	var header := HBoxContainer.new()
	entry.add_child(header)

	var turn_lbl := Label.new()
	turn_lbl.text = "Turn %d" % turn_index
	turn_lbl.add_theme_font_size_override("font_size", 10)
	turn_lbl.add_theme_color_override("font_color", Color(0.50, 0.50, 0.55))
	header.add_child(turn_lbl)

	var spacer := Control.new()
	spacer.size_flags_horizontal = SIZE_EXPAND_FILL
	header.add_child(spacer)

	var mode_lbl := Label.new()
	mode_lbl.text = mod.get("mode_token", "Normal")
	mode_lbl.add_theme_font_size_override("font_size", 10)
	mode_lbl.add_theme_color_override("font_color", Color(0.65, 0.65, 0.45))
	header.add_child(mode_lbl)

	# NT strip
	var nt_snap : Dictionary = mod.get("nt_snapshot", {})
	_build_nt_strip(entry, nt_snap)

	# Oscillatory bands
	var osc_snap : Dictionary = mod.get("osc_snapshot", {})
	if not osc_snap.is_empty():
		_build_osc_row(entry, osc_snap)

	# Metrics
	var metrics : Dictionary = mod.get("metrics_dict", {})
	if not metrics.is_empty():
		_build_metrics_row(entry, metrics)

	# Sleep metrics (grayed out during waking)
	_build_sleep_metrics_row(entry, metrics)

	# Expandable detail: full NT values + differential from previous turn
	var detail := VBoxContainer.new()
	detail.visible = false
	entry.add_child(detail)
	_build_detail(detail, nt_snap, turn_index)
	panel.gui_input.connect(func(ev):
		if ev is InputEventMouseButton and ev.pressed and ev.button_index == MOUSE_BUTTON_LEFT:
			detail.visible = not detail.visible)


func _build_nt_strip(parent: VBoxContainer, nt_snap: Dictionary) -> void:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 1)
	parent.add_child(row)

	for group in NT_GROUPS:
		var key      : String = group[0]
		var nt_name  : String = group[1]
		var color    : Color  = group[2]
		var val      : float  = nt_snap.get(key, 0.5) as float

		var seg := ColorRect.new()
		seg.custom_minimum_size = Vector2(16, 14)
		seg.size_flags_horizontal = SIZE_EXPAND_FILL
		# Interpolate: blue(low) → white(mid) → color(high)
		if val < 0.5:
			seg.color = Color(0.20, 0.30, 0.75).lerp(Color(0.80, 0.80, 0.80), val * 2.0)
		else:
			seg.color = Color(0.80, 0.80, 0.80).lerp(color, (val - 0.5) * 2.0)
		seg.tooltip_text = "%s: %.2f" % [nt_name, val]
		row.add_child(seg)


func _build_osc_row(parent: VBoxContainer, osc: Dictionary) -> void:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 3)
	parent.add_child(row)

	for band in ["delta", "theta", "alpha", "sigma", "beta", "gamma"]:
		var val : float = osc.get(band, 0.0) as float
		var lbl := Label.new()
		lbl.text = band[0].to_upper() + ":%.2f" % val
		lbl.add_theme_font_size_override("font_size", 9)
		lbl.add_theme_color_override("font_color", Color(0.45, 0.55, 0.65))
		row.add_child(lbl)


func _build_metrics_row(parent: VBoxContainer, metrics: Dictionary) -> void:
	var grid := GridContainer.new()
	grid.columns = 4
	grid.add_theme_constant_override("h_separation", 8)
	grid.add_theme_constant_override("v_separation", 2)
	parent.add_child(grid)

	for key in METRIC_KEYS:
		if not metrics.has(key):
			continue
		var val : float = metrics[key] as float

		var lbl := Label.new()
		lbl.text = "%s %.2f" % [key.left(8), val]
		lbl.add_theme_font_size_override("font_size", 9)
		lbl.add_theme_color_override("font_color", Color(0.50, 0.55, 0.60))
		grid.add_child(lbl)


func _build_sleep_metrics_row(parent: VBoxContainer, metrics: Dictionary) -> void:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	parent.add_child(row)
	for key in SLEEP_METRIC_KEYS:
		var val : float = metrics.get(key, 0.0) as float
		var lbl := Label.new()
		lbl.text = "%s: %.2f" % [key.left(12), val]
		lbl.add_theme_font_size_override("font_size", 9)
		# Grayed out during waking (value near 0)
		lbl.add_theme_color_override("font_color", Color(0.30, 0.30, 0.35) if val < 0.01 else Color(0.55, 0.45, 0.70))
		row.add_child(lbl)


func _build_detail(parent: VBoxContainer, nt_snap: Dictionary, _turn_index: int) -> void:
	# Full NT values
	var grid := GridContainer.new()
	grid.columns = 4
	grid.add_theme_constant_override("h_separation", 8)
	grid.add_theme_constant_override("v_separation", 2)
	parent.add_child(grid)

	# Get previous turn's NT snapshot for differential
	var prev_snap : Dictionary = {}
	if _entries.size() >= 2:
		var prev_mod : Dictionary = _entries[_entries.size() - 2].get("modulation", {})
		prev_snap = prev_mod.get("nt_snapshot", {})

	for group in NT_GROUPS:
		var key : String = group[0]
		var nt_name : String = group[1]
		var val : float = nt_snap.get(key, 0.5) as float
		var prev_val : float = prev_snap.get(key, val) as float
		var diff : float = val - prev_val

		var lbl := Label.new()
		var diff_str := ""
		if abs(diff) > 0.001:
			diff_str = " (%+.2f)" % diff
		lbl.text = "%s: %.3f%s" % [nt_name, val, diff_str]
		lbl.add_theme_font_size_override("font_size", 9)
		var col := Color(0.55, 0.60, 0.65)
		if diff > 0.05:
			col = Color(0.45, 0.80, 0.45)
		elif diff < -0.05:
			col = Color(0.80, 0.45, 0.45)
		lbl.add_theme_color_override("font_color", col)
		grid.add_child(lbl)

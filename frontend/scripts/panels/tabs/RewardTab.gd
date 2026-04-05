##
## RewardTab — Tab 1 of the Nerd Stats panel.
## Shows mode token, reward profile, domain weights, and directive.
##
class_name RewardTab
extends ScrollContainer

const MODE_COLORS := {
	"Normal":       Color(0.55, 0.55, 0.55),
	"Learning":     Color(0.30, 0.55, 0.95),
	"SleepMode":    Color(0.45, 0.40, 0.85),
	"Dream":        Color(0.65, 0.35, 0.85),
	"Homework":     Color(0.90, 0.65, 0.20),
	"SelfReflective": Color(0.25, 0.75, 0.72),
}

const DOMAIN_COLORS := {
	"logic":       Color(0.35, 0.65, 0.95),
	"ethics":      Color(0.35, 0.88, 0.55),
	"innovation":  Color(0.95, 0.70, 0.25),
	"attunement":  Color(0.80, 0.45, 0.90),
}

var _content       : VBoxContainer
var _mode_label    : Label
var _profile_label : Label
var _domain_bars   : Dictionary = {}  # domain → ProgressBar
var _directive_lbl : Label
var _urgency_bar   : ProgressBar

# ---------------------------------------------------------------------------

func _ready() -> void:
	horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	_build_ui()


func _build_ui() -> void:
	_content = VBoxContainer.new()
	_content.size_flags_horizontal = SIZE_EXPAND_FILL
	_content.add_theme_constant_override("separation", 8)
	add_child(_content)

	_section("MODE")
	_mode_label = _add_value_label("—", 18)
	_mode_label.add_theme_color_override("font_color", Color(0.55, 0.55, 0.55))

	_section("REWARD PROFILE")
	_profile_label = _add_value_label("—", 13)

	_section("DOMAIN WEIGHTS")
	for domain in ["logic", "ethics", "innovation", "attunement"]:
		var row := HBoxContainer.new()
		_content.add_child(row)

		var lbl := Label.new()
		lbl.text = domain
		lbl.custom_minimum_size = Vector2(80, 0)
		lbl.add_theme_font_size_override("font_size", 11)
		lbl.add_theme_color_override("font_color", DOMAIN_COLORS.get(domain, Color.WHITE))
		row.add_child(lbl)

		var bar := ProgressBar.new()
		bar.max_value             = 1.0
		bar.value                 = 0.5
		bar.size_flags_horizontal = SIZE_EXPAND_FILL
		bar.custom_minimum_size   = Vector2(0, 14)
		bar.show_percentage       = false
		row.add_child(bar)
		_domain_bars[domain] = bar

	_section("DIRECTIVE")
	_directive_lbl = _add_value_label("—", 13)

	_section("URGENCY RISK")
	_urgency_bar = ProgressBar.new()
	_urgency_bar.max_value           = 1.0
	_urgency_bar.value               = 0.0
	_urgency_bar.size_flags_horizontal = SIZE_EXPAND_FILL
	_urgency_bar.custom_minimum_size  = Vector2(0, 14)
	_urgency_bar.show_percentage      = true
	_content.add_child(_urgency_bar)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

func refresh(result: Dictionary) -> void:
	var state   : Dictionary = result.get("state", {})
	var mod     : Dictionary = state.get("modulation", {})
	var reward  : Dictionary = state.get("reward", {})

	# Mode
	var mode : String = mod.get("mode_token", "Normal")
	_mode_label.text = mode
	var mode_color := Color(0.55, 0.55, 0.55)
	for prefix in MODE_COLORS:
		if mode.begins_with(prefix):
			mode_color = MODE_COLORS[prefix]
			break
	_mode_label.add_theme_color_override("font_color", mode_color)

	# Profile
	_profile_label.text = mod.get("reward_profile_name", "—")

	# Domain weights from engine_weights
	var weights : Dictionary = mod.get("engine_weights", {})
	for domain in _domain_bars:
		var bar : ProgressBar = _domain_bars[domain]
		bar.value = weights.get(domain + "_weight", 0.5) as float

	# Directive
	var directive : String = result.get("directive", state.get("answer", {}).get("directive_applied", "—"))
	_directive_lbl.text = directive.to_upper()
	match directive.to_lower():
		"allow":    _directive_lbl.add_theme_color_override("font_color", Color(0.20, 0.85, 0.40))
		"suppress": _directive_lbl.add_theme_color_override("font_color", Color(0.90, 0.25, 0.25))
		"abstain":  _directive_lbl.add_theme_color_override("font_color", Color(0.95, 0.75, 0.15))

	# Urgency risk from phase5 data (nested inside reward dict)
	var p5 : Dictionary = reward.get("phase5_result", {})
	var urgency : float = (p5.get("urgency_risk", p5.get("tonic", {}).get("urgency_risk", 0.0)) as float)
	_urgency_bar.value = urgency


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

func _section(title: String) -> void:
	var lbl := Label.new()
	lbl.text = title
	lbl.add_theme_font_size_override("font_size", 9)
	lbl.add_theme_color_override("font_color", Color(0.40, 0.40, 0.45))
	_content.add_child(lbl)


func _add_value_label(text: String, font_size: int) -> Label:
	var lbl := Label.new()
	lbl.text = text
	lbl.add_theme_font_size_override("font_size", font_size)
	lbl.add_theme_color_override("font_color", Color(0.80, 0.82, 0.86))
	_content.add_child(lbl)
	return lbl

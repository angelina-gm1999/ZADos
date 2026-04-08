##
## LTMMPanel — Long-Term Memory browser.
## 5 sub-tabs: Journal | Thoughts | Knowledge | Identity | Unsolved
##
extends VBoxContainer

const _JournalPanel  = preload("res://scripts/panels/memory/ltmm/JournalPanel.gd")
const _ThoughtsPanel = preload("res://scripts/panels/memory/ltmm/ThoughtsPanel.gd")
const _KnowledgePanel= preload("res://scripts/panels/memory/ltmm/KnowledgePanel.gd")
const _IdentityPanel = preload("res://scripts/panels/memory/ltmm/IdentityPanel.gd")
const _UnsolvedPanel = preload("res://scripts/panels/memory/ltmm/UnsolvedPanel.gd")

var _tabs : TabContainer
var _panels : Array = []

func _ready() -> void:
	add_theme_constant_override("separation", 0)
	_build_ui()


func _build_ui() -> void:
	_tabs = TabContainer.new()
	_tabs.size_flags_vertical = SIZE_EXPAND_FILL
	_tabs.add_theme_constant_override("side_margin", 0)
	add_child(_tabs)

	var journal   : Control = _JournalPanel.new()
	journal.name  = "Journal"
	var thoughts  : Control = _ThoughtsPanel.new()
	thoughts.name = "Thoughts"
	var knowledge : Control = _KnowledgePanel.new()
	knowledge.name = "Knowledge"
	var identity  : Control = _IdentityPanel.new()
	identity.name = "Identity"
	var unsolved  : Control = _UnsolvedPanel.new()
	unsolved.name = "Unsolved"

	_panels = [journal, thoughts, knowledge, identity, unsolved]
	for p in _panels:
		_tabs.add_child(p)

	_tabs.tab_changed.connect(_on_tab_changed)


func refresh() -> void:
	var current := _tabs.get_current_tab()
	if current >= 0 and current < _panels.size():
		var p : Control = _panels[current]
		if p.has_method("refresh"):
			p.refresh()


func _on_tab_changed(tab: int) -> void:
	if tab >= 0 and tab < _panels.size():
		var p : Control = _panels[tab]
		if p.has_method("refresh"):
			p.refresh()

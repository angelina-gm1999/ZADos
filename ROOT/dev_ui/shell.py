"""ZadosShell — the cmd.Cmd-based REPL.

v0.1 scope (build plan §13 step 1+2):
    chat send / last / history / show / clear
    set verbosity {quiet|normal|nerd}
    set autoshow {on|off}
    sess status / close
    quit

Bare lines that don't start with a known command are treated as
`chat send "<line>"`.
"""
from __future__ import annotations

import cmd
import shlex
import traceback
from typing import Optional

from rich.console import Console
from rich.text import Text

from dev_ui.dev_session import DevSession, Verbosity
from dev_ui.render import (
    render_answer_panel,
    render_error_panel,
    render_history_table,
    render_status_line,
    render_turn_block,
    render_turn_detail,
    unwrap_pipeline_result,
)
from dev_ui.render_show import (
    render_classification,
    render_engine_inspector,
    render_engines,
    render_neurochem,
    render_perception,
    render_reward,
    render_thinking,
)
from dev_ui.render_mem import (
    render_log_entries,
    render_logs_overview,
    render_ltmm_overview,
    render_ltmm_store_list,
    render_ltmm_store_show,
    render_mtmm_packet,
    render_mtmm_packets,
    render_mtmm_trends,
    render_stmm_current,
    render_stmm_tracker,
    resolve_ltmm_store,
)
from dev_ui.render_atom import (
    apply_atom_add_link,
    apply_atom_add_node,
    apply_atom_delete,
    apply_atom_set,
    apply_map_export_file,
    apply_map_import_file,
    apply_map_load,
    apply_map_save,
    render_atom_list,
    render_atom_search,
    render_atom_show,
    render_atom_status,
    render_map_list,
)
from dev_ui.render_mode import render_mode_list, render_mode_show
from dev_ui.render_commanded import (
    render_dream_result,
    render_homework_result,
    render_reflective_result,
    render_rem_result,
    render_sleep_status,
)
from dev_ui.render_dev import (
    apply_nt_reset,
    apply_nt_set,
    apply_reward_override,
    apply_reward_reset,
    render_nt_metrics_only,
    render_nt_state,
    render_pipeline_dispatch,
    render_pipeline_error_detail,
    render_pipeline_errors,
    render_pipeline_last,
    render_reward_learned,
    render_reward_map,
    render_reward_profile_detail,
    render_reward_profiles,
)


_KNOWN_COMMANDS = {
    "chat", "show", "mem", "mode", "dev", "nt", "sleep",
    "atom", "map",
    "set", "sess", "help", "quit", "exit", "EOF", "?",
}


def _extract_turn_flag(tokens: list[str]) -> Optional[int]:
    """Pull `--turn N` from a token list. Returns int or None."""
    for i, t in enumerate(tokens):
        if t == "--turn" and i + 1 < len(tokens):
            try:
                return int(tokens[i + 1])
            except ValueError:
                return None
    return None

_SHOW_SUBCOMMANDS = (
    "reward", "neurochem", "engines", "engine",
    "thinking", "classification", "perception",
)

_VALID_VERBOSITIES = ("quiet", "normal", "nerd")


class ZadosShell(cmd.Cmd):
    intro = (
        "ZADOS dev shell.  Type `help` or `?` to list commands.\n"
        "Bare lines are sent to the AI as `chat send`.  Use `quit` to exit."
    )
    prompt = "> "

    def __init__(self, dev: DevSession) -> None:
        super().__init__()
        self.dev = dev
        self.console = Console()
        self._print_status_line()

    # ------------------------------------------------------------------
    # cmd.Cmd hooks
    # ------------------------------------------------------------------

    def precmd(self, line: str) -> str:
        """Route bare lines to `chat send`."""
        stripped = line.strip()
        if not stripped:
            return line
        first = stripped.split(None, 1)[0]
        if first in _KNOWN_COMMANDS or first.startswith("?"):
            return line
        # Treat as chat input. Escape quotes by using shlex.quote.
        # If the user has staged input, prepend it.
        text = stripped
        if self.dev.staged_input:
            text = f"{self.dev.staged_input}\n{text}"
            self.dev.staged_input = None
        return f"chat send {shlex.quote(text)}"

    def postcmd(self, stop: bool, line: str) -> bool:
        if not stop:
            self._print_status_line()
        return stop

    def emptyline(self) -> bool:  # noqa: D401
        # Do nothing on empty input (default would re-run last cmd).
        return False

    def default(self, line: str) -> None:
        self.console.print(render_error_panel(f"unknown command: {line!r}"))

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------

    def do_chat(self, arg: str) -> Optional[bool]:
        """chat <send|last|history|show|clear> [...]"""
        parts = shlex.split(arg) if arg else []
        if not parts:
            self.console.print("usage: chat <send|last|history|show|clear> ...")
            return None
        sub, rest = parts[0], parts[1:]
        handler = getattr(self, f"_chat_{sub}", None)
        if handler is None:
            self.console.print(render_error_panel(f"unknown `chat` subcommand: {sub}"))
            return None
        try:
            handler(rest)
        except Exception as exc:  # noqa: BLE001
            self.console.print(render_error_panel(f"{type(exc).__name__}: {exc}"))
            self.console.print(Text(traceback.format_exc(), style="dim red"))
        return None

    def _chat_send(self, args: list[str]) -> None:
        if not args:
            self.console.print("usage: chat send <text>")
            return
        # Join everything — shlex already handled quotes.
        text = " ".join(args)

        # Capture classification before processing so `show classification` works
        # even when a pipeline error aborts the turn.
        try:
            from zados.core.types import RawInput
            self.dev.last_classification = self.dev.classifier.classify(RawInput(text=text))
        except Exception:  # noqa: BLE001
            self.dev.last_classification = None

        try:
            result = self.dev.classifier.process_text(text)
        except Exception as exc:  # noqa: BLE001
            self.dev.record_error("chat send", exc)
            self.console.print(render_error_panel(f"pipeline error: {type(exc).__name__}: {exc}"))
            self.console.print(Text(traceback.format_exc(), style="dim red"))
            return

        self.dev.last_result = result
        self.dev.history.append(result)

        if self.dev.autoshow:
            if self.dev.verbosity == "quiet":
                self.console.print(render_answer_panel(result))
            else:
                self.console.print(render_turn_block(result, self.dev.verbosity))

    def _chat_last(self, args: list[str]) -> None:
        if self.dev.last_result is None:
            self.console.print("(no turns yet)")
            return
        self.console.print(render_turn_block(self.dev.last_result, self.dev.verbosity))

    def _chat_history(self, args: list[str]) -> None:
        n = 10
        if args:
            try:
                n = int(args[0])
            except ValueError:
                self.console.print("usage: chat history [N]")
                return
        if not self.dev.history:
            self.console.print("(no turns yet)")
            return
        self.console.print(render_history_table(self.dev.history, n))

    def _chat_show(self, args: list[str]) -> None:
        if not args:
            self.console.print("usage: chat show <turn_idx>")
            return
        try:
            idx = int(args[0])
        except ValueError:
            self.console.print("usage: chat show <turn_idx>")
            return
        if not (0 <= idx < len(self.dev.history)):
            self.console.print(
                render_error_panel(f"turn {idx} out of range (have {len(self.dev.history)})")
            )
            return
        self.console.print(render_turn_detail(self.dev.history[idx]))

    def _chat_clear(self, args: list[str]) -> None:
        n = len(self.dev.history)
        self.dev.history.clear()
        self.dev.last_result = None
        self.console.print(f"(cleared {n} turn(s) from REPL history — memory stores untouched)")

    # ------------------------------------------------------------------
    # show
    # ------------------------------------------------------------------

    def do_show(self, arg: str) -> Optional[bool]:
        """show <reward|neurochem|engines|engine <id>|thinking|classification|perception>
        [--turn N] [--full]
        """
        parts = shlex.split(arg) if arg else []
        if not parts:
            self.console.print(
                "usage: show <" + "|".join(_SHOW_SUBCOMMANDS) + "> [--turn N] [--full]"
            )
            return None

        sub = parts[0]
        flags = parts[1:]
        turn_idx: Optional[int] = None
        full = False
        positional: list[str] = []
        i = 0
        while i < len(flags):
            tok = flags[i]
            if tok == "--turn" and i + 1 < len(flags):
                try:
                    turn_idx = int(flags[i + 1])
                except ValueError:
                    self.console.print(render_error_panel("--turn requires an integer"))
                    return None
                i += 2
            elif tok == "--full":
                full = True
                i += 1
            else:
                positional.append(tok)
                i += 1

        result = self._resolve_turn(turn_idx)

        try:
            if sub == "reward":
                if result is None:
                    self.console.print("(no turns yet — `chat send <text>` first)")
                    return None
                self.console.print(render_reward(result.state, self.dev.session))
            elif sub == "neurochem":
                if result is None:
                    self.console.print("(no turns yet — `chat send <text>` first)")
                    return None
                self.console.print(render_neurochem(result.state, full=full))
            elif sub == "engines":
                if result is None:
                    self.console.print("(no turns yet — `chat send <text>` first)")
                    return None
                self.console.print(render_engines(result.state, self.dev.engines.keys()))
            elif sub == "engine":
                if not positional:
                    self.console.print("usage: show engine <id> [--turn N]")
                    return None
                try:
                    eid = int(positional[0])
                except ValueError:
                    self.console.print(render_error_panel("engine id must be an integer (1-32)"))
                    return None
                if result is None:
                    self.console.print("(no turns yet — `chat send <text>` first)")
                    return None
                self.console.print(render_engine_inspector(result.state, eid))
            elif sub == "thinking":
                if result is None:
                    self.console.print("(no turns yet — `chat send <text>` first)")
                    return None
                self.console.print(render_thinking(result.state))
            elif sub == "classification":
                # If the user provides text, classify it.  Else show last.
                if positional:
                    text = " ".join(positional)
                    try:
                        from zados.core.types import RawInput
                        cls = self.dev.classifier.classify(RawInput(text=text))
                    except Exception as exc:  # noqa: BLE001
                        self.console.print(render_error_panel(
                            f"classify failed: {type(exc).__name__}: {exc}"
                        ))
                        return None
                    self.console.print(render_classification(cls))
                else:
                    self.console.print(render_classification(self.dev.last_classification))
            elif sub == "perception":
                if result is None:
                    self.console.print("(no turns yet — `chat send <text>` first)")
                    return None
                self.console.print(render_perception(result.state))
            else:
                self.console.print(render_error_panel(
                    f"unknown `show` subcommand: {sub}.  Try: {', '.join(_SHOW_SUBCOMMANDS)}"
                ))
        except Exception as exc:  # noqa: BLE001
            self.console.print(render_error_panel(
                f"render failed: {type(exc).__name__}: {exc}"
            ))
            self.console.print(Text(traceback.format_exc(), style="dim red"))
        return None

    def _resolve_turn(self, idx: Optional[int]):
        """Return a PipelineResult (unwrapped) for the requested turn, or None.

        Falls back gracefully when the underlying result has no PipelineState
        (e.g. /sleep returns a dict).  In that case prints a hint and returns None.
        """
        if idx is None:
            raw = self.dev.last_result
        else:
            if not (0 <= idx < len(self.dev.history)):
                self.console.print(
                    render_error_panel(f"turn {idx} out of range (have {len(self.dev.history)})")
                )
                return None
            raw = self.dev.history[idx]
        if raw is None:
            return None
        unwrapped = unwrap_pipeline_result(raw)
        if unwrapped is None:
            self.console.print(
                f"(turn produced a {type(raw).__name__} with no PipelineState — "
                f"no pipeline detail to show)"
            )
            return None
        return unwrapped

    # ------------------------------------------------------------------
    # mem
    # ------------------------------------------------------------------

    def do_mem(self, arg: str) -> Optional[bool]:
        """mem <stmm|mtmm|ltmm> ...

        stmm:
          mem stmm current
          mem stmm tracker
        mtmm:
          mem mtmm packets [N]
          mem mtmm packet <id>
          mem mtmm trends
        ltmm:
          mem ltmm                                       (namespace overview)
          mem ltmm <namespace>.<store> [list [N]]
          mem ltmm <namespace>.<store> show <id>
        """
        parts = shlex.split(arg) if arg else []
        if not parts:
            self.console.print("usage: mem <stmm|mtmm|ltmm> ...")
            return None
        tier = parts[0]
        rest = parts[1:]
        try:
            if tier == "stmm":
                self._mem_stmm(rest)
            elif tier == "mtmm":
                self._mem_mtmm(rest)
            elif tier == "ltmm":
                self._mem_ltmm(rest)
            elif tier == "logs":
                self._mem_logs(rest)
            else:
                self.console.print(render_error_panel(
                    f"unknown memory tier: {tier!r} (stmm | mtmm | ltmm | logs)"
                ))
        except Exception as exc:  # noqa: BLE001
            self.console.print(render_error_panel(
                f"render failed: {type(exc).__name__}: {exc}"
            ))
            self.console.print(Text(traceback.format_exc(), style="dim red"))
        return None

    def _mem_stmm(self, args: list[str]) -> None:
        sub = args[0] if args else "current"
        stmm = self.dev.memory.stmm
        if sub == "current":
            self.console.print(render_stmm_current(stmm))
        elif sub == "tracker":
            self.console.print(render_stmm_tracker(stmm))
        else:
            self.console.print(render_error_panel(
                f"unknown stmm subcommand: {sub} (current | tracker)"
            ))

    def _mem_mtmm(self, args: list[str]) -> None:
        if not args:
            args = ["packets"]
        sub = args[0]
        mtmm = self.dev.memory.mtmm
        if sub == "packets":
            n = 10
            if len(args) > 1:
                try:
                    n = int(args[1])
                except ValueError:
                    self.console.print("usage: mem mtmm packets [N]")
                    return
            self.console.print(render_mtmm_packets(mtmm, n))
        elif sub == "packet":
            if len(args) < 2:
                self.console.print("usage: mem mtmm packet <id>")
                return
            self.console.print(render_mtmm_packet(mtmm, args[1]))
        elif sub == "trends":
            self.console.print(render_mtmm_trends(mtmm))
        else:
            self.console.print(render_error_panel(
                f"unknown mtmm subcommand: {sub} (packets | packet | trends)"
            ))

    def _mem_ltmm(self, args: list[str]) -> None:
        if not args:
            self.console.print(render_ltmm_overview(self.dev.memory))
            return
        dotted = args[0]
        rest = args[1:]
        store, err = resolve_ltmm_store(self.dev.memory, dotted)
        if err:
            self.console.print(render_error_panel(err))
            return
        sub = rest[0] if rest else "list"
        if sub == "list":
            n = 20
            if len(rest) > 1:
                try:
                    n = int(rest[1])
                except ValueError:
                    self.console.print("usage: mem ltmm <ns>.<store> list [N]")
                    return
            self.console.print(render_ltmm_store_list(store, dotted, n))
        elif sub == "show":
            if len(rest) < 2:
                self.console.print(f"usage: mem ltmm {dotted} show <id>")
                return
            self.console.print(render_ltmm_store_show(store, dotted, rest[1]))
        else:
            self.console.print(render_error_panel(
                f"unknown ltmm subcommand: {sub} (list | show)"
            ))

    def _mem_logs(self, args: list[str]) -> None:
        if not args:
            self.console.print(render_logs_overview(self.dev.memory))
            return
        name = args[0]
        n = 20
        if len(args) > 1:
            try:
                n = int(args[1])
            except ValueError:
                self.console.print(f"usage: mem logs {name} [N]")
                return
        self.console.print(render_log_entries(self.dev.memory, name, n))

    # ------------------------------------------------------------------
    # atom
    # ------------------------------------------------------------------

    def do_atom(self, arg: str) -> Optional[bool]:
        """atom <list|show|search|status|add|link|set|delete>

        list   [--type T] [--name SUBSTR] [N]
        show   <id>
        search <substring> [N]
        status                                  (AtomSpace summary + type histogram)
        add node <Type> <name> [--strength S --confidence C]
        add link <Type> <id1> <id2> [...] [--strength S --confidence C]
        set    <id> [--strength S --confidence C --sti N --lti N]
        delete <id>
        """
        parts = shlex.split(arg) if arg else []
        if not parts:
            self.console.print("usage: atom <list|show|search|status|add|link|set|delete> ...")
            return None
        sub = parts[0]
        rest = parts[1:]
        atomspace = self.dev.engines.get(9)
        try:
            if sub == "list":
                self._atom_list(atomspace, rest)
            elif sub == "show":
                if not rest:
                    self.console.print("usage: atom show <id>")
                    return None
                self.console.print(render_atom_show(atomspace, rest[0]))
            elif sub == "search":
                if not rest:
                    self.console.print("usage: atom search <substring> [N]")
                    return None
                n = 30
                if len(rest) > 1:
                    try:
                        n = int(rest[1])
                    except ValueError:
                        pass
                self.console.print(render_atom_search(atomspace, rest[0], n))
            elif sub == "status":
                self.console.print(render_atom_status(atomspace))
            elif sub == "add":
                self._atom_add(atomspace, rest)
            elif sub == "link":
                self._atom_link(atomspace, rest)
            elif sub == "set":
                self._atom_set(atomspace, rest)
            elif sub == "delete":
                if not rest:
                    self.console.print("usage: atom delete <id>")
                    return None
                self.console.print(apply_atom_delete(atomspace, rest[0]))
            else:
                self.console.print(render_error_panel(f"unknown atom subcommand: {sub}"))
        except Exception as exc:  # noqa: BLE001
            self.dev.record_error(f"atom {sub}", exc)
            self.console.print(render_error_panel(
                f"atom failed: {type(exc).__name__}: {exc}"
            ))
            self.console.print(Text(traceback.format_exc(), style="dim red"))
        return None

    def _atom_list(self, atomspace: Any, args: list[str]) -> None:
        type_filter: Optional[str] = None
        name_filter: Optional[str] = None
        limit = 30
        i = 0
        positional: list[str] = []
        while i < len(args):
            t = args[i]
            if t == "--type" and i + 1 < len(args):
                type_filter = args[i + 1]; i += 2
            elif t == "--name" and i + 1 < len(args):
                name_filter = args[i + 1]; i += 2
            else:
                positional.append(t); i += 1
        if positional:
            try:
                limit = int(positional[0])
            except ValueError:
                pass
        self.console.print(render_atom_list(atomspace, type_filter, name_filter, limit))

    def _atom_add(self, atomspace: Any, args: list[str]) -> None:
        # form: atom add node <Type> <name> [--strength S --confidence C]
        if not args or args[0] != "node" or len(args) < 3:
            self.console.print("usage: atom add node <Type> <name> [--strength S --confidence C]")
            return
        atom_type = args[1]
        # name may contain spaces; take everything up to first flag
        name_tokens: list[str] = []
        i = 2
        while i < len(args) and not args[i].startswith("--"):
            name_tokens.append(args[i]); i += 1
        name = " ".join(name_tokens)
        flags = self._parse_weight_flags(args[i:])
        s = flags.get("strength", 1.0)
        c = flags.get("confidence", 0.9)
        self.console.print(apply_atom_add_node(atomspace, atom_type, name, s, c))

    def _atom_link(self, atomspace: Any, args: list[str]) -> None:
        # form: atom link <Type> <id1> <id2> [...] [--strength S --confidence C]
        if len(args) < 3:
            self.console.print(
                "usage: atom link <Type> <id1> <id2> [...] [--strength S --confidence C]"
            )
            return
        link_type = args[0]
        ids: list[str] = []
        i = 1
        while i < len(args) and not args[i].startswith("--"):
            ids.append(args[i]); i += 1
        flags = self._parse_weight_flags(args[i:])
        s = flags.get("strength", 1.0)
        c = flags.get("confidence", 0.9)
        self.console.print(apply_atom_add_link(atomspace, link_type, ids, s, c))

    def _atom_set(self, atomspace: Any, args: list[str]) -> None:
        if not args:
            self.console.print(
                "usage: atom set <id> [--strength S --confidence C --sti N --lti N]"
            )
            return
        atom_id = args[0]
        flags = self._parse_weight_flags(args[1:])
        self.console.print(apply_atom_set(
            atomspace, atom_id,
            strength=flags.get("strength"),
            confidence=flags.get("confidence"),
            sti=flags.get("sti"),
            lti=flags.get("lti"),
        ))

    # ------------------------------------------------------------------
    # map
    # ------------------------------------------------------------------

    def do_map(self, arg: str) -> Optional[bool]:
        """map <list|save|load|export|import>

        list                  — saved knowledge maps
        save <name> [desc]    — snapshot AtomSpace → KnowledgeMapStore
        load <name|id>        — restore AtomSpace from a saved map
        export <path>         — dump AtomSpace to a JSON file
        import <path>         — load a JSON file into the AtomSpace
        """
        parts = shlex.split(arg) if arg else []
        if not parts:
            self.console.print("usage: map <list|save|load|export|import> ...")
            return None
        sub = parts[0]
        rest = parts[1:]
        atomspace = self.dev.engines.get(9)
        try:
            if sub == "list":
                self.console.print(render_map_list(self.dev.memory))
            elif sub == "save":
                if not rest:
                    self.console.print("usage: map save <name> [description]")
                    return None
                name = rest[0]
                desc = " ".join(rest[1:]) if len(rest) > 1 else ""
                self.console.print(apply_map_save(self.dev.memory, atomspace, name, desc))
            elif sub == "load":
                if not rest:
                    self.console.print("usage: map load <name|id>")
                    return None
                self.console.print(apply_map_load(self.dev.memory, atomspace, rest[0]))
            elif sub == "export":
                if not rest:
                    self.console.print("usage: map export <path>")
                    return None
                self.console.print(apply_map_export_file(atomspace, rest[0]))
            elif sub == "import":
                if not rest:
                    self.console.print("usage: map import <path>")
                    return None
                self.console.print(apply_map_import_file(atomspace, rest[0]))
            else:
                self.console.print(render_error_panel(f"unknown map subcommand: {sub}"))
        except Exception as exc:  # noqa: BLE001
            self.dev.record_error(f"map {sub}", exc)
            self.console.print(render_error_panel(
                f"map failed: {type(exc).__name__}: {exc}"
            ))
            self.console.print(Text(traceback.format_exc(), style="dim red"))
        return None

    # ------------------------------------------------------------------
    # mode
    # ------------------------------------------------------------------

    def do_mode(self, arg: str) -> Optional[bool]:
        """mode <list|set|show|briefing> ..."""
        parts = shlex.split(arg) if arg else []
        if not parts:
            self.console.print("usage: mode <list|set|show|briefing> ...")
            return None
        sub = parts[0]
        rest = parts[1:]
        try:
            if sub == "list":
                self.console.print(render_mode_list())
            elif sub == "show":
                state = (self.dev.last_result.state
                         if getattr(self.dev.last_result, "state", None) is not None
                         else None) if self.dev.last_result is not None else None
                self.console.print(render_mode_show(self.dev.session, state))
            elif sub == "set":
                if not rest:
                    self.console.print("usage: mode set <name>")
                    return None
                self._mode_set(rest[0])
            elif sub == "homework":
                self._mode_homework(rest)
            elif sub == "reflective":
                self._mode_reflective(rest)
            elif sub == "briefing":
                if not rest:
                    # Show current
                    b = getattr(self.dev.session, "mission_briefing", None)
                    self.console.print(f"current briefing: {b or '(none)'}")
                    return None
                text = " ".join(rest)
                self.dev.orchestrator.set_mission_briefing(text)
                self.console.print(f"briefing set: {text[:160]}")
            else:
                self.console.print(render_error_panel(
                    f"unknown mode subcommand: {sub} "
                    f"(list | set | show | briefing | homework | reflective)"
                ))
        except Exception as exc:  # noqa: BLE001
            self.console.print(render_error_panel(
                f"mode failed: {type(exc).__name__}: {exc}"
            ))
            self.console.print(Text(traceback.format_exc(), style="dim red"))
        return None

    def _mode_set(self, name: str) -> None:
        """Set learning mode, regular mode, or arbitrary mode token.

        Supported short forms:
          M1..M5            → learning_mode_N + reward profile
          Regular / Normal  → reset to default
          <other>           → set as raw mode token (no profile validation)
        """
        from zados.core.mode_profiles import profile_for_mode, profile_for_learning_mode

        session = self.dev.session
        if session is None:
            self.console.print(render_error_panel("no session"))
            return

        norm = name.strip()
        if norm.upper() in ("M1", "M2", "M3", "M4", "M5"):
            mode_num = int(norm[1])
            session.active_learning_mode = f"M{mode_num}"
            session.session_mode = "learning"
            session.reward_profile_name = profile_for_learning_mode(mode_num)
            self.console.print(
                f"learning mode set: M{mode_num}  profile={session.reward_profile_name}"
            )
        elif norm.lower() in ("regular", "normal"):
            session.active_learning_mode = None
            session.session_mode = "regular"
            session.initial_mode = "Normal"
            session.reward_profile_name = profile_for_mode("RegularInput")
            self.console.print(
                f"reset to regular mode  profile={session.reward_profile_name}"
            )
        else:
            # Treat as a raw mode token from MODE_TO_PROFILE.
            session.initial_mode = norm
            session.reward_profile_name = profile_for_mode(norm)
            self.console.print(
                f"mode token set: {norm}  profile={session.reward_profile_name}"
            )

    def _mode_homework(self, rest: list[str]) -> None:
        """`mode homework run` — invoke /homework via classifier."""
        sub = rest[0] if rest else "run"
        if sub != "run":
            self.console.print(render_error_panel(
                f"unknown homework subcommand: {sub} (only `run` is supported)"
            ))
            return
        result = self._run_commanded("/homework", "mode homework")
        if result is not None:
            self.console.print(render_homework_result(result))

    def _mode_reflective(self, rest: list[str]) -> None:
        """`mode reflective run` — invoke /reflective via classifier."""
        sub = rest[0] if rest else "run"
        if sub != "run":
            self.console.print(render_error_panel(
                f"unknown reflective subcommand: {sub} (only `run` is supported)"
            ))
            return
        result = self._run_commanded("/reflective", "mode reflective")
        if result is not None:
            self.console.print(render_reflective_result(result))

    # ------------------------------------------------------------------
    # sleep
    # ------------------------------------------------------------------

    def do_sleep(self, arg: str) -> Optional[bool]:
        """sleep <rem|dream|triage|status|exit>"""
        parts = shlex.split(arg) if arg else ["rem"]
        sub = parts[0]
        try:
            if sub == "rem":
                result = self._run_commanded("/sleep rem", "sleep rem")
                if result is not None:
                    self.console.print(render_rem_result(result))
            elif sub == "dream":
                result = self._run_commanded("/sleep dream", "sleep dream")
                if result is not None:
                    self.console.print(render_dream_result(result))
            elif sub == "triage":
                # No dedicated triage pipeline; falls back to REM (light consolidation).
                self.console.print(
                    "(no dedicated triage pipeline — running /sleep rem with light consolidation)"
                )
                result = self._run_commanded("/sleep rem", "sleep triage")
                if result is not None:
                    self.console.print(render_rem_result(result))
            elif sub == "status":
                self.console.print(render_sleep_status(self.dev.session))
            elif sub == "exit":
                if self.dev.session is not None:
                    self.dev.session.session_mode = "regular"
                    self.dev.session.active_learning_mode = None
                self.console.print("sleep mode exited — back to regular.")
            else:
                self.console.print(render_error_panel(
                    f"unknown sleep subcommand: {sub} (rem | dream | triage | status | exit)"
                ))
        except Exception as exc:  # noqa: BLE001
            self.dev.record_error(f"sleep {sub}", exc)
            self.console.print(render_error_panel(f"sleep failed: {type(exc).__name__}: {exc}"))
            self.console.print(Text(traceback.format_exc(), style="dim red"))
        return None

    # ------------------------------------------------------------------
    # Internal — run a commanded pipeline and record it in history
    # ------------------------------------------------------------------

    def _run_commanded(self, command: str, context: str) -> Optional[Any]:
        """Send a `/...` command through the classifier and append to history.

        Returns the raw result dict (or None on failure).  Errors are
        captured into runtime_errors and surfaced to the user.
        """
        try:
            result = self.dev.classifier.process_text(command)
        except Exception as exc:  # noqa: BLE001
            self.dev.record_error(context, exc)
            self.console.print(render_error_panel(
                f"{context} failed: {type(exc).__name__}: {exc}"
            ))
            self.console.print(Text(traceback.format_exc(), style="dim red"))
            return None
        # Track in history so `chat history` shows commanded turns too.
        self.dev.history.append(result)
        self.dev.last_result = result
        return result

    # ------------------------------------------------------------------
    # dev
    # ------------------------------------------------------------------

    def do_dev(self, arg: str) -> Optional[bool]:
        """dev <reward|nt|pipeline> ...

        reward:
          dev reward profiles
          dev reward profile <name>
          dev reward map
          dev reward learned
          dev reward override --logic V --ethics V --innovation V --attunement V
          dev reward reset
        nt:
          dev nt show [--full]
          dev nt set <name> <value>
          dev nt reset
        pipeline:
          dev pipeline last [--full]
          dev pipeline dispatch [--turn N]
          dev pipeline errors [<idx>]
        """
        parts = shlex.split(arg) if arg else []
        if not parts:
            self.console.print("usage: dev <reward|nt|pipeline> ...")
            return None
        group = parts[0]
        rest = parts[1:]
        try:
            if group == "reward":
                self._dev_reward(rest)
            elif group == "nt":
                self._dev_nt(rest)
            elif group == "pipeline":
                self._dev_pipeline(rest)
            elif group == "defaults":
                self.console.print(
                    "(dev defaults is not implemented in v1 — hardcoded defaults are "
                    "read-only via `mem ltmm identity.hardcoded list`.)"
                )
            else:
                self.console.print(render_error_panel(
                    f"unknown dev group: {group} (reward | nt | pipeline)"
                ))
        except Exception as exc:  # noqa: BLE001
            self.dev.record_error(f"dev {group}", exc)
            self.console.print(render_error_panel(
                f"dev failed: {type(exc).__name__}: {exc}"
            ))
            self.console.print(Text(traceback.format_exc(), style="dim red"))
        return None

    # --- dev reward -----------------------------------------------------

    def _dev_reward(self, args: list[str]) -> None:
        if not args:
            self.console.print("usage: dev reward <profiles|profile|map|learned|override|reset>")
            return
        sub = args[0]
        rest = args[1:]
        if sub == "profiles":
            self.console.print(render_reward_profiles())
        elif sub == "profile":
            if not rest:
                self.console.print("usage: dev reward profile <name>")
                return
            self.console.print(render_reward_profile_detail(rest[0]))
        elif sub == "map":
            self.console.print(render_reward_map())
        elif sub == "learned":
            self.console.print(render_reward_learned(self.dev.session))
        elif sub == "reset":
            self.console.print(apply_reward_reset(self.dev.session))
        elif sub == "override":
            weights = self._parse_weight_flags(rest)
            required = {"logic", "ethics", "innovation", "attunement"}
            missing = required - set(weights.keys())
            if missing:
                self.console.print(render_error_panel(
                    "all four weights required: --logic V --ethics V --innovation V --attunement V"
                    f"  (missing: {', '.join(sorted(missing))})"
                ))
                return
            self.console.print(apply_reward_override(self.dev.session, weights))
        else:
            self.console.print(render_error_panel(f"unknown reward subcommand: {sub}"))

    def _parse_weight_flags(self, tokens: list[str]) -> dict[str, float]:
        """Parse `--logic 0.7 --ethics 0.3 ...` into a dict."""
        out: dict[str, float] = {}
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.startswith("--") and i + 1 < len(tokens):
                key = t[2:].lower()
                try:
                    out[key] = float(tokens[i + 1])
                except ValueError:
                    self.console.print(render_error_panel(
                        f"value for {t} must be a float (got {tokens[i + 1]!r})"
                    ))
                i += 2
            else:
                i += 1
        return out

    # --- dev nt ---------------------------------------------------------

    def _dev_nt(self, args: list[str]) -> None:
        if not args:
            args = ["show"]
        sub = args[0]
        rest = args[1:]
        if sub == "show":
            full = "--full" in rest
            self.console.print(render_nt_state(self.dev.neurochem, full=full))
        elif sub == "set":
            if len(rest) < 2:
                self.console.print("usage: dev nt set <name> <value>")
                return
            try:
                value = float(rest[1])
            except ValueError:
                self.console.print(render_error_panel(
                    f"value must be a float in [0.0, 1.0] (got {rest[1]!r})"
                ))
                return
            self.console.print(apply_nt_set(self.dev.neurochem, rest[0], value))
        elif sub == "reset":
            self.console.print(apply_nt_reset(self.dev.neurochem))
        else:
            self.console.print(render_error_panel(f"unknown nt subcommand: {sub}"))

    # --- dev pipeline ---------------------------------------------------

    def _dev_pipeline(self, args: list[str]) -> None:
        if not args:
            self.console.print("usage: dev pipeline <last|dispatch|errors> [...]")
            return
        sub = args[0]
        rest = args[1:]
        if sub == "last":
            full = "--full" in rest
            turn_idx = _extract_turn_flag(rest)
            result = self._resolve_turn(turn_idx)
            self.console.print(render_pipeline_last(result, full=full))
        elif sub == "dispatch":
            turn_idx = _extract_turn_flag(rest)
            result = self._resolve_turn(turn_idx)
            self.console.print(render_pipeline_dispatch(result))
        elif sub == "errors":
            if rest:
                try:
                    idx = int(rest[0])
                except ValueError:
                    self.console.print("usage: dev pipeline errors [<idx>]")
                    return
                self.console.print(render_pipeline_error_detail(self.dev.runtime_errors, idx))
            else:
                self.console.print(render_pipeline_errors(self.dev.runtime_errors))
        else:
            self.console.print(render_error_panel(
                f"unknown pipeline subcommand: {sub} (last | dispatch | errors)"
            ))

    # ------------------------------------------------------------------
    # nt — top-level convenience (spec §8)
    # ------------------------------------------------------------------

    def do_nt(self, arg: str) -> Optional[bool]:
        """nt <show|full|set|reset|metrics>

        Convenience shortcuts equivalent to `show neurochem` / `dev nt ...`.
        """
        parts = shlex.split(arg) if arg else ["show"]
        sub = parts[0]
        rest = parts[1:]
        try:
            if sub == "show":
                self.console.print(render_nt_state(self.dev.neurochem, full=False))
            elif sub == "full":
                self.console.print(render_nt_state(self.dev.neurochem, full=True))
            elif sub == "set":
                if len(rest) < 2:
                    self.console.print("usage: nt set <name> <value>")
                    return None
                try:
                    v = float(rest[1])
                except ValueError:
                    self.console.print(render_error_panel("value must be a float"))
                    return None
                self.console.print(apply_nt_set(self.dev.neurochem, rest[0], v))
            elif sub == "reset":
                self.console.print(apply_nt_reset(self.dev.neurochem))
            elif sub == "metrics":
                self.console.print(render_nt_metrics_only(self.dev.neurochem))
            else:
                self.console.print(render_error_panel(
                    f"unknown nt subcommand: {sub} (show | full | set | reset | metrics)"
                ))
        except Exception as exc:  # noqa: BLE001
            self.dev.record_error(f"nt {sub}", exc)
            self.console.print(render_error_panel(f"nt failed: {type(exc).__name__}: {exc}"))
            self.console.print(Text(traceback.format_exc(), style="dim red"))
        return None

    # ------------------------------------------------------------------
    # set
    # ------------------------------------------------------------------

    def do_set(self, arg: str) -> Optional[bool]:
        """set <key> <value>   — keys: verbosity, autoshow"""
        parts = shlex.split(arg) if arg else []
        if len(parts) < 2:
            self.console.print(
                f"usage: set <key> <value>\n"
                f"  verbosity: {', '.join(_VALID_VERBOSITIES)}\n"
                f"  autoshow:  on, off"
            )
            return None
        key, value = parts[0], parts[1]
        if key == "verbosity":
            if value not in _VALID_VERBOSITIES:
                self.console.print(render_error_panel(
                    f"verbosity must be one of: {', '.join(_VALID_VERBOSITIES)}"
                ))
                return None
            self.dev.verbosity = value  # type: ignore[assignment]
            self.console.print(f"verbosity = {value}")
        elif key == "autoshow":
            self.dev.autoshow = value.lower() in ("on", "true", "1", "yes")
            self.console.print(f"autoshow = {self.dev.autoshow}")
        else:
            self.console.print(render_error_panel(f"unknown setting: {key}"))
        return None

    # ------------------------------------------------------------------
    # sess
    # ------------------------------------------------------------------

    def do_sess(self, arg: str) -> Optional[bool]:
        """sess <status|close|drift>"""
        parts = shlex.split(arg) if arg else []
        if not parts:
            self.console.print("usage: sess <status|close|drift>")
            return None
        sub = parts[0]
        if sub == "status":
            self._sess_status()
        elif sub == "close":
            self._sess_close()
        elif sub == "drift":
            self._sess_drift()
        else:
            self.console.print(render_error_panel(f"unknown `sess` subcommand: {sub}"))
        return None

    def _sess_status(self) -> None:
        s = self.dev.session
        if s is None:
            self.console.print("(no session — closed)")
            return
        lines = [
            f"session_id:      {getattr(s, 'session_id', '?')}",
            f"branch:          {getattr(s, 'branch', '?')}",
            f"initial_mode:    {getattr(s, 'initial_mode', '?')}",
            f"active_learning: {getattr(s, 'active_learning_mode', None) or '-'}",
            f"session_mode:    {getattr(s, 'session_mode', '?')}",
            f"profile:         {getattr(s, 'reward_profile_name', '?')}",
            f"briefing:        {getattr(s, 'mission_briefing', None) or '(none)'}",
            f"REPL turns:      {len(self.dev.history)}",
            f"engines loaded:  {len(self.dev.engines)}/32 "
            f"(errors: {len(self.dev.stack.engine_errors)})",
        ]
        self.console.print("\n".join(lines))

    def _sess_close(self) -> None:
        summary = self.dev.orchestrator.close_session()
        self.console.print("session closed:")
        for k, v in (summary or {}).items():
            self.console.print(f"  {k}: {v}")

    def _sess_drift(self) -> None:
        drift = self.dev.orchestrator.check_drift()
        self.console.print(f"drift detected: {drift}")

    # ------------------------------------------------------------------
    # quit / exit / EOF
    # ------------------------------------------------------------------

    def do_quit(self, arg: str) -> bool:
        """Close the session and exit the shell."""
        try:
            self.dev.orchestrator.close_session()
        except Exception as exc:  # noqa: BLE001
            self.console.print(render_error_panel(
                f"close_session failed: {type(exc).__name__}: {exc}"
            ))
        self.console.print("bye.")
        return True

    do_exit = do_quit
    do_EOF = do_quit

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _print_status_line(self) -> None:
        self.console.print(render_status_line(self.dev))

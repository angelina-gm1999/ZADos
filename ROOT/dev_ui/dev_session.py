"""DevSession — single in-process holder for the ZADOS stack + REPL state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

from zados.bootstrap import ZadosStack, boot_zados

Verbosity = Literal["quiet", "normal", "nerd"]


@dataclass
class DevSession:
    """Holds the live ZADOS stack and the REPL's transient state.

    Constructed once at shell startup; passed by reference to every command
    handler.
    """
    stack: ZadosStack
    history: List[Any] = field(default_factory=list)   # turn results (PipelineResult-likes)
    last_result: Any = None
    last_classification: Any = None                     # last ClassificationResult
    verbosity: Verbosity = "normal"
    autoshow: bool = True
    staged_input: Optional[str] = None                  # prefill for next chat send
    runtime_errors: List[dict] = field(default_factory=list)  # captured exceptions

    def record_error(self, context: str, exc: BaseException) -> None:
        """Append an error record for later inspection via `dev pipeline errors`."""
        import time
        import traceback as _tb
        self.runtime_errors.append({
            "timestamp": time.time(),
            "context": context,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": _tb.format_exc(),
            "turn": len(self.history),
        })

    # Convenience accessors --------------------------------------------------

    @property
    def orchestrator(self) -> Any:
        return self.stack.orchestrator

    @property
    def classifier(self) -> Any:
        return self.stack.classifier

    @property
    def memory(self) -> Any:
        return self.stack.memory

    @property
    def neurochem(self) -> Any:
        return self.stack.neurochem

    @property
    def engines(self) -> dict:
        return self.stack.engines

    @property
    def session(self) -> Any:
        """Live SessionState from the orchestrator (None if closed)."""
        return self.orchestrator.session


def build_dev_session(register_engines: bool = True) -> DevSession:
    """Boot the ZADOS stack and wrap it in a DevSession."""
    stack = boot_zados(register_engines=register_engines, open_session=True)
    return DevSession(stack=stack)

from __future__ import annotations


from typing import Dict, Any, Set


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class SymbolicRecombinationSubmodule(RewardSubmodule):
    """
    Evaluates whether known symbolic elements are being recombined
    into novel configurations.


    This does NOT assess semantic quality or correctness.
    """


    @property
    def name(self) -> str:
        return "symbolic_recombination"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - symbols_used: iterable of hashable symbols
        - known_symbol_pairs: iterable of tuple(symbol, symbol)
        """
        symbols = set(state.get("symbols_used", []))
        known_pairs = set(tuple(p) for p in state.get("known_symbol_pairs", []))


        recombined_pairs: Set[tuple] = set()


        symbols_list = list(symbols)
        for i in range(len(symbols_list)):
            for j in range(i + 1, len(symbols_list)):
                pair = (symbols_list[i], symbols_list[j])
                if pair not in known_pairs:
                    recombined_pairs.add(pair)


        if not symbols:
            score = 0.0
        else:
            score = min(1.0, len(recombined_pairs) / max(1, len(symbols)))


        flags = {}


        if score > 0.7 and not known_pairs:
            flags["unanchored_recombination"] = RewardFlag(
                name="unanchored_recombination",
                severity="warning",
                message="High symbolic recombination without prior symbol pair anchors",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "symbols_used": list(symbols),
                "recombined_pairs": list(recombined_pairs),
            },
        )

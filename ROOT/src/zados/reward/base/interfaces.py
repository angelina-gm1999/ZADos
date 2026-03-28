from __future__ import annotations


from abc import ABC, abstractmethod
from typing import Dict, Any


from .types import RewardContext, RewardSubscore, RewardDomainResult




class RewardSubmodule(ABC):
    """
    A submodule computes a single named subscore and optional flags.
    """


    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


    @abstractmethod
    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        state: model-agnostic structured state (dict in Phase 0).
        ctx: RewardContext (mode, timestamp, metadata).
        """
        raise NotImplementedError




class RewardDomain(ABC):
    """
    A domain aggregates multiple submodules and returns a domain-level result.
    """


    @property
    @abstractmethod
    def domain_name(self) -> str:
        raise NotImplementedError


    @abstractmethod
    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardDomainResult:
        raise NotImplementedError

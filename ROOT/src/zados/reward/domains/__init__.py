"""
Reward domain orchestrators.

Each domain evaluates a cognitive dimension and produces a RewardDomainResult.
Submodules within each domain are accessible via the domain's own package.
"""

from .ethics import EthicsDomain
from .innovation import InnovationDomain
from .logic import LogicDomain
from .human_attunement import HumanAttunementDomain

__all__ = [
    "EthicsDomain",
    "InnovationDomain",
    "LogicDomain",
    "HumanAttunementDomain",
]

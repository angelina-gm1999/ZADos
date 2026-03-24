"""Tests for shared memory types (MemoryPacket, MemoryTier, enums)."""
import pytest
from datetime import datetime

from zados.memory.types import (
    CompressionLevel,
    MemoryPacket,
    MemoryTier,
    SpeakerID,
)


def test_memory_tier_values():
    assert MemoryTier.STMM == "STMM"
    assert MemoryTier.MTMM == "MTMM"
    assert MemoryTier.LTMM == "LTMM"


def test_compression_level_values():
    assert CompressionLevel.FULL     == "full"
    assert CompressionLevel.SEMANTIC == "semantic"
    assert CompressionLevel.SYMBOLIC == "symbolic"


def test_speaker_id_values():
    assert SpeakerID.USER   == "user"
    assert SpeakerID.SYSTEM == "system"


def test_memory_packet_defaults():
    pkt = MemoryPacket()
    assert pkt.source_tier      == MemoryTier.STMM
    assert pkt.destination_tier == MemoryTier.MTMM
    assert pkt.user_message     == ""
    assert pkt.turn_index       == 0
    assert pkt.flags            == []
    assert pkt.embedding        is None


def test_memory_packet_unique_ids():
    p1 = MemoryPacket()
    p2 = MemoryPacket()
    assert p1.packet_id != p2.packet_id


def test_memory_packet_to_dict():
    pkt = MemoryPacket(user_message="hello", system_response="world", turn_index=3)
    d   = pkt.to_dict()
    assert d["user_message"]   == "hello"
    assert d["system_response"] == "world"
    assert d["turn_index"]     == 3
    assert "timestamp"         in d
    assert "packet_id"         in d


def test_memory_packet_trust_weight_default():
    pkt = MemoryPacket()
    assert 0.0 <= pkt.trust_weight <= 1.0


def test_memory_packet_custom_fields():
    pkt = MemoryPacket(
        user_message="test",
        reward_scores={"ethics": 0.8, "logic": 0.9},
        flags=["FLAG_A", "FLAG_B"],
        emotional_significance=0.75,
    )
    assert pkt.reward_scores["ethics"] == 0.8
    assert len(pkt.flags) == 2
    assert pkt.emotional_significance == 0.75

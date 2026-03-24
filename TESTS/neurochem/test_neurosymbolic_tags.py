import pytest
from zados.neurochem.neurosymbolic import (
    NeurotransmitterTag,
    ReceptorTag,
    OscillationBandTag,
    ModifierTag,
    ConcentrationComponentTag,
    encode_neurosymbolic_triplet,
    parse_neurosymbolic_triplet,
)
from zados.neurochem.neurosymbolic.tags import (
    GateTag,
    PhasicTonicMarker,
    MASTERGRID_MODIFIER_MAP,
)


def test_neurotransmitter_tags_exist():
    """Neurotransmitter tags are defined."""
    assert NeurotransmitterTag.DA.value == "dopamine"
    assert NeurotransmitterTag.SEROTONIN.value == "5-HT"
    assert NeurotransmitterTag.GABA.value == "gaba"


def test_receptor_tags_exist():
    """Receptor tags are defined."""
    assert ReceptorTag.DA_D1.value == "DA_D1"
    assert ReceptorTag.SEROTONIN_2A.value == "5HT_2A"
    assert ReceptorTag.GLU_NMDA.value == "GLU_NMDA"


def test_oscillation_band_tags():
    """Oscillation band tags are defined."""
    assert OscillationBandTag.DELTA.value == "delta"
    assert OscillationBandTag.GAMMA.value == "gamma"


def test_modifier_tags():
    """Modifier tags are defined."""
    assert ModifierTag.UP_DENSITY.value == "↑density"
    assert ModifierTag.DESENSITIZED.value == "desensitized"


def test_concentration_component_tags():
    """Concentration component tags are defined."""
    assert ConcentrationComponentTag.TONIC.value == "tonic"
    assert ConcentrationComponentTag.PHASIC.value == "phasic"


def test_encode_simple_triplet():
    """Can encode a simple neurosymbolic triplet."""
    encoded = encode_neurosymbolic_triplet(
        NeurotransmitterTag.DA,
        ReceptorTag.DA_D1,
        ModifierTag.UP_DENSITY,
    )
    assert "DA" in encoded
    assert "D1" in encoded
    assert "↑density" in encoded


def test_encode_triplet_with_oscillation_gate():
    """Can encode triplet with oscillation gating."""
    encoded = encode_neurosymbolic_triplet(
        NeurotransmitterTag.GLU,
        ReceptorTag.GLU_NMDA,
        ModifierTag.UP_AFFINITY,
        OscillationBandTag.GAMMA,
    )
    assert encoded.startswith("gamma{")
    assert encoded.endswith("}")
    assert "GLU" in encoded
    assert "NMDA" in encoded


def test_parse_simple_triplet():
    """Can parse a simple triplet."""
    parsed = parse_neurosymbolic_triplet("DA→D1:↑density")
    assert parsed["neurotransmitter"] == "DA"
    assert parsed["receptor"] == "D1"
    assert parsed["modifier"] == "↑density"
    assert parsed["oscillation_gate"] is None


def test_parse_triplet_with_oscillation_gate():
    """Can parse triplet with oscillation gating."""
    parsed = parse_neurosymbolic_triplet("gamma{GLU→NMDA:↑affinity}")
    assert parsed["neurotransmitter"] == "GLU"
    assert parsed["receptor"] == "NMDA"
    assert parsed["modifier"] == "↑affinity"
    assert parsed["oscillation_gate"] == "gamma"


def test_encode_decode_roundtrip():
    """Encode then decode produces consistent result."""
    encoded = encode_neurosymbolic_triplet(
        NeurotransmitterTag.SEROTONIN,
        ReceptorTag.SEROTONIN_2A,
        ModifierTag.DOWN_SENSITIVITY,
    )
    parsed = parse_neurosymbolic_triplet(encoded)
    
    assert "SEROTONIN" in parsed["neurotransmitter"] or "5HT" in parsed["neurotransmitter"]
    assert "2A" in parsed["receptor"]
    assert "↓sensitivity" in parsed["modifier"]


def test_encode_decode_with_gate_roundtrip():
    """Encode then decode with gate produces consistent result."""
    encoded = encode_neurosymbolic_triplet(
        NeurotransmitterTag.DA,
        ReceptorTag.DA_D3,
        ModifierTag.UP_RELEASE,
        OscillationBandTag.THETA,
    )
    parsed = parse_neurosymbolic_triplet(encoded)
    
    assert parsed["oscillation_gate"] == "theta"
    assert "DA" in parsed["neurotransmitter"]
    assert "D3" in parsed["receptor"]


def test_parse_invalid_format_raises():
    """Parsing invalid format raises ValueError."""
    with pytest.raises(ValueError, match="Invalid triplet format"):
        parse_neurosymbolic_triplet("invalid_string")


def test_parse_missing_modifier_raises():
    """Parsing triplet missing modifier raises ValueError."""
    with pytest.raises(ValueError, match="Invalid triplet format"):
        parse_neurosymbolic_triplet("DA→D1")


def test_all_neurotransmitter_tags_unique():
    """All neurotransmitter tags have unique values."""
    values = [tag.value for tag in NeurotransmitterTag]
    assert len(values) == len(set(values))


def test_all_receptor_tags_unique():
    """All receptor tags have unique values."""
    values = [tag.value for tag in ReceptorTag]
    assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# Phase 29: Extended vocabulary tests
# ---------------------------------------------------------------------------

class TestExtendedModifierTags:
    """Tests for K.1.4 / K.7.2C modifier extensions."""

    def test_new_modifier_tags_exist(self):
        """All 14 new ModifierTag members are accessible."""
        new_tags = [
            ModifierTag.UP_COUPLING, ModifierTag.DOWN_COUPLING,
            ModifierTag.UP_AVAILABILITY, ModifierTag.DOWN_AVAILABILITY,
            ModifierTag.UP_ACT, ModifierTag.DOWN_ACT,
            ModifierTag.UP_NOISE, ModifierTag.DOWN_NOISE,
            ModifierTag.UP_RECOV, ModifierTag.DOWN_RECOV,
            ModifierTag.UP_SATURATION, ModifierTag.DOWN_SATURATION,
            ModifierTag.BIND, ModifierTag.UNBIND,
        ]
        assert len(new_tags) == 14
        for tag in new_tags:
            assert isinstance(tag, ModifierTag)

    def test_modifier_values_unique(self):
        """All ModifierTag values remain unique after extension."""
        values = [tag.value for tag in ModifierTag]
        assert len(values) == len(set(values))

    def test_saturation_modifier_values(self):
        """Saturation modifier values match spec notation."""
        assert ModifierTag.UP_SATURATION.value == "↑S"
        assert ModifierTag.DOWN_SATURATION.value == "↓S"

    def test_activation_modifier_values(self):
        """Activation modifier values are correct."""
        assert ModifierTag.UP_ACT.value == "↑activation"
        assert ModifierTag.DOWN_ACT.value == "↓activation"


class TestGateTag:
    """Tests for oscillation gate tokens (K.7.2D)."""

    def test_gate_tag_count(self):
        """GateTag has 7 members (5 bands + 2 CFC)."""
        assert len(GateTag) == 7

    def test_single_band_gates(self):
        """Single-band gates are defined."""
        assert GateTag.DELTA.value == "DELTA"
        assert GateTag.THETA.value == "THETA"
        assert GateTag.ALPHA.value == "ALPHA"
        assert GateTag.BETA.value == "BETA"
        assert GateTag.GAMMA.value == "GAMMA"

    def test_cfc_gates(self):
        """Cross-frequency coupling gates are defined."""
        assert GateTag.THETA_GAMMA.value == "THETA_GAMMA"
        assert GateTag.ALPHA_BETA.value == "ALPHA_BETA"


class TestPhasicTonicMarker:
    """Tests for phasic/tonic signal markers (K.3)."""

    def test_marker_count(self):
        """PhasicTonicMarker has 2 members."""
        assert len(PhasicTonicMarker) == 2

    def test_marker_values(self):
        """Marker values are correct."""
        assert PhasicTonicMarker.PHASIC.value == "phasic"
        assert PhasicTonicMarker.TONIC.value == "tonic"


class TestMastergridModifierMap:
    """Tests for mastergrid ASCII modifier mapping (K.7.2C)."""

    def test_map_completeness(self):
        """Map covers 15 standard ASCII modifier tokens."""
        assert len(MASTERGRID_MODIFIER_MAP) == 15

    def test_density_mapping(self):
        """Density tokens map correctly."""
        assert MASTERGRID_MODIFIER_MAP["UP_DENS"] == ModifierTag.UP_DENSITY
        assert MASTERGRID_MODIFIER_MAP["DOWN_DENS"] == ModifierTag.DOWN_DENSITY

    def test_state_mapping(self):
        """State tokens map correctly."""
        assert MASTERGRID_MODIFIER_MAP["DESENS"] == ModifierTag.DESENSITIZED
        assert MASTERGRID_MODIFIER_MAP["INTERNAL"] == ModifierTag.INTERNALIZED
        assert MASTERGRID_MODIFIER_MAP["UPREG"] == ModifierTag.UPREGULATED

    def test_activation_mapping(self):
        """Activation tokens map correctly."""
        assert MASTERGRID_MODIFIER_MAP["UP_ACT"] == ModifierTag.UP_ACT
        assert MASTERGRID_MODIFIER_MAP["DOWN_ACT"] == ModifierTag.DOWN_ACT

    def test_all_values_are_modifier_tags(self):
        """All map values are ModifierTag instances."""
        for value in MASTERGRID_MODIFIER_MAP.values():
            assert isinstance(value, ModifierTag)

    def test_existing_encode_decode_unchanged(self):
        """Existing triplet encode/decode still works (backward compat)."""
        encoded = encode_neurosymbolic_triplet(
            NeurotransmitterTag.DA,
            ReceptorTag.DA_D1,
            ModifierTag.UP_DENSITY,
        )
        assert "DA" in encoded
        assert "D1" in encoded
        assert "↑density" in encoded

        parsed = parse_neurosymbolic_triplet(encoded)
        assert parsed["neurotransmitter"] == "DA"
        assert parsed["receptor"] == "D1"
        assert parsed["modifier"] == "↑density"
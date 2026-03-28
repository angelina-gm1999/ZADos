"""
Tests for unified mastergrid parser (Appendix K.7).

Phase 30: Parse and encode all 6 canonical mastergrid forms.
"""

import pytest

from zados.neurochem.neurosymbolic.parser import (
    ParsedTriplet,
    ParsedOperator,
    ParsedStateExpr,
    ParsedTrigger,
    classify_entry,
    parse_triplet_entry,
    parse_operator_entry,
    parse_state_entry,
    parse_trigger_entry,
    parse_mastergrid,
    encode_entry,
    encode_mastergrid,
)


# ---------------------------------------------------------------------------
# Classification (K.7.5 step 2)
# ---------------------------------------------------------------------------

class TestClassifyEntry:
    """Tests for entry classification."""

    def test_classify_base_triplet(self):
        assert classify_entry("DA->D1:UP_ACT") == "triplet"

    def test_classify_gated_triplet(self):
        assert classify_entry("THETA{OXT->OXTR:UP_ACT}") == "gated_triplet"

    def test_classify_cfc_gated(self):
        assert classify_entry("THETA_GAMMA{DA->D3:UP_ACT}") == "gated_triplet"

    def test_classify_operator_int(self):
        assert classify_entry("INT(D2)") == "operator"

    def test_classify_operator_upr(self):
        assert classify_entry("UPR(OXTR)") == "operator"

    def test_classify_operator_switch(self):
        assert classify_entry("SWITCH(D1->D3)") == "operator"

    def test_classify_state(self):
        assert classify_entry("STATE(CreativeDrive)=0.4*DA_D1") == "state"

    def test_classify_trigger(self):
        assert classify_entry("IF(beta>0.6)=>ACTIVATE(LogicMode)") == "trigger"

    def test_classify_activate(self):
        assert classify_entry("ACTIVATE(RecursiveSynthesis)") == "activate"


# ---------------------------------------------------------------------------
# Triplet parsing (Forms 1-3)
# ---------------------------------------------------------------------------

class TestParseTripletEntry:
    """Tests for base and gated triplet parsing."""

    def test_parse_base_triplet(self):
        t = parse_triplet_entry("DA->D1:UP_ACT")
        assert t.nt == "DA"
        assert t.receptor == "D1"
        assert t.modifiers == ("UP_ACT",)
        assert t.gate is None
        assert t.signal_mode is None

    def test_parse_multi_modifier(self):
        t = parse_triplet_entry("DA->D2:UP_ACT,DOWN_NOISE")
        assert t.modifiers == ("UP_ACT", "DOWN_NOISE")

    def test_parse_gated(self):
        t = parse_triplet_entry("GAMMA{Glu->NMDA:UP_AFF}")
        assert t.nt == "Glu"
        assert t.receptor == "NMDA"
        assert t.modifiers == ("UP_AFF",)
        assert t.gate == "GAMMA"

    def test_parse_cfc_gated(self):
        t = parse_triplet_entry("THETA_GAMMA{DA->D3:UP_ACT,UP_AFF}")
        assert t.gate == "THETA_GAMMA"
        assert t.nt == "DA"
        assert t.receptor == "D3"
        assert t.modifiers == ("UP_ACT", "UP_AFF")

    def test_parse_phasic_unicode(self):
        t = parse_triplet_entry("DA\u2022->D1:UP_ACT")
        assert t.nt == "DA"
        assert t.signal_mode == "phasic"

    def test_parse_phasic_ascii(self):
        t = parse_triplet_entry("DA.P->D1:UP_ACT")
        assert t.nt == "DA"
        assert t.signal_mode == "phasic"

    def test_parse_tonic(self):
        t = parse_triplet_entry("5HT.T->5HT1A:UP_NOISE")
        assert t.nt == "5HT"
        assert t.signal_mode == "tonic"
        assert t.receptor == "5HT1A"

    def test_parse_tonic_tilde(self):
        t = parse_triplet_entry("5HT~->5HT1A:UP_NOISE")
        assert t.nt == "5HT"
        assert t.signal_mode == "tonic"

    def test_parse_gated_phasic_combined(self):
        t = parse_triplet_entry("THETA{DA.P->D3:UP_ACT}")
        assert t.gate == "THETA"
        assert t.nt == "DA"
        assert t.signal_mode == "phasic"
        assert t.receptor == "D3"

    def test_parse_unicode_arrow(self):
        """Handles Unicode → arrow."""
        t = parse_triplet_entry("DA\u2192D1:UP_ACT")
        assert t.nt == "DA"
        assert t.receptor == "D1"


# ---------------------------------------------------------------------------
# Operator parsing (Form 4)
# ---------------------------------------------------------------------------

class TestParseOperatorEntry:
    """Tests for plasticity operator parsing."""

    def test_parse_int(self):
        op = parse_operator_entry("INT(D2)")
        assert op.operator == "INT"
        assert op.target == "D2"
        assert op.target_b is None

    def test_parse_upr(self):
        op = parse_operator_entry("UPR(OXTR)")
        assert op.operator == "UPR"
        assert op.target == "OXTR"

    def test_parse_switch(self):
        op = parse_operator_entry("SWITCH(D1->D3)")
        assert op.operator == "SWITCH"
        assert op.target == "D1"
        assert op.target_b == "D3"

    def test_parse_dsn(self):
        op = parse_operator_entry("DSN(D2)")
        assert op.operator == "DSN"
        assert op.target == "D2"

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError):
            parse_operator_entry("UNKNOWN(X)")


# ---------------------------------------------------------------------------
# State expression parsing (Form 5)
# ---------------------------------------------------------------------------

class TestParseStateEntry:
    """Tests for STATE(X)=... parsing."""

    def test_parse_state_simple(self):
        se = parse_state_entry("STATE(Rigidity)=0.50*NE_NE_B1+0.40*DA_D2-0.30*CB1_CB1")
        assert se.name == "Rigidity"
        assert len(se.terms) == 3
        assert se.terms[0] == (0.50, "NE_NE_B1")
        assert se.terms[1] == (0.40, "DA_D2")
        assert se.terms[2] == (-0.30, "CB1_CB1")

    def test_parse_state_single_term(self):
        se = parse_state_entry("STATE(X)=1.0*DA_D1")
        assert se.name == "X"
        assert len(se.terms) == 1
        assert se.terms[0] == (1.0, "DA_D1")

    def test_invalid_state_raises(self):
        with pytest.raises(ValueError):
            parse_state_entry("NOT_A_STATE=1.0*X")


# ---------------------------------------------------------------------------
# Trigger parsing (Form 6)
# ---------------------------------------------------------------------------

class TestParseTriggerEntry:
    """Tests for IF(condition)=>... parsing."""

    def test_parse_trigger_simple(self):
        trig = parse_trigger_entry("IF(beta>0.6)=>ACTIVATE(LogicMode)")
        assert trig.condition == "beta>0.6"
        assert trig.activate_mode == "LogicMode"
        assert len(trig.actions) == 0

    def test_parse_trigger_with_actions(self):
        trig = parse_trigger_entry("IF(Fatigue>0.7)=>INT(D2);ACTIVATE(Containment)")
        assert trig.condition == "Fatigue>0.7"
        assert trig.activate_mode == "Containment"
        assert len(trig.actions) == 1
        assert isinstance(trig.actions[0], ParsedOperator)
        assert trig.actions[0].operator == "INT"

    def test_parse_trigger_complex_condition(self):
        trig = parse_trigger_entry(
            "IF(theta_gamma>0.55 AND S_NMDA>0.5)=>THETA_GAMMA{Glu->NMDA:UP_AFF};ACTIVATE(RecursiveSynthesis)"
        )
        assert "theta_gamma>0.55" in trig.condition
        assert "AND" in trig.condition
        assert trig.activate_mode == "RecursiveSynthesis"
        assert len(trig.actions) == 1
        assert isinstance(trig.actions[0], ParsedTriplet)
        assert trig.actions[0].gate == "THETA_GAMMA"

    def test_parse_trigger_no_activate(self):
        trig = parse_trigger_entry("IF(beta>0.6)=>DA->D2:UP_ACT")
        assert trig.activate_mode is None
        assert len(trig.actions) == 1

    def test_invalid_trigger_raises(self):
        with pytest.raises(ValueError):
            parse_trigger_entry("NOT_A_TRIGGER")


# ---------------------------------------------------------------------------
# Full mastergrid parsing
# ---------------------------------------------------------------------------

class TestParseMastergrid:
    """Tests for top-level mastergrid parsing."""

    def test_parse_empty_string(self):
        assert parse_mastergrid("") == []

    def test_parse_whitespace_only(self):
        assert parse_mastergrid("   ") == []

    def test_parse_single_triplet(self):
        entries = parse_mastergrid("DA->D1:UP_ACT")
        assert len(entries) == 1
        assert isinstance(entries[0], ParsedTriplet)

    def test_parse_multi_entry(self):
        text = "DA.P->D1:UP_ACT | DA.T->D2:DOWN_NOISE | THETA{DA.P->D3:UP_ACT}"
        entries = parse_mastergrid(text)
        assert len(entries) == 3
        assert all(isinstance(e, ParsedTriplet) for e in entries)
        assert entries[0].signal_mode == "phasic"
        assert entries[1].signal_mode == "tonic"
        assert entries[2].gate == "THETA"

    def test_parse_mixed_forms(self):
        text = "DA->D1:UP_ACT | INT(D2) | IF(beta>0.6)=>ACTIVATE(LogicMode)"
        entries = parse_mastergrid(text)
        assert len(entries) == 3
        assert isinstance(entries[0], ParsedTriplet)
        assert isinstance(entries[1], ParsedOperator)
        assert isinstance(entries[2], ParsedTrigger)

    def test_whitespace_tolerance(self):
        """Extra whitespace is stripped."""
        t = parse_triplet_entry("  DA  ->  D1  :  UP_ACT  ")
        assert t.nt == "DA"
        assert t.receptor == "D1"
        assert t.modifiers == ("UP_ACT",)

    def test_invalid_triplet_raises(self):
        with pytest.raises(ValueError):
            parse_triplet_entry("not_valid_at_all")

    def test_k76_pfc_dopamine_example(self):
        """K.7.6 example: PFC dopamine logic tuning."""
        text = (
            "DA.P->D1:UP_ACT | DA.T->D2:DOWN_NOISE | "
            "THETA{DA.P->D3:UP_ACT} | "
            "IF(beta>0.6 AND S_DA-D2>0.7)=>ACTIVATE(LogicMode)"
        )
        entries = parse_mastergrid(text)
        assert len(entries) == 4
        assert isinstance(entries[0], ParsedTriplet)
        assert entries[0].signal_mode == "phasic"
        assert isinstance(entries[3], ParsedTrigger)
        assert entries[3].activate_mode == "LogicMode"

    def test_k76_ethics_containment_example(self):
        """K.7.6 example: ethics containment motif."""
        text = "DELTA{GABA->GABAB:UP_ACT,UP_RECOV} | IF(Fatigue>0.7)=>INT(D2);ACTIVATE(Containment)"
        entries = parse_mastergrid(text)
        assert len(entries) == 2
        assert isinstance(entries[0], ParsedTriplet)
        assert entries[0].gate == "DELTA"
        assert entries[0].modifiers == ("UP_ACT", "UP_RECOV")
        assert isinstance(entries[1], ParsedTrigger)
        assert entries[1].activate_mode == "Containment"


# ---------------------------------------------------------------------------
# Roundtrip encoding
# ---------------------------------------------------------------------------

class TestRoundtrip:
    """Tests for encode/decode roundtrip consistency."""

    def test_roundtrip_base_triplet(self):
        text = "DA->D1:UP_ACT"
        entries = parse_mastergrid(text)
        encoded = encode_mastergrid(entries)
        assert encoded == "DA->D1:UP_ACT"

    def test_roundtrip_gated(self):
        text = "GAMMA{Glu->NMDA:UP_AFF}"
        entries = parse_mastergrid(text)
        encoded = encode_mastergrid(entries)
        assert encoded == "GAMMA{Glu->NMDA:UP_AFF}"

    def test_roundtrip_operator(self):
        text = "INT(D2)"
        entries = parse_mastergrid(text)
        encoded = encode_mastergrid(entries)
        assert encoded == "INT(D2)"

    def test_roundtrip_switch(self):
        text = "SWITCH(D1->D3)"
        entries = parse_mastergrid(text)
        encoded = encode_mastergrid(entries)
        assert encoded == "SWITCH(D1->D3)"

    def test_roundtrip_phasic(self):
        text = "DA.P->D1:UP_ACT"
        entries = parse_mastergrid(text)
        encoded = encode_mastergrid(entries)
        assert encoded == "DA.P->D1:UP_ACT"

    def test_roundtrip_full_mastergrid(self):
        text = "DA.P->D1:UP_ACT | INT(D2) | GAMMA{Glu->NMDA:UP_AFF}"
        entries = parse_mastergrid(text)
        encoded = encode_mastergrid(entries)
        re_entries = parse_mastergrid(encoded)
        assert len(re_entries) == 3
        # Verify each entry round-trips correctly
        for orig, reparse in zip(entries, re_entries):
            assert type(orig) == type(reparse)

    def test_roundtrip_trigger(self):
        text = "IF(beta>0.6)=>INT(D2);ACTIVATE(LogicMode)"
        entries = parse_mastergrid(text)
        encoded = encode_mastergrid(entries)
        assert "IF(beta>0.6)=>" in encoded
        assert "INT(D2)" in encoded
        assert "ACTIVATE(LogicMode)" in encoded

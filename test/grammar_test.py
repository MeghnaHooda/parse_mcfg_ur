import pytest
import sys
from collections import defaultdict

from parse_mcfg_ur.grammar import MCFGRuleElement, MCFGRuleElementInstance, MCFGRule
from parse_mcfg_ur.grammar import MCFGGrammar, MCFGChart, MCFGParser

@pytest.fixture
def sample_grammar_text():
    return """
S(uv) -> NP(u) VP(v)
S(uv) -> NPwh(u) VP(v)
S(vuw) -> Aux(u) Swhmain(v, w)
S(uwv) -> NPdisloc(u, v) VP(w)
S(uwv) -> NPwhdisloc(u, v) VP(w)
Sbar(uv) -> C(u) S(v)
Sbarwh(v, uw) -> C(u) Swhemb(v, w)
Sbarwh(u, v) -> NPwh(u) VP(v)
Swhmain(v, uw) -> NP(u) VPwhmain(v, w)
Swhmain(w, uxv) -> NPdisloc(u, v) VPwhmain(w, x)
Swhemb(v, uw) -> NP(u) VPwhemb(v, w)
Swhemb(w, uxv) -> NPdisloc(u, v) VPwhemb(w, x)
Src(v, uw) -> NP(u) VPrc(v, w)
Src(w, uxv) -> NPdisloc(u, v) VPrc(w, x)
Src(u, v) -> N(u) VP(v)
Swhrc(u, v) -> Nwh(u) VP(v)
Swhrc(v, uw) -> NP(u) VPwhrc(v, w)
Sbarwhrc(v, uw) -> C(u) Swhrc(v, w)
VP(uv) -> Vpres(u) NP(v)
VP(uv) -> Vpres(u) Sbar(v)
VPwhmain(u, v) -> NPwh(u) Vroot(v)
VPwhmain(u, wv) -> NPwhdisloc(u, v) Vroot(w)
VPwhmain(v, uw) -> Vroot(u) Sbarwh(v, w)
VPwhemb(u, v) -> NPwh(u) Vpres(v)
VPwhemb(u, wv) -> NPwhdisloc(u, v) Vpres(w)
VPwhemb(v, uw) -> Vpres(u) Sbarwh(v, w)
VPrc(u, v) -> N(u) Vpres(v)
VPrc(v, uw) -> Vpres(u) Nrc(v, w)
VPwhrc(u, v) -> Nwh(u) Vpres(v)
VPwhrc(v, uw) -> Vpres(u) Sbarwhrc(v, w)
NP(uv) -> D(u) N(v)
NP(uvw) -> D(u) Nrc(v, w)
NPdisloc(uv, w) -> D(u) Nrc(v, w)
NPwh(uv) -> Dwh(u) N(v)
NPwh(uvw) -> Dwh(u) Nrc(v, w)
NPwhdisloc(uv, w) -> Dwh(u) Nrc(v, w)
Nrc(v, uw) -> C(u) Src(v, w)
Nrc(u, vw) -> N(u) Swhrc(v, w)
Nrc(u, vwx) -> Nrc(u, v) Swhrc(w, x)
Dwh(which)
Nwh(who)
D(the)
D(a)
N(greyhound)
N(human)
Vpres(believes)
Vroot(believe)
Aux(does)
C(that)
""".strip()
# @pytest.mark.usefixtures("sample_grammar_text")

class TestMCFGRuleParsing:
    def test_all_rules_parse(self, sample_grammar_text):
        for line in sample_grammar_text.splitlines():
            line = line.strip()
            if not line or '->' not in line:
                continue
            try:
                rule = MCFGRule.from_string(line)
                assert isinstance(rule, MCFGRule)
            except Exception as e:
                pytest.fail(f"Failed to parse rule: '{line}'\nError: {e}")
@pytest.fixture
def simple_grammar_text():
        return """
    S(xy) -> NP(x) VP(y)
    NP(x) -> D(x)
    VP(x) -> V(x)
    D(the)
    V(runs)
    """.strip()
@pytest.fixture
def simple_grammar(sample_grammar_text):
    return MCFGGrammar(sample_grammar_text)

@pytest.fixture
def parser(simple_grammar):
    return MCFGParser(simple_grammar)

class TestMCFGparser:
    def test_parser_recognize_accepts_valid_string(self, parser):
        string = ["the", "runs"]
        assert parser.recognize(string) == True

    def test_parser_recognize_rejects_invalid_string(self, parser):
        string = ["the", "barks"]
        assert parser.recognize(string) == False

    def test_parser_parse_returns_chart(self, parser):
        string = ["the", "runs"]
        chart = parser.parse(string)

    #     # Should contain an entry for S from 0 to 2
    #     entries = chart.get(0, 2)
    #     assert any(entry.symbol == "S" for entry in entries)

    # def test_parser_chart_entries_have_backpointers(parser):
    #     string = ["the", "runs"]
    #     chart = parser._parse(string)

    #     entries = chart.get(0, 2)
    #     s_entries = [e for e in entries if e.symbol == "S"]
    #     assert s_entries, "No S entry found"
    #     for entry in s_entries:
    #         assert entry.backpointers is not None
    #         assert len(entry.backpointers) == 2

# class TestMCFGComponents:
     
#     #  @pytest.fixture(autouse=True)
#     def grammar(self, sample_grammar_text):
#         self.grammar = MCFGRule.from_string(sample_grammar_text)

    # def test_mcfg_grammar_loads_rules(self):
    #     rules = self.grammar.rules_for_lhs
    #     assert 'S' in rules
    #     assert any(rule.left_side == 'S' for rule in self.grammar.rules)
        # assert any(rule.is_terminal for rule in self.grammar.rules)

    # def test_mcfg_chart_entry_repr(self):
        # entry = MCFGChartEntry('NP', (['D', 'N'], (0, 1)))
        # assert repr(entry) == "NP → [D,N], [0,1]"

    # def test_mcfg_chart_add_and_get_entries(self):
        # chart = MCFGChart(length=3)
        # entry1 = MCFGChartEntry('NP', ['D', 'N'], (0, 1))
        # entry2 = MCFGChartEntry('VP', ['V'], (1, 2))
        
        # chart.add(0, 1, entry1)
        # chart.add(1, 2, entry2)

        # entries_01 = chart.get(0, 1)
        # entries_12 = chart.get(1, 2)

        # assert entry1 in entries_01
        # assert entry2 in entries_12
    
    # def test_getitem_access(self):
    #     chart = MCFGChart(length=3)
    #     # entry = MCFGChartEntry('X', ['A'], [(0, 1)])
    #     chart.add(0, 1, entry)
    #     assert entry in chart[0, 1]

    # def test_does_not_duplicate_entries(self):
    #     chart = MCFGChart(length=3)
    #     # entry = MCFGChartEntry('X', ['A'], [(0, 1)])
    #     # chart.add(0, 1, entry)
    #     chart.add(0, 1, entry)  # Add again

    #     assert len(chart.get(0, 1)) == 1  # Still only one entry

    # def test_chart_initializes_correctly(self):
    #     chart = MCFGChart(length=5)
    #     assert chart.length == 5
    #     assert isinstance(chart.entries, defaultdict)
                
class TestMCFGRuleElement:
    def test_str_and_eq(self):
        el1 = MCFGRuleElement("S", (0,), (1,))
        el2 = MCFGRuleElement("S", (0,), (1,))
        el3 = MCFGRuleElement("S", (1,), (0,))
        assert str(el1) == "S(0, 1)"
        assert el1 == el2
        assert el1 != el3

    def test_unique_string_variables(self):
        el = MCFGRuleElement("X", (0,), (1,), (2,))
        assert el.unique_string_variables == {0, 1, 2}


class TestMCFGRuleElementInstance:
    def test_eq_and_str(self):
        inst1 = MCFGRuleElementInstance("NP", (0, 2))
        inst2 = MCFGRuleElementInstance("NP", (0, 2))
        inst3 = MCFGRuleElementInstance("NP", (1, 3))
        assert inst1 == inst2
        assert inst1 != inst3
        assert str(inst1) == "NP([0, 2])"

    def test_properties(self):
        inst = MCFGRuleElementInstance("VP", (2, 5))
        assert inst.variable == "VP"
        assert inst.string_spans == ((2, 5),)


class TestMCFGRule:
    def test_valid_rule(self):
        left = MCFGRuleElement("S", (0,), (1,))
        right1 = MCFGRuleElement("NP", (0,))
        right2 = MCFGRuleElement("VP", (1,))
        rule = MCFGRule(left, right1, right2)
        assert rule.left_side == left
        assert rule.right_side == (right1, right2)
        assert not rule.is_epsilon

    def test_epsilon_rule(self):
        epsilon_rule = MCFGRule(MCFGRuleElement("X", ("Y",)))
        assert epsilon_rule.is_epsilon
        assert epsilon_rule.string_yield() == "X"

    def test_invalid_shared_variables(self):
        left = MCFGRuleElement("S", (0,), (1,))
        right1 = MCFGRuleElement("A", (0,))
        right2 = MCFGRuleElement("B", (0,))
        with pytest.raises(ValueError, match="right side variables cannot share"):
            MCFGRule(left, right1, right2)

    def test_invalid_left_right_variable_mismatch(self):
        left = MCFGRuleElement("S", (0,), (1,))
        right1 = MCFGRuleElement("A", (0,))
        right2 = MCFGRuleElement("B", (2,))
        with pytest.raises(ValueError, match="number of arguments to instantiate must"):
            MCFGRule(left, right1, right2)

    def test_from_string(self):
        rule = MCFGRule.from_string("S(01) -> NP(0) VP(1)")
        assert rule.left_side.variable == "S"
        assert rule.right_side[0].variable == "NP"
        assert rule.right_side[1].variable == "VP"
        assert rule.left_side.string_variables == ((0, 1),)

    def test_instantiate_left_side(self):
        rule = MCFGRule.from_string("S(01) -> NP(0) VP(1)")
        inst1 = MCFGRuleElementInstance("NP", (0, 2))
        inst2 = MCFGRuleElementInstance("VP", (2, 5))
        left_instance = rule.instantiate_left_side(inst1, inst2)
        assert left_instance == MCFGRuleElementInstance("S", (0, 5))

    def test_instantiate_adjacency_error(self):
        rule = MCFGRule.from_string("S(01) -> NP(0) VP(1)")
        inst1 = MCFGRuleElementInstance("NP", (0, 2))
        inst2 = MCFGRuleElementInstance("VP", (3, 5))  # not adjacent
        with pytest.raises(ValueError, match="must be adjacent"):
            rule.instantiate_left_side(inst1, inst2)

    def test_instantiate_alignment_error(self):
        rule = MCFGRule.from_string("S(01) -> NP(0) VP(1)")
        inst1 = MCFGRuleElementInstance("NP", (0, 2))
        inst2 = MCFGRuleElementInstance("VP", (2, 5), (6, 7))  # mismatch
        with pytest.raises(ValueError, match="do not align with rule"):
            rule.instantiate_left_side(inst1, inst2)
    
    # def test_parse_all_rules_from_fixture(self, sample_grammar_text):
    #     for line in sample_grammar_text.strip().splitlines():
    #         line = line.strip()
    #         if not line or "(" not in line:  # skip empty lines and terminals
    #             continue
    #         try:
    #             rule = MCFGRule.from_string(line)
    #             assert isinstance(rule, MCFGRule)
    #         except Exception as e:
    #             pytest.fail(f"Failed to parse rule: {line}\nError: {e}")

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
    D(the)
    D(a)
    N(greyhound)
    N(human)
    Vpres(believes)
    NP(x) -> D(x)
    NP(x) -> N(x)
    NP(x) -> N(x)
    NP(xy) -> D(x) N(y)
    VP(xy) -> Vpres(x) NP(y)
    S(xy) -> NP(x) VP(y)
    """.strip()
@pytest.fixture
def simple_grammar(sample_grammar_text):
    return MCFGGrammar(sample_grammar_text)

@pytest.fixture
def parser(simple_grammar):
    return MCFGParser(simple_grammar)

class TestMCFGparser:

    @pytest.mark.parametrize("valid_sentence", [
        ['the', 'human', 'believes', 'the', 'greyhound'], #simple sentence
        ['a', 'greyhound', 'believes', 'a', 'human'], #simple sentence with different determiner
        ['which', 'greyhound', 'believes', 'the', 'human'], #wh sentence
        ['that', 'the', 'human', 'believes', 'the', 'greyhound'], #clause sentence
        ['who', 'the', 'human','believes', 'the', 'greyhound'],  # subject wh-question
    ])
    def test_parser_accepts_valid_sentences(self, parser, valid_sentence):
        assert parser.recognize(valid_sentence) is True

    @pytest.mark.parametrize("invalid_sentence", [
        ['human', 'believes', 'the', 'greyhound'],  # missing determiner
        ['the', 'dog', 'the', 'cat'],              # missing verb
        ['believes', 'the', 'dog'],                # starts with verb
        ['the', 'teacher'],                        # incomplete
        ['the', 'robot', 'scans'],                 # missing object
    ])
    def test_parser_rejects_invalid_sentences(self, parser, invalid_sentence):
        assert parser.recognize(invalid_sentence) is False

    def test_parser_parse_produces_chart_with_S(self, parser):
        sentence = ['the', 'human', 'believes', 'the', 'greyhound']
        chart = list(parser.parse(sentence))
        assert any(entry.variable == "S" for entry in chart)

    def test_parser_recognize_accepts_valid_string(self, parser):
        string = ['the', 'human', 'believes','the', 'greyhound']
        assert parser.recognize(string) == True

    def test_parser_recognize_rejects_invalid_string(self, parser):
        string = ['the', 'human']
        assert parser.recognize(string) == False

    def test_parser_parse_returns_chart(self, parser):
        string = ['the', 'human', 'believes','the', 'greyhound']
        chart = list(parser.parse(string))

        assert any(entry.variable == "S" for entry in chart)


class TestMCFGComponents:

    def test_mcfg_grammar_parses_all_rules(self, simple_grammar):
        assert len(simple_grammar) == 49

    def test_mcfg_grammar_contains_terminal_and_nonterminal(self,simple_grammar):
        assert any(rule.is_terminal() for rule in simple_grammar)
        assert any(rule.left_side.variable == 'S' for rule in simple_grammar)
        assert any(rule.left_side.variable == 'NP' and len(rule.right_side) == 2 for rule in simple_grammar)
    def test_chart_add_and_get_single_entry(self):
        chart = MCFGChart()
        instance = chart.add('NP', ((0, 2),))
        assert instance is not None
        assert instance.variable == 'NP'
        assert instance.string_spans == ((0, 2),)

        # Duplicate should not be added
        assert chart.add('NP', ((0, 2),)) is None

        entries = chart.get(((0, 2),))
        assert len(entries) == 1
        assert entries[0].variable == 'NP'
    
    def test_chart_all_entries(self):
        chart = MCFGChart()
        chart.add('D', ((0, 1),))
        chart.add('N', ((1, 2),))
        chart.add('NP', ((0, 2),))

        entries = chart.all_entries()
        assert len(entries) == 3
        vars_in_chart = {entry.variable for entry in entries}
        assert vars_in_chart == {'D', 'N', 'NP'}

    def test_chart_initializes_correctly(self):
        chart = MCFGChart()
        assert isinstance(chart.chart, dict)
        assert len(chart.chart) == 0
        assert chart.all_entries() == []
                
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

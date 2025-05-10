import re
from collections import defaultdict

StringVariables = tuple[int, ...]
SpanIndices = tuple[int, ...]
SpanMap = dict[int, SpanIndices]


"""Type aliases for readability

Attributes
----------
StringVariables : tuple[int, ...]
    A tuple of integers representing string variable positions (e.g., (0,), (1,)).

SpanMap : dict[int, SpanIndices]
    A mapping from string variable index to a span in the input string.

SpanIndices : tuple[int, ...]
    This represents the half-open interval [start, end), where `start` is the index of the first token in the span and `end` is the index immediately after the last token
"""


class MCFGRuleElement:

    """A multiple context free grammar rule element

    Parameters
    ----------
    variable : str
        The name of the non-terminal variable (e.g., 'S', 'NP').
    string_variables : tuple[int, ...]
        Tuples of integer indices representing variables' positions in the final output string.

    Attributes
    ----------
    variable : str
        The non-terminal symbol.
    string_variables : tuple[StringVariables, ...]
        The list of string variable tuples (e.g., (0,), (1,)).
    """

    def __init__(self, variable: str, *string_variables: StringVariables):
        self._variable = variable
        self._string_variables = string_variables

    def __str__(self) -> str:
        strvars = ', '.join(
            ''.join(str(v) for v in vtup)
            for vtup in self._string_variables
        )
        
        return f"{self._variable}({strvars})"

    def __eq__(self, other) -> bool:
        vareq = self._variable == other._variable
        strvareq = self._string_variables == other._string_variables
        
        return vareq and strvareq
        
    def to_tuple(self) -> tuple[str, tuple[StringVariables, ...]]:
        return (self._variable, self._string_variables)

    def __hash__(self) -> int:
        return hash(self.to_tuple())
        
    @property
    def variable(self) -> str:
        return self._variable

    @property
    def string_variables(self) -> tuple[StringVariables, ...]:
        return self._string_variables

    @property    
    def unique_string_variables(self) -> set[int]:
        return {
            i
            for tup in self.string_variables
            for i in tup
        }


class MCFGRuleElementInstance:
    """An instantiated multiple context free grammar rule element

    Parameters
    ----------
    symbol:  variable : str
        The non-terminal symbol being instantiated.
    string_spans : tuple[SpanIndices, ...]
        Tuples of (start, end) indices representing the spans in the input string.


    Attributes
    ----------
    symbol: variable : str
        The non-terminal symbol.
    string_spans : tuple[SpanIndices, ...]
        Spans over the input string this element covers.
    """
    def __init__(self, variable: str, *string_spans: SpanIndices):
        self._variable = variable
        self._string_spans = string_spans

    def __eq__(self, other: 'MCFGRuleElementInstance') -> bool:
        vareq = self._variable == other._variable
        strspaneq = self._string_spans == other._string_spans
        
        return vareq and strspaneq
        
    def to_tuple(self) -> tuple[str, tuple[SpanIndices, ...]]:
        return (self._variable, self._string_spans)

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __str__(self):
        strspans = ', '.join(
            str(list(stup))
            for stup in self._string_spans
        )
        
        return f"{self._variable}({strspans})"

    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def variable(self) -> str:
        return self._variable

    @property
    def string_spans(self) -> tuple[SpanIndices, ...]:
        return self._string_spans
    

class MCFGRule:
    """A linear multiple context free grammar rule

    Parameters
    ----------
    left_side : MCFGRuleElement
        The left-hand side of the rule.
    right_side : tuple[MCFGRuleElement, ...]
        The right-hand side elements of the rule.

    Attributes
    ----------
    left_side : MCFGRuleElement
        The left-hand side of the rule.
    right_side : tuple[MCFGRuleElement, ...]
        The right-hand side elements of the rule.
    """

    def __init__(self, left_side: MCFGRuleElement, *right_side: MCFGRuleElement):
        self._left_side = left_side
        self._right_side = right_side

        self._validate()
    
    def is_terminal(self):
        return len(self._right_side) == 1 and isinstance(self._right_side[0], MCFGRuleElement) and not self._right_side[0].string_variables

    def to_tuple(self) -> tuple[MCFGRuleElement, tuple[MCFGRuleElement, ...]]:
        return (self._left_side, self._right_side)

    def __hash__(self) -> int:
        return hash(self.to_tuple())
    
    def __repr__(self) -> str:
        return '<Rule: '+str(self)+'>'
        
    def __str__(self) -> str:
        if self.is_epsilon:
            return str(self._left_side)                

        else:
            return str(self._left_side) +\
                ' -> ' +\
                ' '.join(str(el) for el in self._right_side)

    def __eq__(self, other: 'MCFGRule') -> bool:
        left_side_equal = self._left_side == other._left_side
        right_side_equal = self._right_side == other._right_side

        return left_side_equal and right_side_equal

    def _validate(self):
        vs = [
            el.unique_string_variables
            for el in self.right_side
        ]
        sharing = any(
            vs1.intersection(vs2)
            for i, vs1 in enumerate(vs)
            for j, vs2 in enumerate(vs)
            if i < j
        )

        if sharing:
            raise ValueError(
                'right side variables cannot share '
                'string variables'
            )

        if not self.is_epsilon:
            left_vars = self.left_side.unique_string_variables
            right_vars = {
                var for el in self.right_side
                for var in el.unique_string_variables
            }
            if left_vars != right_vars:
                raise ValueError(
                    'number of arguments to instantiate must '
                    'be equal to number of unique string_variables'
                )
        
    @property
    def left_side(self) -> MCFGRuleElement:
        return self._left_side

    @property
    def right_side(self) -> tuple[MCFGRuleElement, ...]:
        return self._right_side

    @property
    def is_epsilon(self) -> bool:
        return len(self._right_side) == 0

    @property
    def unique_variables(self) -> set[str]:
        return {
            el.variable
            for el in [self._left_side]+list(self._right_side)
        }

    def instantiate_left_side(self, *right_side: MCFGRuleElementInstance) -> MCFGRuleElementInstance:
        """Instantiate the left side of the rule given an instantiated right side

        Parameters
        ----------
        right_side : MCFGRuleElementInstance
            The instantiated right side of the rule.

        Returns
        -------
        MCFGRuleElementInstance
            The instantiated left-side element.
        """
        
        if self.is_epsilon:
            strvars = tuple(v[0] for v in self._left_side.string_variables)
            strconst = tuple(el.variable for el in right_side)
            
            if strconst == strvars:
                return MCFGRuleElementInstance(
                    self._left_side.variable,
                    *[s for el in right_side for s in el.string_spans]
                )

        new_spans = []
        span_map = self._build_span_map(right_side)
        
        for vs in self._left_side.string_variables:
            for i in range(1,len(vs)):
                end_prev = span_map[vs[i-1]][1]
                begin_curr = span_map[vs[i]][0]

                if end_prev != begin_curr:
                    raise ValueError(
                        f"Spans {span_map[vs[i-1]]} and {span_map[vs[i]]} "
                        f"must be adjacent according to {self} but they "
                        "are not."
                    )
                
            begin_span = span_map[vs[0]][0]
            end_span = span_map[vs[-1]][1]

            new_spans.append((begin_span, end_span))

        return MCFGRuleElementInstance(
            self._left_side.variable, *new_spans
        )

    
    def _build_span_map(self, right_side: tuple[MCFGRuleElementInstance, ...]) -> SpanMap:
        """Construct a mapping from string variables to string spans
        Parameters
        ----------
        right_side : tuple[MCFGRuleElementInstance, ...]
            Instantiated elements of the rule's right-hand side.

        Returns
        -------
        SpanMap
            A dictionary mapping string variable indices to string spans.

        """
        
        if self._right_side_aligns(right_side):
            return {
                strvar[0]: strspan
                for elem, eleminst in zip(
                    self._right_side,
                    right_side
                )
                for strvar, strspan in zip(
                    elem.string_variables,
                    eleminst.string_spans
                )
            }
        else:
            raise ValueError(
                f"Instantiated right side {right_side} do not "
                f"align with rule's right side {self._right_side}"
            )

    def _right_side_aligns(self, right_side: tuple[MCFGRuleElementInstance, ...]) -> bool:
        """Check whether the right side aligns
        Parameters
        ----------
        right_side : tuple[MCFGRuleElementInstance, ...]
            Instantiated rule elements.

        Returns
        -------
        bool
            True if the structure matches, False otherwise.
        """

        if len(right_side) == len(self._right_side):
            vars_match = all(
                elem.variable == eleminst.variable
                for elem, eleminst in zip(self._right_side, right_side)
            )
            strvars_match = all(
                len(elem.string_variables) == len(eleminst.string_spans)
                for elem, eleminst in zip(self._right_side, right_side)
            )

            return vars_match and strvars_match
        else:
            return False 

    @classmethod
    def from_string(cls, rule_string) -> 'MCFGRule':
        """
        Parse a rule from its string representation.

        Parameters
        ----------
        rule_string : str
            A string representation of a rule, e.g., "S(01) -> NP(0) VP(1)".

        Returns
        -------
        MCFGRule
            The constructed rule object.
        """
        elem_strs = re.findall(r'(\w+)\(((?:\w+,? ?)+?)\)', rule_string)

        elem_tuples = [(var, [v.strip()
                              for v in svs.split(',')])
                       for var, svs in elem_strs]

        if len(elem_tuples) == 1:
            return cls(MCFGRuleElement(elem_tuples[0][0],
                                   tuple(w for w in elem_tuples[0][1])))

        else:
            strvars = [v for _, sv in elem_tuples[1:] for v in sv]

            # no duplicate string variables
            try:
                assert len(strvars) == len(set(strvars))
            except AssertionError:
                msg = 'variables duplicated on right side of '+rule_string
                raise ValueError(msg)

            
            elem_left = MCFGRuleElement(elem_tuples[0][0],
                                    *[tuple([strvars.index(v)
                                             for v in re.findall('('+'|'.join(strvars)+')', vs)])
                                      for vs in elem_tuples[0][1]])

            elems_right = [MCFGRuleElement(var, *[(strvars.index(sv),)
                                              for sv in svs])
                           for var, svs in elem_tuples[1:]]

            return cls(elem_left, *elems_right)
        
    def string_yield(self):
        """
        Get the string yield of an epsilon rule.

        Returns
        -------
        str
        The yield (output string) of the epsilon rule, which is simply the variable name.
        """
        if self.is_epsilon:
            return self._left_side.variable
        else:
            raise ValueError(
                'string_yield is only implemented for epsilon rules'
            )
from typing import List,Tuple
def MCFGGrammar(grammar_text: str) -> List[MCFGRule]:
    rules = []
    for line in grammar_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rules.append(MCFGRule.from_string(line))
    print(rules)
    return rules

class MCFGChart:
    def __init__(self):
        self.chart = defaultdict(list)

    def add(self, symbol: str, spans: Tuple[Tuple[int, int], ...]):
        instance = MCFGRuleElementInstance(symbol, *spans)
        print("inMCFGCHart",instance)
        if instance not in self.chart[spans]:
            
            self.chart[spans].append(instance)
            return instance
        return None

    def get(self, spans: Tuple[Tuple[int, int], ...]) -> List[MCFGRuleElementInstance]:
        return self.chart.get(spans, [])

    def all_entries(self) -> List[MCFGRuleElementInstance]:
        return [item for sublist in self.chart.values() for item in sublist]


class MCFGParser:
    def __init__(self, rules: List[MCFGRule]):
        self.rules = rules
        self.rules_by_rhs = self._index_rules_by_rhs()

    def _index_rules_by_rhs(self):
        index = defaultdict(list)
        for rule in self.rules:
            key = tuple(r.variable for r in rule.right_side)
            index[key].append(rule)
        return index

    def parse(self, tokens: List[str]) -> List[MCFGRuleElementInstance]:
        n = len(tokens)
        chart = MCFGChart()

        # Insert terminal rules
        for i, token in enumerate(tokens):
            for rule in self.rules:
                if rule.is_terminal():
                    terminal = rule.left_side.string_variables[0][0]
                    if terminal == token:
                        instance = MCFGRuleElementInstance(rule.left_side.variable, (i, i+1))
                        chart.add(instance.variable, instance.string_spans)


        # Combine spans bottom-up
        for length in range(1, n+1):
            for i in range(n - length + 1):
                j = i + length
                for k in range(i+1, j):
                    left_spans = [(inst.variable, inst) for inst in chart.get(((i, k),))]
                    right_spans = [(inst.variable, inst) for inst in chart.get(((k, j),))]
                    for (left_var, left_inst) in left_spans:
                        for (right_var, right_inst) in right_spans:
                            key = (left_var, right_var)
                            print("rule",rule)
                            for rule in self.rules_by_rhs.get(key, []):
                                print(f"Trying to match key: {key}")
                                print(f"Available rules for key: {self.rules_by_rhs.get(key)}")
                                try:
                                    combined = rule.instantiate_left_side(left_inst, right_inst)
                                    print(f"Instantiated: {combined}")
                                    chart.add(combined.variable, combined.string_spans)
                                except ValueError:
                                    continue

        return chart.all_entries()
    def recognize(self, tokens: List[str], start_symbol: str = "S") -> bool:
        """
        Check if the given tokens can be derived from the grammar's start symbol.

        Parameters
        ----------
        tokens : List[str]
            The input string as a list of terminal symbols.
        start_symbol : str
            The start symbol of the grammar (default is 'S').

        Returns
        -------
        bool
            True if the string is recognized by the grammar, False otherwise.
        """
        final_instances = self.parse(tokens)
        
        for inst in final_instances:
            # print(inst, inst.variable, start_symbol, len(inst.string_spans), inst.string_spans[0])
            if (inst.variable == start_symbol and
                len(inst.string_spans) == 1 and
                inst.string_spans[0] == (0, len(tokens))):
                return True

        return False


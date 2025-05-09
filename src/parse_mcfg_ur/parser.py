from enum import Enum

from typing import Dict
from collections import defaultdict

from typing import Literal
from functools import lru_cache

from enum import Enum

class NormalForm(Enum):
    CNF = 0
    BNF = 1
    GNF = 2

class Rule:
    """A context free grammar rule

    Parameters
    ----------
    left_side
    right_side
    """

    def __init__(self, left_side: str, *right_side: str):
        self._left_side = left_side
        self._right_side = right_side

    def __repr__(self) -> str:
        return self._left_side + ' -> ' + ' '.join(self._right_side)
        
    def to_tuple(self) -> tuple[str, tuple[str, ...]]:
        return (self._left_side, self._right_side)
        
    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __eq__(self, other: 'Rule') -> bool:
        left_side_equal = self._left_side == other._left_side
        right_side_equal = self._right_side == other._right_side

        return left_side_equal and right_side_equal

    def validate(self, alphabet: set[str], variables: set[str], normal_form: NormalForm = NormalForm.CNF):
        """validate the rule

        Parameters
        ----------
        alphabet : set(str)
        variables : set(str)
        """

        if self._left_side not in variables:
            msg = "left side of rule must be a variable"
            raise ValueError(msg)

        acceptable = alphabet | variables | {''}
            
        if not all([s in acceptable for s in self._right_side]):
            msg = "right side of rule must contain only" +\
                  "a variable, a symbol, or the empty string"
            raise ValueError(msg)

        if normal_form == NormalForm.CNF:
            try:
                if len(self.right_side) == 1:
                    assert self.right_side[0] in alphabet
                elif len(self.right_side) == 2:
                    assert all([s in variables for s in self.right_side])
                else:
                    raise AssertionError

            except AssertionError:
                raise ValueError(f"{self} is not in CNF")
        

    @property
    def left_side(self) -> str:
        return self._left_side

    @property
    def right_side(self) -> tuple[str, ...]:
        return self._right_side

    @property
    def is_unary(self) -> bool:
        return len(self._right_side) == 1
    
    @property
    def is_binary(self) -> bool:
        return len(self._right_side) == 2



Mode = Literal["recognize", "parse"]

class ContextFreeGrammar:

    """
    A context free grammar

    Parameters
    ----------
    alphabet : set(str)
    variables : set(str)
    rules : set(Rule)
    start_variable : str

    Attributes
    ----------
    alphabet : set(str)
    variables : set(str)
    rules : set(Rule)
    start_variable : str

    Methods
    -------
    reduce(left_side)
    """
    
    # this will need to be filled in by the parser class, once it is defined:
    # - CKYParser
    # - EarleyParser
    parser_class = None
    
    def __init__(self, alphabet: set[str], variables: set[str], rules: set[Rule], start_variable: str):
        self._alphabet = alphabet
        self._variables = variables
        self._rules = rules
        self._start_variable = start_variable
        
        self._validate_variables()
        self._validate_rules()

        if self.parser_class is not None:
            self._parser = self.parser_class(self)
        else:
            self._parser = None

    def __call__(self, string: str | list[str], mode: Mode = "recognize"):
        if self._parser is not None:
            return self._parser(string, mode)
        else:
            raise AttributeError("no parser is specified")
        
    def _validate_variables(self):
        if self._alphabet & self._variables:
            raise ValueError('alphabet and variables must not share elements')
        
        if self._start_variable not in self._variables:
            raise ValueError('start variable must be in set of variables')

    def _validate_rules(self):
        if self.parser_class is not None:
            for r in self._rules:
                r.validate(self._alphabet, self._variables,
                           self.parser_class.normal_form)

    @property            
    def alphabet(self) -> set[str]:
        return self._alphabet

    @property    
    def variables(self) -> set[str]:
        return self._variables
   
    @lru_cache(2**10)
    def rules(self, left_side: str | None = None) -> set[Rule]:
        if left_side is None:
            return self._rules
        else:
            return {rule for rule in self._rules 
                    if rule.left_side == left_side}

    @property
    def start_variable(self) -> str:
        return self._start_variable

    @lru_cache(2**14)
    def parts_of_speech(self, word: str | None = None) -> set[str]:
        if word is None:
            return {rule.left_side for rule in self._rules 
                    if rule.is_unary 
                    if rule.right_side[0] in self._alphabet}
        
        else:
            return {rule.left_side for rule in self._rules 
                    if rule.is_unary 
                    if rule.right_side[0] == word}
  
    @property
    def phrase_variables(self) -> set[str]:
        try:
            return self._phrase_variables
        except AttributeError:
            self._phrase_variables = {rule.left_side for rule in self._rules 
                                      if not rule.is_unary or 
                                      rule.right_side[0] not in self._alphabet}
            return self._phrase_variables

    @lru_cache(2^15)
    def reduce(self, *right_side: str) -> set[str]:
        """
        the nonterminals that can be rewritten as right_side

        Parameters
        ----------
        right_side

        Returns
        -------
        set(str)
        """
        return {r.left_side for r in self._rules
                if r.right_side == tuple(right_side)}
    
class ContextFreeGrammar(ContextFreeGrammar):
    
    @classmethod
    def from_treebank(cls, treebank: TreeBank) -> 'ContextFreeGrammar':
        rules = set()
        variables = set()
        alphabet = set()
    
        for _, tree in treebank:
            for rule in tree.rules:
                rules.add(rule)
                variables.add(rule.left_side)
                for sym in rule.right_side:
                    if sym.islower() or sym in {',', '.', '?', '!'}:
                        alphabet.add(sym)
                    elif sym:  # ignore empty
                        variables.add(sym)
    
        return cls(alphabet=alphabet,
                   variables=variables,
                   rules=rules,
                   start_variable="S")

SpanIndices = tuple[int, int]
CKYBackPointer = tuple[str, SpanIndices]

class Chart(ABC):

    @property
    def parses(self):
        raise NotImplementedError

class ChartEntry(ABC):

    def __hash__(self) -> int:
        raise NotImplementedError

    @property
    def backpointers(self):
        raise NotImplementedError

class CKYChartEntry(ChartEntry):
    """
    A chart entry for a CKY chart

    Parameters
    ----------
    symbol
    backpointers

    Attributes
    ----------
    symbol
    backpointers
    """

    def __init__(self, symbol: str, *backpointers: CKYBackPointer):
        self._symbol = symbol
        self._backpointers = backpointers

    def to_tuple(self):
        return (self._symbol, self._backpointers)
        
    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __eq__(self, other: 'CKYChartEntry') -> bool:
        return self.to_tuple() == other.to_tuple()
    
    def __repr__(self) -> str:
        return self._symbol + ' -> ' + ' '.join(
            f"{bp[0]}({bp[1][0]}, {bp[1][1]})" 
            for bp in self.backpointers
        )

    def __str__(self) -> str:
        return self.__repr__()
    
    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def backpointers(self) -> tuple[CKYBackPointer, ...]:
        return self._backpointers

class CKYChart(Chart):
    """
    A chart for a CKY parser

    Jurafsky & Martin call this a table

    Parameters
    ----------
    input_size

    Attributes
    ----------
    parses
    """

    def __init__(self, input_size: int):
        self._input_size = input_size
        
        self._chart: list[list[set[CKYChartEntry]]] = [
            [set({})
             for _ in range(input_size+1)]
            for _ in range(input_size+1)
        ]
        
    def __getitem__(self, index: SpanIndices) -> set[CKYChartEntry]:
        i, j = index

        self._validate_index(i, j)
        
        return self._chart[i][j]

    def __setitem__(self, key: SpanIndices, item: set[CKYChartEntry]):
        i, j = key

        self._chart[i][j] = item
        
    def _validate_index(self, i, j):
        try:
            assert i < j
        except AssertionError:
            msg = "cannot index into the lower " +\
                  "triangle of the chart"
            raise ValueError(msg)

        try:
            self._chart[i]
        except IndexError:
            msg = "row index is too large"
            raise ValueError(msg)

        try:
            self._chart[i][j]
        except IndexError:
            msg = "column index is too large"
            raise ValueError(msg)

    @property
    def parses(self) -> set[Tree]:
        try:
            return self._parses
        except AttributeError:
            self._parses = self._construct_parses()
            return self._parses

    def _construct_parses(self, entry: Union['CKYChartEntry', None] = None) -> set[Tree]:
        """Construct the parses implied by the chart

        Parameters
        ----------
        entry
        """
        raise NotImplementedError
    
class ContextFreeGrammarParser(ABC):
    
    def __init__(self, grammar: ContextFreeGrammar):
        self._grammar = grammar

    def __call__(self, string, mode="recognize"):
        if mode == "recognize":
            return self._recognize(string)
        elif mode == "parse":
            return self._parse(string)            
        else:
            msg = 'mode must be "parse" or "recognize"'
            raise ValueError(msg)

class CKYParser(ContextFreeGrammarParser):
    """
    A CKY parser

    Parameters
    ----------
    grammar : ContextFreeGrammar
    """
    
    normal_form = NormalForm.CNF
    
    def _fill_chart(self, string: list[str]) -> CKYChart:
        raise NotImplementedError

    def _parse(self, string):
        chart = self._fill_chart(string)
        return chart.parses
        
    def _recognize(self, string):
        chart = self._fill_chart(string)
        
        return any([self._grammar.start_variable == entry.symbol
                    for entry in chart[0,len(string)]])

class CKYParser(CKYParser):
    
    normal_form = NormalForm.CNF

    def _fill_chart(self, string: list[str]) -> CKYChart:
        n = len(string)
        chart = CKYChart(n)

        # Precompute rules by RHS length and content for efficiency
        unary_rules = [rule for rule in self._grammar.rules() if len(rule.right_side) == 1]
        binary_rules = [rule for rule in self._grammar.rules() if len(rule.right_side) == 2]


        # Fill the diagonal (span length 1): terminal rules
        for i in range(n):
            word = string[i].lower()
            for rule in unary_rules:
                if rule.right_side[0] == word:
                    entry = CKYChartEntry(rule.left_side)
                    chart[i, i+1].add(entry)

        # Fill the rest of the chart (span length >= 2)
        for span_len in range(2, n+1):
            for i in range(n - span_len + 1):
                j = i + span_len
                for k in range(i+1, j):
                    B_entries = chart[i, k]
                    C_entries = chart[k, j]
                    for B in B_entries:
                        for C in C_entries:
                            for rule in binary_rules:
                                if rule.right_side == (B.symbol, C.symbol):
                                    entry = CKYChartEntry(rule.left_side,
                                                          (B.symbol, (i, k)),
                                                          (C.symbol, (k, j)))
                                    chart[i, j].add(entry)

        return chart

    
class CKYChart(CKYChart):

    def _construct_parses(self, entry: CKYChartEntry | None = None) -> set[Tree]:
        if entry is None:
            parses = set()
            for e in self[0, self._input_size]:
                if hasattr(self, '_grammar') and e.symbol == self._grammar.start_variable:
                    parses.update(self._construct_parses(e))
                elif not hasattr(self, '_grammar') or e.symbol == "S":  # fallback
                    parses.update(self._construct_parses(e))
            return parses

        if not entry.backpointers:
            # Terminal symbol case
            return {Tree(entry.symbol, [Tree("TOKEN")])}  # Placeholder; adjust if tokens available
        else:
            children_sets = []
            for sym, (i, j) in entry.backpointers:
                child_entries = [e for e in self[i, j] if e.symbol == sym]
                children_sets.append([self._construct_parses(e) for e in child_entries])

            # Cartesian product of all children parse sets
            from itertools import product
            parses = set()
            for combo in product(*children_sets):
                for subtrees in product(*combo):
                    parses.add(Tree(entry.symbol, list(subtrees)))
            return parses

class CKYParser(CKYParser):
    
    normal_form = NormalForm.CNF

    def _fill_chart(self, string: list[str]) -> CKYChart:
        n = len(string)
        chart = CKYChart(n)

        # Precompute rules by RHS length and content for efficiency
        unary_rules = [rule for rule in self._grammar.rules() if len(rule.right_side) == 1]
        binary_rules = [rule for rule in self._grammar.rules() if len(rule.right_side) == 2]


        # Fill the diagonal (span length 1): terminal rules
        for i in range(n):
            word = string[i].lower()
            for rule in unary_rules:
                if rule.right_side[0] == word:
                    entry = CKYChartEntry(rule.left_side)
                    chart[i, i+1].add(entry)

        # Fill the rest of the chart (span length >= 2)
        for span_len in range(2, n+1):
            for i in range(n - span_len + 1):
                j = i + span_len
                for k in range(i+1, j):
                    B_entries = chart[i, k]
                    C_entries = chart[k, j]
                    for B in B_entries:
                        for C in C_entries:
                            for rule in binary_rules:
                                if rule.right_side == (B.symbol, C.symbol):
                                    entry = CKYChartEntry(rule.left_side,
                                                          (B.symbol, (i, k)),
                                                          (C.symbol, (k, j)))
                                    chart[i, j].add(entry)

        return chart

class DottedRule(Rule):
    
    def __init__(self, rule: Rule, span: SpanIndices, dot: int = 0):
        self._rule = rule
        self._left_side = rule.left_side
        self._right_side = rule.right_side
        
        self._span = span
        self._dot = dot
    
    def to_tuple(self) -> tuple[Rule, SpanIndices, int]:
        return self._rule, self._span, self._dot
    
    def __hash__(self) -> int:
        return hash(self.to_tuple())
    
    def __eq__(self, other) -> bool:
        return self.to_tuple() == other.to_tuple()
    
    def __repr__(self) -> str:
        return self._left_side + ' -> ' +\
               ' '.join(self._right_side[:self._dot]) +\
               ' . ' +\
               ' '.join(self._right_side[self._dot:]) +\
               ' [' + str(self._span[0]) + ', ' + str(self._span[1]) + ']'
    
    def complete(self, new_left_edge: int) -> tuple['DottedRule', int]:
        """Complete the next symbol in this rule
        
        Parameters
        ----------
        new_left_edge

        Returns
        -------
        new_dotted_rule
        completed_symbol
        old_left_edge
        """
        dot = self._dot + 1
        span = (self._span[0], new_left_edge)

        return DottedRule(self._rule, span, dot)

    @property
    def next_symbol(self) -> str:
        if self.is_complete:
            raise AttributeError('dotted rule is already complete')
        else:
            return self._right_side[self._dot]
        
    @property
    def dot(self) -> int:
        return self._dot
    
    @property
    def span(self) -> SpanIndices:
        return self._span
    
    @property
    def is_complete(self) -> bool:
        return self._dot == len(self._right_side)
    
    @property
    def left_side(self) -> str:
        return self._rule.left_side
    
EarleyBackPointer = tuple[str, int]

class EarleyChartEntry(ChartEntry):
    """A chart entry for a Earley chart

    Parameters
    ----------
    dotted_rule
    backpointers
    """

    def __init__(self, dotted_rule: DottedRule, *backpointers: EarleyBackPointer):
        self._dotted_rule = dotted_rule
        self._backpointers = backpointers

    def to_tuple(self) -> tuple[DottedRule, tuple[EarleyBackPointer, ...]]:
        return self._dotted_rule, self._backpointers
        
    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __eq__(self, other) -> bool:
        return self.to_tuple() == other.to_tuple()
    
    def __repr__(self) -> str:
        return self._dotted_rule.__repr__()

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def backpointers(self) -> tuple[EarleyBackPointer, ...]:
        return self._backpointers
    
    @property
    def dotted_rule(self):
        return self._dotted_rule
    
    @property
    def next_symbol(self) -> str:
        return self._dotted_rule.next_symbol
    
    @property
    def span(self) -> tuple[int]:
        return self._dotted_rule.span
    
    @property
    def is_complete(self):
        return self._dotted_rule.is_complete
    
    def complete(self, entry: 'EarleyChartEntry', new_left_edge: int) -> 'EarleyChartEntry':
        new_dotted_rule = self._dotted_rule.complete(new_left_edge)
        # print("at bp",new_dotted_rule)
        
        bp = (self._dotted_rule.next_symbol, self._dotted_rule.span[1])
        # print("bp",bp)
        backpointers = self._backpointers + (bp,)
        
        return EarleyChartEntry(new_dotted_rule, backpointers)
    
    def is_completion_of(self, other: 'EarleyChartEntry') -> bool:
        return self._dotted_rule.left_side == other.dotted_rule.next_symbol

class EarleyChart(Chart):
    """A chart for an Earley parser

    Parameters
    ----------
    input_size
    """
    def __init__(self, input_size: int, start_variable: str = 'S'):
        self._start_variable = start_variable
        
        self._chart: list[set[EarleyChartEntry]] = [
            set() for _ in range(input_size+1)
        ]
        
    def __getitem__(self, index) -> set[EarleyChartEntry]:
        return self._chart[index]

    def __setitem__(self, key, item) -> None:
        self._chart[key] = item

    @property
    def parses(self) -> set[Tree]:
        try:
            return self._parses
        except AttributeError:
            self._parses = set()
            
            for entry in self._chart[-1]:
                is_start = entry.dotted_rule.left_side == self._start_variable
                covers_string = entry.dotted_rule.span == (0, self.input_size)
                
                if is_start and covers_string:
                    self._parses.add(self._construct_parses(entry))
            
            return self._parses

    # def _construct_parses(self, entry: 'EarleyChartEntry') -> Tree:
    #     """Construct the parses implied by the chart

    #     Parameters
    #     ----------
    #     entry
    #     """
    #     raise NotImplementedError     
    
    @property
    def input_size(self) -> int:
        return len(self._chart) - 1
    
class EarleyParser(ContextFreeGrammarParser):
    """
    An Earley parser

    Parameters
    ----------
    grammar : ContextFreeGrammar
    """
    normal_form = None
                    
    def _predict(self, chart: EarleyChart, entry: EarleyChartEntry, position: int):
        for rule in self._grammar.rules(entry.next_symbol):
            span = (position, position)
            dotted_rule = DottedRule(rule, span)
            entry = EarleyChartEntry(dotted_rule)

            chart[position].add(entry)
            
    def _scan(self, chart: EarleyChart, entry: EarleyChartEntry, position: int):

        word = self._string[position]
        pos_for_word = self._grammar.parts_of_speech(word)
        # print(entry)
        if entry.next_symbol == word:
            scanned_entry = entry.complete(entry, position + 1)
            chart[position+1].add(scanned_entry)
        
        if entry.next_symbol in pos_for_word:
            rule = Rule(entry.next_symbol, word)
            span = (position, position+1)
            dotted_rule = DottedRule(rule, span, dot=1)
            
            unary_entry = EarleyChartEntry(dotted_rule)
            
            chart[position+1].add(unary_entry)

        
    def _complete(self, chart: EarleyChart, entry: EarleyChartEntry, position: int):
        start, end = entry.span
        
        for prev_entry in chart[start]:
            if not prev_entry.is_complete and entry.is_completion_of(prev_entry):
                completed_entry = prev_entry.complete(entry, end)
                
                chart[position].add(completed_entry)
        
    def _parse(self, string: str | list[str]):
        chart = self._fill_chart(string)
        return chart.parses

    def _recognize(self, string: str | list[str]):
        chart = self._fill_chart(string)
        
        for entry in chart[-1]:
            is_start = entry.dotted_rule.left_side == self._grammar.start_variable
            covers_string = entry.dotted_rule.span == (0, chart.input_size)
            
            if is_start and covers_string:
                return True
            
        else:
            return False
        
class EarleyParser(EarleyParser):
    
    def _fill_chart(self, string: list[str]) -> EarleyChart:
        self._string = string
        chart = EarleyChart(len(string), self._grammar.start_variable)

        # Add dotted start rules to chart[0]
        for rule in self._grammar.rules(self._grammar.start_variable):
            dotted = DottedRule(rule, (0, 0))
            entry = EarleyChartEntry(dotted)
            chart[0].add(entry)

        # Main chart filling loop
        for i in range(len(string) + 1):
            changed = True
            while changed:
                changed = False
                current_entries = list(chart[i])
                for entry in current_entries:
                    if not entry.is_complete:
                        next_sym = entry.next_symbol
                        if next_sym.isupper():  # Predict
                            before = len(chart[i])
                            self._predict(chart, entry, i)
                            changed |= len(chart[i]) > before
                        elif i < len(string):  # Scan
                             # print("at scan", entry, i)
                            before = len(chart[i + 1])
                            self._scan(chart, entry, i)
                            changed |= len(chart[i + 1]) > before
                    else:  # Complete
                        before = len(chart[i])
                        # print("at complete", entry, i)
                        self._complete(chart, entry, i)
                        changed |= len(chart[i]) > before
                # print("chart",chart[i])
        return chart


        # raise NotImplementedError
    
class EarleyChart(EarleyChart):
    
    def _construct_parses(self, entry: 'EarleyChartEntry') -> Tree:
        """Construct the parses implied by the chart

        Parameters
        ----------
        entry : EarleyChartEntry

        Returns
        -------
        NLTK Tree
        """
        # Create the root node for this entry using the left-hand side of the dotted rule
        node = Tree(entry.dotted_rule.left_side, [])

        # Base case: terminal rule (dot is at the end, and RHS is a terminal string)
        if len(entry.dotted_rule.right_side) == 1 and isinstance(entry.dotted_rule.right_side[0], str):
            terminal_symbol = entry.dotted_rule.right_side[0]
            if terminal_symbol in self.grammar().alphabet:
                node.append(terminal_symbol)
                return node


        # If there are backpointers, recursively construct the children
        print(entry.backpointers)
        for bp1 in entry.backpointers[0]:
            if len(bp1)==1:
                bp1 = bp1[0]
            # for symbol, position in bp1:
            symbol = bp1[0]
            position = bp1[1]
            print(self[position])
            for potential_child in self[position]:
                # print(potential_child.dotted_rule.left_side, symbol, potential_child.is_complete)
                if (potential_child.dotted_rule.left_side == symbol and potential_child.is_complete):
                    print("here")
                    # Recursively construct the child's subtree
                    child = self._construct_parses(potential_child)
                    node.append(child)
                    break  # Stop after finding the first matching completed child

        return node

class EarleyParser(EarleyParser):
    
    def __call__(self, string, mode="recognize"):
        if mode == "recognize":
            return self._recognize(string)
        elif mode == "parse":
            return self._parse(string)  
        elif mode == "predict":
            return self._predict_next_word(string)  
        else:
            msg = 'mode must be "parse", "recognize", or "predict"'
            raise ValueError(msg)
    
    def _predict_next_word(self, prefix: list[str]) -> Dict[str, set[str]]:
        chart = self._fill_chart(prefix)
        position = len(prefix)

        next_symbols: Dict[str, set[str]] = defaultdict(set)

        for entry in chart[position]:
            if not entry.is_complete:
                try:
                    next_sym = entry.next_symbol
                except AttributeError:
                    continue

                # If next symbol is a terminal (i.e., word)
                if next_sym in self._grammar.alphabet:
                    next_symbols[next_sym].add(next_sym)

                # If next symbol is a non-terminal
                elif next_sym in self._grammar.variables:
                    for rule in self._grammar.rules(next_sym):
                        if len(rule.right_side) == 1:
                            rhs = rule.right_side[0]
                            if rhs in self._grammar.alphabet:
                                next_symbols[next_sym].add(rhs)

        return dict(next_symbols)

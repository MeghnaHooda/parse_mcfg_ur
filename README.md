# Multiple Context-Free Grammars Parser

This repository implements an **agenda-based parser** for **Multiple Context-Free Grammars (MCFGs)** as a final project for a computational linguistics course. The parser is based on the agenda-driven architecture described in **Shieber et al. (1995)** and uses inference rules laid out in **Kallmeyer (2013)**.

---

## 📚 Project Overview

This package provides a modular, testable system for parsing with MCFGs, which extend context-free grammars to handle more complex linguistic structures. The implementation supports rule definition, parsing, and parse tree construction, and includes a full test suite.

### Key Features

- Support for both terminal and nonterminal MCFG rules
- Agenda-based chart parsing algorithm
- Typed, modular code with detailed docstrings (NumPy style)
- Structured test suite covering all components

---

## 🗂️ Package Structure

```text
mcfg_parser/
├── grammar/        # Rule and grammar representation
│   └── MCFGRule, MCFGRuleElement, MCFGRuleElementInstance, MCFGGrammar
├── parser/         # Agenda-based parser and parsing logic
│   └── MCFGParser, MCFGChart
├── tree/           # Parse tree representation and construction
│   └── ParseTree
├── tests/          # Pytest-based test suite
│   ├── test_grammar.py
│   ├── test_parser.py
│   └── test_tree.py
└── README.md
```



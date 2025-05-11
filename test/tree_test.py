import pytest
from parse_mcfg_ur.tree import Tree


def test_tree_initialization():
    t = Tree("S", [Tree("NP"), Tree("VP")])
    assert t.data == "S"
    assert len(t.children) == 2
    assert t.children[0].data == "NP"


def test_tree_equality_and_hash():
    t1 = Tree("A", [Tree("B"), Tree("C")])
    t2 = Tree("A", [Tree("B"), Tree("C")])
    assert t1 == t2
    assert hash(t1) == hash(t2)


def test_tree_str_repr():
    t = Tree("A", [Tree("B"), Tree("C")])
    assert str(t) == "B C"
    assert "A" in repr(t)


def test_tree_to_tuple():
    t = Tree("X", [Tree("Y"), Tree("Z")])
    assert t.to_tuple() == ("X", (("Y", ()), ("Z", ())))


def test_tree_contains():
    t = Tree("S", [Tree("NP"), Tree("VP")])
    assert "NP" in t
    assert "VP" in t
    assert "V" not in t


def test_tree_getitem():
    t = Tree("S", [Tree("NP"), Tree("VP", [Tree("V"), Tree("NP")])])
    assert t[1].data == "VP"
    assert t[1, 0].data == "V"
    assert t[()].data == "S"


def test_tree_terminals():
    t = Tree("S", [Tree("NP", [Tree("John")]), Tree("VP", [Tree("eats"), Tree("apples")])])
    assert t.terminals == ["John", "eats", "apples"]


def test_tree_index():
    t = Tree("S", [Tree("NP", [Tree("N")]), Tree("VP", [Tree("V"), Tree("NP", [Tree("N")])])])
    indices = t.index("N")
    assert indices == [(0, 0), (1, 1, 0)]


def test_tree_relabel():
    t = Tree("S", [Tree("NP"), Tree("VP")])
    relabeled = t.relabel(lambda x: x.lower())
    assert relabeled.data == "s"
    assert relabeled.children[0].data == "np"


def test_tree_from_list():
    treelist = ["S", "NP", "VP"]
    t = Tree.from_list(treelist)
    assert t.data == "S"
    assert len(t.children) == 2
    assert t.children[1].data == "VP"

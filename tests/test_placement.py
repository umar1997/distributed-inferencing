"""Phase 1.3: reading expert lists, the layer -> expert -> node map, and the coverage check."""

import pytest

from moe_dist.experts.placement import CoverageError, Placement, parse_expert_ids

LAYERS = list(range(24))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0-10", list(range(11))),
        ("0,4,8,12", [0, 4, 8, 12]),
        ("12, 0 ,8", [0, 8, 12]),
        ("0-3,8,12-15", [0, 1, 2, 3, 8, 12, 13, 14, 15]),
        ("31", [31]),
        ("none", []),
        ("", []),
    ],
)
def test_parse_expert_ids(text, expected):
    assert parse_expert_ids(text, 32) == expected


@pytest.mark.parametrize(
    "text, message",
    [
        ("0-40", "out of range"),
        ("1,1", "more than once"),
        ("0-3,2", "more than once"),
        ("5-2", "bad part '5-2'"),
        ("a-b", "bad part"),
        ("1;2", "bad part"),
    ],
)
def test_bad_expert_text_gives_clear_errors(text, message):
    with pytest.raises(ValueError, match=message):
        parse_expert_ids(text, 32)


def three_nodes():
    return {"master": list(range(0, 10)), "worker-1": list(range(10, 21)), "worker-2": list(range(21, 32))}


def test_same_in_every_layer():
    p = Placement.same_in_every_layer(three_nodes(), LAYERS, 32)
    assert p.node_ids() == ["master", "worker-1", "worker-2"]
    for layer in LAYERS:
        assert p.owners[layer][15] == ["worker-1"]
        assert p.experts_of("worker-2", layer) == list(range(21, 32))
    p.check_coverage()  # complete: no error


def test_owner_table():
    p = Placement.same_in_every_layer(three_nodes(), LAYERS, 32)
    table, node_ids = p.owner_table(5)
    assert node_ids == ["master", "worker-1", "worker-2"]
    assert table.tolist() == [0] * 10 + [1] * 11 + [2] * 11


def test_missing_expert_in_all_layers():
    nodes = three_nodes()
    nodes["worker-1"].remove(17)
    p = Placement.same_in_every_layer(nodes, LAYERS, 32)
    assert p.missing() == {layer: [17] for layer in LAYERS}
    with pytest.raises(CoverageError, match=r"missing experts: \[17\] in all layers"):
        p.check_coverage()
    table, _ = p.owner_table(0)
    assert table[17] == -1


def test_missing_expert_in_some_layers():
    p = Placement.same_in_every_layer({"a": list(range(32))}, [0, 1, 2], 32)
    p.owners[1][5] = []
    with pytest.raises(CoverageError, match=r"layer 1: \[5\]"):
        p.check_coverage()


def test_node_without_experts_is_not_a_node():
    p = Placement.same_in_every_layer({"master": [], "w": list(range(32))}, LAYERS, 32)
    assert p.node_ids() == ["w"]


def test_backup_copy_uses_first_owner():
    p = Placement.same_in_every_layer({"a": list(range(32)), "b": [3]}, LAYERS, 32)
    assert p.owners[0][3] == ["a", "b"]
    table, node_ids = p.owner_table(0)
    assert node_ids[table[3]] == "a"

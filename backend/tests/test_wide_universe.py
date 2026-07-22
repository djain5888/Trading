"""Tests for the wide NSE universe config and cap-tier segmentation helpers."""

from __future__ import annotations

from app.backtest.validation import limit_by_tier, partition_by_tier
from app.config.watchlist import CAP_TIERS, load_wide_universe

# -- Universe loading ------------------------------------------------------


def test_wide_universe_loads_with_tiers_and_sectors() -> None:
    """The bundled wide universe parses into a sizeable, fully-annotated config."""
    config = load_wide_universe()

    assert 50 <= len(config.symbols) <= 100  # a wide, multi-cap universe
    assert config.index_symbol == "NIFTY"
    # Every symbol carries both a sector and a recognised cap tier.
    for symbol in config.symbols:
        assert symbol in config.sectors and config.sectors[symbol]
        assert config.tiers.get(symbol) in CAP_TIERS


def test_wide_universe_spans_every_cap_tier() -> None:
    """The universe covers large, mid and small caps (not just one bucket)."""
    tiers = load_wide_universe().tiers
    present = {tier for tier in tiers.values()}

    assert present == set(CAP_TIERS)
    for tier in CAP_TIERS:
        members = [s for s, t in tiers.items() if t == tier]
        assert len(members) >= 10  # a meaningful sample per tier


# -- Tier partitioning -----------------------------------------------------


def test_partition_by_tier_groups_and_orders() -> None:
    """Symbols are grouped by tier, ordered large -> mid -> small, order kept."""
    symbols = ["A", "B", "C", "D", "E"]
    tiers = {"A": "small", "B": "large", "C": "mid", "D": "large", "E": "small"}

    groups = partition_by_tier(symbols, tiers)

    assert list(groups) == ["large", "mid", "small"]
    assert groups["large"] == ["B", "D"]  # original order preserved within a tier
    assert groups["mid"] == ["C"]
    assert groups["small"] == ["A", "E"]


def test_partition_by_tier_drops_untiered_symbols() -> None:
    """A symbol with no known tier is excluded from every group."""
    groups = partition_by_tier(["A", "B"], {"A": "large"})

    assert groups == {"large": ["A"]}


# -- Balanced truncation ---------------------------------------------------


def test_limit_by_tier_is_balanced_across_tiers() -> None:
    """A limit is spread round-robin so each tier keeps roughly equal weight."""
    symbols = [f"L{i}" for i in range(5)] + [f"M{i}" for i in range(5)]
    symbols += [f"S{i}" for i in range(5)]
    tiers = {**{f"L{i}": "large" for i in range(5)}}
    tiers.update({f"M{i}": "mid" for i in range(5)})
    tiers.update({f"S{i}": "small" for i in range(5)})

    chosen = limit_by_tier(symbols, tiers, 6)

    assert len(chosen) == 6
    picked = partition_by_tier(chosen, tiers)
    # Two from each of the three tiers — perfectly balanced.
    assert {tier: len(members) for tier, members in picked.items()} == {
        "large": 2,
        "mid": 2,
        "small": 2,
    }
    # Original ordering is preserved in the result.
    assert chosen == sorted(chosen, key=symbols.index)


def test_limit_by_tier_returns_all_when_under_limit() -> None:
    """A limit at or above the universe size returns every symbol unchanged."""
    symbols = ["A", "B", "C"]
    tiers = {"A": "large", "B": "mid", "C": "small"}

    assert limit_by_tier(symbols, tiers, 10) == symbols
    assert limit_by_tier(symbols, tiers, 0) == []

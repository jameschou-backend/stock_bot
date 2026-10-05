from pathlib import Path

from skills.strategy_scanner.catalog import CATALOG, get_catalog


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ACTIVE = {
    "original_breakout", "original_red", "poc_red_priority", "poc_up_red",
    "momentum", "risk_momentum", "near_high", "contraction_breakout",
    "donchian20", "donchian55", "bollinger_reclaim", "ma_pullback",
}


def test_catalog_has_unique_complete_metadata_and_real_sources():
    assert isinstance(CATALOG, tuple)
    ids = [row["id"] for row in CATALOG]
    assert len(ids) == len(set(ids))
    fields = {"id", "name", "family", "kind", "status", "description",
              "required_data", "preferred_regimes", "version", "source_paths",
              "source_urls", "variants", "data_gaps", "reusable_interfaces"}
    kinds = {"entry", "ranking", "filter", "context", "exit", "allocation", "execution", "diagnostic"}
    for row in CATALOG:
        assert fields <= row.keys(), row["id"]
        for key in ("id", "name", "family", "kind", "description", "version"):
            assert isinstance(row[key], str) and row[key].strip(), (row["id"], key)
        assert row["kind"] in kinds
        assert row["status"] in {"active", "catalog_only"}
        for key in ("required_data", "preferred_regimes", "source_paths", "source_urls", "variants", "data_gaps", "reusable_interfaces"):
            assert isinstance(row[key], list)
            assert all(isinstance(value, str) and value.strip() for value in row[key])
        assert row["required_data"] and row["source_paths"]
        for relative in row["source_paths"]:
            assert not Path(relative).is_absolute(), (row["id"], relative)
            assert (ROOT / relative).is_file(), (row["id"], relative)
        assert all(url.startswith("https://") for url in row["source_urls"])
        assert set(row["preferred_regimes"]) <= {"trend_up", "trend_down", "range"}
        assert row["live_qualified"] is False
        assert row["returns_inherited"] is False
        if row["status"] == "catalog_only":
            assert row["data_gaps"], row["id"]


def test_active_catalog_exactly_matches_evaluator_not_old_factory():
    from skills.strategy_scanner.engine import ACTIVE_IDS

    active = {row["id"] for row in CATALOG if row["status"] == "active"}
    assert active == EXPECTED_ACTIVE == set(ACTIVE_IDS)
    legacy = [row for row in CATALOG if row["family"] == "legacy_factory"]
    assert len(legacy) == 6
    assert all(row["status"] == "catalog_only" for row in legacy)


def test_poc_priority_is_not_silently_promoted_to_hard_gate():
    rows = {row["id"]: row for row in CATALOG}
    assert rows["poc_red_priority"]["kind"] == "ranking"
    assert rows["poc_up_red"]["kind"] == "entry"
    assert "不是 POC 硬門檻" in rows["poc_red_priority"]["description"]
    assert "不含訊號日" in rows["poc_up_red"]["description"]


def test_research_families_and_non_entry_layers_remain_discoverable():
    families = {row["family"] for row in CATALOG}
    assert {
        "price_breakout", "volume_profile", "momentum", "mean_reversion",
        "pullback", "early_launch", "fundamental", "theme", "sector_rotation",
        "chip", "broker", "machine_learning", "legacy_factory", "entry_quality",
        "liquidity", "market_context", "chart_pattern", "exit", "reentry",
        "allocation", "execution", "diagnostic",
    } <= families
    for row in CATALOG:
        if row["kind"] in {"exit", "allocation", "execution", "diagnostic"}:
            assert row["status"] == "catalog_only"


def test_missing_pit_families_are_catalog_only_with_explicit_gap():
    rows = {row["id"]: row for row in CATALOG}
    for sid in ("revenue_growth", "revenue_surprise", "financial_quality",
                "guidance_surprise", "revenue_pead", "theme_catalyst",
                "news_event_group", "holder_strength", "institutional_absorption",
                "institutional_flow", "broker_concentration", "broker_persistence",
                "sector_breadth_flow", "chain_flow_context"):
        assert rows[sid]["status"] == "catalog_only"
        assert rows[sid]["data_gaps"]


def test_get_catalog_deep_copies_nested_lists_and_dicts():
    before = get_catalog()
    changed = get_catalog()
    changed[0]["id"] = "changed"
    changed[0]["required_data"].append("invented")
    changed[0]["source_paths"].clear()
    changed.append({"id": "new"})
    assert get_catalog() == before

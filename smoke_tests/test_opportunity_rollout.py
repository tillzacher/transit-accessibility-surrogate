"""Opportunity scoring cannot change any other relation."""

from dataclasses import replace
import numpy as np
import pytest
from modules.scoring_scenarios import ScenarioTile
from modules.generated_suite_scoring import score_set_channels


def fixture_tiles():
    tile = ScenarioTile(
        np.ones((2, 2)),
        np.ones((2, 2)),
        np.ones((2, 2, 2)),
        np.full((2, 2, 2), 10.0),
        np.ones((2, 2), bool),
        np.eye(2, dtype=bool),
    )
    names = [
        "connected_stop_with_poi",
        "isolated_stop_with_poi",
        "route_poi_reachable",
        "route_poi_unreachable",
        "stop_near_rail_connected",
        "stop_near_rail_isolated",
        "scaled_bus_full",
        "route_single_line",
        "route_redundant_overlap",
        "localized_route",
        "stub_edge_full_route",
        "stub_edge_origin_only",
    ]
    return {n: replace(tile) for n in names}


def test_rollout_changes_only_opportunity_measurement():
    tiles = fixture_tiles()
    tiles["route_poi_unreachable"].poi_raw = np.full((2, 2, 2), 9.0)
    null = replace(tiles["route_poi_reachable"], poi_raw=np.full((2, 2, 2), 9.0))
    old = score_set_channels(tiles, lambda t: t.poi_raw)
    new = score_set_channels(
        tiles,
        lambda t: t.poi_raw,
        opportunity_background=null,
        opportunity_eligible=np.ones(2, dtype=bool),
    )
    for key in old:
        if key != "poi_reachability_sensitivity":
            for field in old[key]:
                np.testing.assert_array_equal(old[key][field], new[key][field])
    assert not old["poi_reachability_sensitivity"]["pass"].any()
    assert new["poi_reachability_sensitivity"]["pass"].all()


def test_null_cannot_change_fixed_context():
    tiles = fixture_tiles()
    null = replace(tiles["route_poi_reachable"], rail_raw=np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="rail_raw"):
        score_set_channels(tiles, lambda t: t.poi_raw, opportunity_background=null)

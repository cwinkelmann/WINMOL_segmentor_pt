"""Amodal bridging must reconstruct the hidden trunk — and refuse when it shouldn't.

The refusals matter as much as the bridges. A shared `id` is an assumption about the
data, not a guarantee, so the cases where bridging declines are what stop a digitizing
slip from being welded into a 40 m phantom stem.
"""
import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("rasterio")
pytest.importorskip("fiona")

from shapely.affinity import rotate, translate  # noqa: E402
from shapely.geometry import box  # noqa: E402

from scripts.amodal_stems import bridge_tree  # noqa: E402


def _piece(x0, length=3.0, width=0.4, angle=0.0):
    """A stem fragment lying along +x, optionally rotated about the origin."""
    g = box(x0, -width / 2, x0 + length, width / 2)
    return rotate(g, angle, origin=(0, 0)) if angle else g


def test_bridges_a_gap_and_yields_one_connected_stem():
    # two 3 m fragments 2 m apart: the classic branch-occlusion break
    frags = [_piece(0.0), _piece(5.0)]
    geom, refused = bridge_tree(frags)

    assert not refused
    assert geom.geom_type == "Polygon", "bridged fragments must form ONE connected part"
    modal = sum(f.area for f in frags)
    assert geom.area > modal, "bridging must add the hidden span, not just merge"
    # the reconstructed span is the 2 m gap at roughly the stem width
    assert geom.area == pytest.approx(modal + 2.0 * 0.4, rel=0.25)


def test_bridge_follows_the_stem_at_any_orientation():
    """The axis is fitted, not assumed, so a diagonal stem must bridge just as well."""
    for angle in (0, 31, 67, 115, 174):
        frags = [_piece(0.0, angle=angle), _piece(5.0, angle=angle)]
        geom, refused = bridge_tree(frags)
        assert not refused, f"angle {angle} refused"
        assert geom.geom_type == "Polygon", f"angle {angle} left the stem disconnected"


def test_many_fragments_join_into_a_single_trunk():
    frags = [_piece(x) for x in (0.0, 4.0, 8.5, 13.0, 17.0)]
    geom, refused = bridge_tree(frags)

    assert not refused
    assert geom.geom_type == "Polygon"
    # end-to-end extent is preserved: 17 + 3 m of stem
    minx, _, maxx, _ = geom.bounds
    assert maxx - minx == pytest.approx(20.0, abs=0.1)


def test_refuses_a_gap_wider_than_max_gap():
    """A shared id 30 m apart is more likely a digitizing slip than one trunk."""
    frags = [_piece(0.0), _piece(33.0)]
    geom, refused = bridge_tree(frags, max_gap_m=20.0)

    assert [r["reason"] for r in refused] == ["gap_too_wide"]
    assert refused[0]["gap_m"] == pytest.approx(30.0, abs=0.1)
    assert geom.geom_type == "MultiPolygon", "a refused bridge must leave them separate"


def test_refuses_fragments_that_are_not_collinear():
    """Two ids that happen to collide should not be welded across open ground."""
    frags = [_piece(0.0), translate(_piece(5.0), yoff=6.0)]
    geom, refused = bridge_tree(frags, max_offset=2.0)

    assert refused and refused[0]["reason"] == "not_collinear"
    assert geom.geom_type == "MultiPolygon"


def test_single_fragment_is_returned_untouched():
    frag = _piece(0.0)
    geom, refused = bridge_tree([frag])
    assert not refused
    assert geom.area == pytest.approx(frag.area)


def test_touching_fragments_need_no_bridge():
    frags = [_piece(0.0, length=3.0), _piece(3.0, length=3.0)]
    geom, refused = bridge_tree(frags)
    assert not refused
    assert geom.area == pytest.approx(6.0 * 0.4, rel=0.02), "no phantom area added"


def test_taper_is_preserved_across_the_gap():
    """A trunk narrows toward the crown; the bridge should not fatten it back up."""
    frags = [_piece(0.0, width=0.6), _piece(5.0, width=0.2)]
    geom, _ = bridge_tree(frags)

    # bridge half-width is the mean of the two fragment widths, so the reconstructed
    # span sits between them rather than taking the wider one
    added = geom.area - sum(f.area for f in frags)
    assert 2.0 * 0.2 < added < 2.0 * 0.6

"""Tests for projection operators PR1 and PR2."""

import numpy as np
import pytest


class TestPR1:
    """Test PR1 — projection onto unit-circle arc [l, u]."""

    def test_inside_arc_normalizes(self):
        """Point inside arc should be normalized to unit modulus."""
        # Stub — implementation after Step 2
        pass

    def test_outside_arc_snaps_to_nearest(self):
        """Point outside arc should snap to nearest boundary."""
        pass

    def test_full_circle_returns_normalized(self):
        """When arc is full circle, any point normalizes."""
        pass

    def test_zero_input_returns_midpoint(self):
        """Zero-magnitude input should return arc midpoint."""
        pass


class TestPR2:
    """Test PR2 — projection onto convex hull of arc."""

    def test_inside_hull_unchanged(self):
        """Point inside convex hull (M1) should be unchanged."""
        pass

    def test_m5_normalizes_to_circle(self):
        """Point outside circle + arc side (M5) should normalize."""
        pass

    def test_full_circle_projects_to_disk(self):
        """Full circle arc projects to unit disk."""
        pass

    def test_chord_projection_m4(self):
        """Point below chord (M4) projects onto chord."""
        pass

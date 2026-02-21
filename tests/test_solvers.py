"""Tests for LB and UB solvers."""

import numpy as np
import pytest


class TestLBSolverCVXPY:
    """Test QP-LB via CVXPY."""

    def test_feasible_returns_finite(self):
        """Feasible problem should return finite LB."""
        pass

    def test_relaxation_lower_than_ub(self):
        """LB should always be <= UB for the same region."""
        pass


class TestLBSolverGP:
    """Test QP-LB via gradient projection."""

    def test_converges_to_cvxpy(self):
        """GP-LB should converge to similar value as CVXPY-LB."""
        pass


class TestUBSolverSLSQP:
    """Test QP-UB via SLSQP."""

    def test_feasible_constant_modulus(self):
        """Solution should have unit modulus for all elements."""
        pass


class TestUBSolverGP:
    """Test QP-UB via gradient projection."""

    def test_monotone_decrease(self):
        """Objective should not increase across GP iterations."""
        pass

"""
Validate the qutrit parameter-shift rule: ±2π/3 shifts.
"""

import numpy as np
import pytest
from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.simulator.statevector_sim import StatevectorSimulator
from triflowqc.core.variable import Variable
from triflowqc.gates.parametric import R_GM


SHIFT = 2 * np.pi / 3
SIN_SHIFT = np.sin(SHIFT)  # sin(2π/3) = √3/2


class TestParameterShiftRule:
    """Validate the qutrit parameter-shift rule numerically."""

    def _finite_diff_gradient(self, circuit_fn, theta, h=1e-5):
        """Numerical finite-difference gradient for comparison."""
        f_plus = circuit_fn(theta + h)
        f_minus = circuit_fn(theta - h)
        return (f_plus - f_minus) / (2 * h)

    def _param_shift_gradient(self, circuit_fn, theta):
        """Parameter-shift gradient with ±2π/3 shift."""
        f_plus = circuit_fn(theta + SHIFT)
        f_minus = circuit_fn(theta - SHIFT)
        return (f_plus - f_minus) / (2 * SIN_SHIFT)

    def test_single_rotation_gradient(self):
        """
        Test parameter-shift gradient for a single R_GM rotation.
        
        Circuit: R_GM(k, θ) on |0⟩, measure P(0).
        """
        sim = StatevectorSimulator()

        for k in range(1, 9):
            def circuit_fn(angle, _k=k):
                qc = QutritCircuit(1)
                qc.r_gm(_k, float(angle), 0)
                result = sim.run(qc)
                return float(result.probabilities[0])

            for theta_val in [0.5, 1.0, 2.0, 3.0]:
                ps_grad = self._param_shift_gradient(circuit_fn, theta_val)
                fd_grad = self._finite_diff_gradient(circuit_fn, theta_val)
                assert np.isclose(ps_grad, fd_grad, atol=1e-4), \
                    f"k={k}, θ={theta_val}: PS={ps_grad:.6f}, FD={fd_grad:.6f}"

    def test_two_parameter_circuit(self):
        """
        Test gradient for a circuit with two parametric gates.
        """
        sim = StatevectorSimulator()

        def observable(theta1, theta2):
            qc = QutritCircuit(1)
            qc.rx3(float(theta1), 0)
            qc.rz3(float(theta2), 0)
            result = sim.run(qc)
            return float(result.probabilities[0])

        theta1, theta2 = 1.0, 0.5

        # Gradient w.r.t. theta1
        def f1(t):
            return observable(t, theta2)

        ps_grad1 = self._param_shift_gradient(f1, theta1)
        fd_grad1 = self._finite_diff_gradient(f1, theta1)
        assert np.isclose(ps_grad1, fd_grad1, atol=1e-4)

        # Gradient w.r.t. theta2
        def f2(t):
            return observable(theta1, t)

        ps_grad2 = self._param_shift_gradient(f2, theta2)
        fd_grad2 = self._finite_diff_gradient(f2, theta2)
        assert np.isclose(ps_grad2, fd_grad2, atol=1e-4)

    def test_gradient_at_zero(self):
        """
        At θ=0, the gradient of P(0) for R_GM(k, θ)|0⟩ should be 0
        for diagonal generators (k=3, 8) since they don't mix |0⟩.
        """
        sim = StatevectorSimulator()

        for k in [3, 8]:
            def circuit_fn(angle, _k=k):
                qc = QutritCircuit(1)
                qc.r_gm(_k, float(angle), 0)
                result = sim.run(qc)
                return float(result.probabilities[0])

            grad = self._param_shift_gradient(circuit_fn, 0.0)
            # For diagonal generators, P(0) = cos²(θ/2) at leading order
            # d/dθ cos²(θ/2) at θ=0 = 0
            assert np.isclose(grad, 0.0, atol=1e-4), \
                f"k={k}: gradient at θ=0 should be ~0, got {grad}"

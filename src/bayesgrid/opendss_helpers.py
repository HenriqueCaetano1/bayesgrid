import numpy as np
from scipy.optimize import minimize_scalar

# --- Helper Dictionaries for Phase Handling ---
PHASE_CODE_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'AB', 4: 'BC', 5: 'CA', 6: 'ABC'}
PHASE_TO_DSS_NODE = {'A': '.1', 'B': '.2', 'C': '.3'}
PHASE_TO_MATRIX_IDX = {'A': 0, 'B': 1, 'C': 2}


def carson_modificada_Zabc(R1, X1, distancias):
    """Calculates the 3x3 Zabc matrix from positive sequence impedance."""
    dab, dbc, dac = distancias
    posicoes = {'a': (0, 0),'b': (dab, 0),'c': (dac, 0)}
    Z1_alvo = R1 + 1j * X1
    r = R1
    K = 0.12134
    constante = 7.934
    
    r_earth = 0.05916 # Ohm/km (Earth return resistance at 60 Hz)
    
    def montar_Zabc(GMR):
        chaves = ['a', 'b', 'c']
        Z = np.zeros((3, 3), dtype=complex)
        for i, fase_i in enumerate(chaves):
            xi, yi = posicoes[fase_i]
            for j, fase_j in enumerate(chaves):
                xj, yj = posicoes[fase_j]
                if i == j:
                    # Self-impedance
                    Z[i, j] = r + r_earth + 1j * K * (np.log(1 / GMR) + constante)
                else:
                    # Mutual-impedance
                    Dij = abs(xi - xj)
                    Z[i, j] = r_earth + 1j * K * (np.log(1 / Dij) + constante)
        return Z
    
    def Zabc_to_sequence(Zabc):
        a = np.exp(1j * 2 * np.pi / 3)
        A = np.array([[1, 1, 1],[1, a**2, a],[1, a, a**2]])
        A_inv = np.linalg.inv(A)
        return A_inv @ Zabc @ A

    def erro_GMR(GMR):
        if GMR <= 0: return 1e6
        Zabc = montar_Zabc(GMR)
        Zseq = Zabc_to_sequence(Zabc)
        Z1_calc = Zseq[1, 1]
        return abs(Z1_calc - Z1_alvo)

    res = minimize_scalar(erro_GMR, bounds=(0.0001, 0.1), method='bounded')
    GMR_otimo = res.x
    Zabc_final = montar_Zabc(GMR_otimo)
    return Zabc_final, GMR_otimo


def format_matrix_for_dss(matrix):
    """Formats a 1x1, 2x2, or 3x3 numpy matrix into the OpenDSS lower-triangle string."""
    n = matrix.shape[0]
    if n == 1:
        # Format for 1-phase: [R11]
        return f"[{matrix[0, 0]:.6f}]"
    elif n == 2:
        # Format for 2-phase: [R11 | R21 R22]
        return f"[{matrix[0, 0]:.6f} | {matrix[1, 0]:.6f} {matrix[1, 1]:.6f}]"
    elif n == 3:
        # Format for 3-phase: [R11 | R21 R22 | R31 R32 R33]
        return (f"[{matrix[0, 0]:.6f} | "
                f"{matrix[1, 0]:.6f} {matrix[1, 1]:.6f} | "
                f"{matrix[2, 0]:.6f} {matrix[2, 1]:.6f} {matrix[2, 2]:.6f}]")
    else:
        return "[]"

def sequence_to_phase_matrix(r1, x1, z0_to_z1_ratio=2.8):
    """
    Converts sequence impedances (Z0, Z1) to a symmetrical 3x3 phase
    impedance matrix (Zabc) for a transposed line.
    """
    z1 = r1 + 1j * x1
    z0 = z1 * z0_to_z1_ratio
    zs = (z0 + 2 * z1) / 3.0
    zm = (z0 - z1) / 3.0
    Zabc = np.array([
        [zs, zm, zm],
        [zm, zs, zm],
        [zm, zm, zs]
    ])
    return Zabc
import numpy as np
from pyscf import scf, gto
from scipy import optimize


def main0():
    for atom in [
        "O 0 0 0; H -1 0 0; H 1 0 0",
        "O 0 1 0; H -1 0 0; H 1 0 0",
        "O 0 0.6 0; H -0.6 0 0; H 0.6 0 0",
    ]:
        mol_h2o = gto.M(atom=atom, basis="ccpvdz")
        rhf_h2o = scf.RHF(mol_h2o)
        e_h2o = rhf_h2o.kernel()

        print(f"{mol_h2o.atom =}")
        print(f"energy = {e_h2o:.3f}")
        print()


def _cost_fn(params: np.ndarray) -> float:
    """
    We're optimizing the H-H distance (a) and the (HH)-O distance (b).

    ::

               O
               ▲
               │0.578
               │
      H◄───────┴─────1.5──►H

    """
    h_h_dist, h_o_dist = params

    # O.x is at X = 0
    # O.y is (b)
    # O.z is at Z = 0
    o_geometry = f"O 0 {h_o_dist} 0"
    # H.x is at +/- (a/2)
    # H.y is at Y = 0
    # H.z is at Z = 0
    h_x = h_h_dist / 2
    h1_geometry = f"H {-h_x} 0 0"
    h2_geometry = f"H {h_x} 0 0"

    atom = f"{o_geometry}; {h1_geometry}; {h2_geometry}"

    mol_h2o = gto.M(atom=atom, basis="ccpvdz")
    rhf_h2o = scf.RHF(mol_h2o)
    e_h2o = rhf_h2o.kernel()

    return e_h2o


def main1():
    print(_cost_fn(h_h_dist=1.8, h_o_dist=0.8))
    print(_cost_fn(h_h_dist=1.2, h_o_dist=0.2))


def main2():
    opt_result = optimize.minimize(
        fun=_cost_fn,
        x0=np.array([2.0, 0.0]),
    )
    print(opt_result)


if __name__ == "__main__":
    main2()

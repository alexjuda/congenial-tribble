from pyscf import scf, gto


def main():
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


if __name__ == "__main__":
    main()

from pyscf import scf, gto


def main():
    mol_h2o = gto.M(atom="O 0 0 0; H 0 1 0; H 0 0 1", basis="ccpvdz")
    rhf_h2o = scf.RHF(mol_h2o)
    e_h2o = rhf_h2o.kernel()

    print(f"{mol_h2o.atom =}")
    print(f"energy = {e_h2o}")


if __name__ == "__main__":
    main()

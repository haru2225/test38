#!/usr/bin/env python3
"""Export saved valid test38 generation frames as a periodic extended XYZ movie."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ase.io
from ase import Atoms
import numpy as np


def export(folder, output=None):
    folder = Path(folder)
    metadata = json.loads((folder / "generation.json").read_text())
    trajectory = np.load(folder / "positions.npy", mmap_mode="r", allow_pickle=False)
    valid = metadata["valid_frames"]
    meta = metadata["dataset_metadata"]
    types = np.asarray(meta["type_ids"], dtype=int)
    if not isinstance(valid, int) or not 1 <= valid <= len(trajectory):
        raise ValueError("Invalid valid_frames in generation.json")
    if trajectory.shape[1:] != (len(types), 3):
        raise ValueError("Trajectory shape does not match site metadata")
    cell = np.asarray(metadata["cell_angstrom"], dtype=float)
    output = Path(output) if output else folder / "generation.xyz"
    if output.resolve() in {(folder / "positions.npy").resolve(), (folder / "generation.json").resolve()}:
        raise ValueError("Output must not overwrite generation data")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    settings = metadata["settings"]
    initial_state = settings.get("initial_state", "reference")
    with temporary.open("w") as stream:
        for step in range(valid):
            pos = np.asarray(trajectory[step])
            if not np.isfinite(pos).all():
                raise ValueError(f"Non-finite coordinates at generation step {step}")
            atoms = Atoms(
                numbers=[meta["species"][i]["atomic_number"] for i in types],
                masses=[meta["species"][i]["mass_amu"] for i in types],
                positions=pos, cell=cell, pbc=True,
            )
            atoms.set_array("cg_type", types.copy())
            atoms.set_array("site_id", np.arange(len(types), dtype=int))
            atoms.set_array("cg_species", np.asarray([meta["species"][i]["name"] for i in types]))
            atoms.info.update(generation_step=step, initial_state=initial_state,
                              is_equilibrium_trajectory=False)
            atoms.wrap()
            ase.io.write(stream, atoms, format="extxyz")
    final_path = folder / "final.extxyz"
    if metadata.get("complete") and final_path.is_file():
        movie_final = ase.io.read(temporary, index=-1, format="extxyz")
        saved_final = ase.io.read(final_path)
        if (not np.array_equal(movie_final.numbers, saved_final.numbers)
                or not np.allclose(movie_final.cell.array, saved_final.cell.array, atol=1e-7, rtol=0)
                or not np.allclose(movie_final.positions, saved_final.positions, atol=1e-5, rtol=0)):
            temporary.unlink()
            raise ValueError(
                "Final XYZ frame differs from final.extxyz in the same generated directory; "
                "check that both files belong to the same generation run"
            )
    temporary.replace(output)
    print(f"XYZ movie: {output} ({valid} frames; initial_state={initial_state})", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export(args.generated, args.output)


if __name__ == "__main__":
    main()

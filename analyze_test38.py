#!/usr/bin/env python3
"""Check an unfinished test38 generation and an optional CGMD trajectory.

This is a structural smoke test, not an equilibrium or force-field validation.
It reports finite values, periodic minimum distances, frame-to-frame motion and
an RDF comparison against the prepared reference dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ase.io
import numpy as np
import torch


def samples(n: int, limit: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(n, limit), dtype=int))


def periodic_distances(pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Return all unique pair distances using the minimum-image convention."""
    frac = np.asarray(pos) @ np.linalg.inv(np.asarray(cell))
    delta = frac[:, None, :] - frac[None, :, :]
    delta -= np.rint(delta)
    cart = np.einsum("...i,ij->...j", delta, cell)
    dist = np.sqrt(np.einsum("...i,...i->...", cart, cart))
    return dist[np.triu_indices(len(pos), k=1)]


def trajectory_report(positions, cells, indices, r_max, bins, reference=None):
    positions = np.asarray(positions)
    cells = np.asarray(cells)
    finite = bool(np.isfinite(positions).all() and np.isfinite(cells).all())
    mins, close, rms_steps, hist = [], [], [], []
    previous = None
    for i in indices:
        pos, cell = positions[i], cells[i]
        distances = periodic_distances(pos, cell)
        mins.append(float(distances.min()))
        close.append(int(np.count_nonzero(distances < 0.5)))
        hist.append(np.histogram(distances, bins=bins, range=(0.0, r_max))[0])
        if previous is not None:
            delta = pos - previous
            delta -= np.rint(delta @ np.linalg.inv(cell)) @ cell
            rms_steps.append(float(np.sqrt(np.mean(delta * delta))))
        previous = pos
    result = {
        "frames": int(len(positions)),
        "sites": int(positions.shape[1]),
        "sampled_frames": int(len(indices)),
        "finite": finite,
        "max_abs_coordinate": float(np.nanmax(np.abs(positions))),
        "minimum_pair_distance_A": float(np.min(mins)),
        "frames_with_pair_below_0.5A": int(np.count_nonzero(np.asarray(close))),
        "mean_sampled_frame_rms_step_A": float(np.mean(rms_steps)) if rms_steps else 0.0,
        "max_sampled_frame_rms_step_A": float(np.max(rms_steps)) if rms_steps else 0.0,
        "rdf_histogram": np.sum(hist, axis=0).astype(int).tolist(),
    }
    if reference is not None:
        ref_pos, ref_cell = reference
        delta = positions[0] - ref_pos[0]
        delta -= np.rint(delta @ np.linalg.inv(ref_cell[0])) @ ref_cell[0]
        result["first_frame_rms_to_reference_A"] = float(np.sqrt(np.mean(delta * delta)))
    return result


def read_reference(folder):
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text())
    positions = np.load(folder / "positions.npy", mmap_mode="r", allow_pickle=False)
    cells = np.load(folder / "cells.npy", mmap_mode="r", allow_pickle=False)
    if positions.shape[0] != meta["frames"] or cells.shape[0] != meta["frames"]:
        raise ValueError("Reference shapes disagree with metadata")
    return positions, cells, meta


def read_generated(folder, checkpoint):
    folder = Path(folder)
    positions = np.load(folder / "positions.npy", mmap_mode="r", allow_pickle=False)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cell = np.asarray(ck["cell_angstrom"], dtype=float)
    cells = np.broadcast_to(cell, (len(positions), 3, 3))
    generation = json.loads((folder / "generation.json").read_text()) if (folder / "generation.json").exists() else {}
    valid = generation.get("valid_frames")
    if not isinstance(valid, int) or not 1 <= valid <= len(positions):
        raise ValueError("generation.json must specify valid_frames within the stored trajectory")
    positions = positions[:valid]
    cells = cells[:valid]
    return positions, cells, generation


def read_extxyz(path, limit):
    frames = ase.io.read(path, index=f":{limit}")
    if not isinstance(frames, list):
        frames = [frames]
    positions = np.asarray([a.get_positions() for a in frames], dtype=float)
    cells = np.asarray([a.cell.array for a in frames], dtype=float)
    if not np.isfinite(cells).all() or np.any(np.abs(np.linalg.det(cells)) < 1e-8):
        raise ValueError("CGMD extxyz must contain nonzero periodic cells")
    return positions, cells


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--generated", type=Path, required=True, help="Directory containing positions.npy")
    p.add_argument("--cgmd", type=Path, help="Optional CGMD md.extxyz to inspect")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--r-max", type=float, default=10.0)
    p.add_argument("--bins", type=int, default=100)
    args = p.parse_args()
    if args.max_frames < 1 or args.r_max <= 0 or args.bins < 1:
        p.error("max-frames, r-max and bins must be positive")

    ref_pos, ref_cells, meta = read_reference(args.reference)
    ref_idx = samples(len(ref_pos), args.max_frames)
    ref_report = trajectory_report(ref_pos, ref_cells, ref_idx, args.r_max, args.bins)
    gen_pos, gen_cells, generation = read_generated(args.generated, args.checkpoint)
    gen_idx = samples(len(gen_pos), args.max_frames)
    gen_report = trajectory_report(
        gen_pos, gen_cells, gen_idx, args.r_max, args.bins, (ref_pos, ref_cells)
    )
    # Compare normalized RDF histograms so the score is independent of frame count.
    ref_rdf = np.asarray(ref_report["rdf_histogram"], dtype=float)
    gen_rdf = np.asarray(gen_report["rdf_histogram"], dtype=float)
    ref_rdf /= max(ref_rdf.sum(), 1.0)
    gen_rdf /= max(gen_rdf.sum(), 1.0)
    gen_report["rdf_l1_to_reference"] = float(np.abs(ref_rdf - gen_rdf).sum())

    result = {
        "reference": ref_report,
        "generated": gen_report,
        "generation_complete": generation.get("complete"),
        "generation_valid_frames": generation.get("valid_frames"),
        "settings": {"r_max_A": args.r_max, "bins": args.bins, "max_frames": args.max_frames},
        "scientific_caveat": "Structural smoke diagnostics only; this does not establish equilibrium or physical CGMD validity.",
    }
    if args.cgmd:
        cg_pos, cg_cells = read_extxyz(args.cgmd, args.max_frames)
        cg_report = trajectory_report(cg_pos, cg_cells, samples(len(cg_pos), args.max_frames), args.r_max, args.bins)
        cg_rdf = np.asarray(cg_report["rdf_histogram"], dtype=float)
        cg_rdf /= max(cg_rdf.sum(), 1.0)
        cg_report["rdf_l1_to_reference"] = float(np.abs(ref_rdf - cg_rdf).sum())
        result["cgmd"] = cg_report

    warnings = []
    for name in ("generated", "cgmd"):
        report = result.get(name)
        if report is None:
            continue
        if not report["finite"]:
            warnings.append(f"{name}: non-finite value")
        if report["minimum_pair_distance_A"] < 0.5:
            warnings.append(f"{name}: pair distance below 0.5 A")
        if report["max_sampled_frame_rms_step_A"] > 10.0:
            warnings.append(f"{name}: frame-to-frame RMS displacement above 10 A")
    if generation.get("complete") is False:
        warnings.append("generated: trajectory is an incomplete checkpointed run")
    result["warnings"] = warnings
    result["status"] = "review" if warnings else "pass"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "warnings": warnings, "output": str(args.output)}))


if __name__ == "__main__":
    main()

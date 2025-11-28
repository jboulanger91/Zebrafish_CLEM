#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers for mapping clem_zfish1 neuron meshes & synapses with ANTs and skeletonizing them.

This module provides a minimal subset of the original ANTsRegistrationHelpers class,
stripped down to only what is needed by:

    register_and_skeletonize.py

Main responsibilities
---------------------
- Wrap ANTs' `antsApplyTransformsToPoints` to map 3D point clouds (e.g. mesh vertices,
  synapse coordinates) through a precomputed deformation field.
- Map soma/axon/dendrite meshes into a reference brain.
- Map synapse CSV files into the reference space.
- Skeletonize merged axon+dendrite meshes using `skeletor` and `navis`.
- Embed pre- and postsynaptic sites into the resulting SWC skeletons.

Requirements
------------
Environment variables must be set before instantiation:

    os.environ["ANTs_use_threads"] = "<num_threads>"
    os.environ["ANTs_bin_path"] = "/path/to/ANTs/bin"

The `ANTs_bin_path` directory must contain `antsApplyTransformsToPoints`.
"""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import navis
import numpy as np
import pandas as pd
import skeletor as sk
import tomllib
import trimesh as tm
import csv


class ANTsRegistrationHelpers:
    """
    Minimal helper class to interface with ANTs for point mapping and to
    map + skeletonize clem_zfish1 neurons.

    Parameters
    ----------
    manual_opts_dict : dict, optional
        Dictionary to override defaults in `self.opts_dict`. Only keys present
        in `self.opts_dict` are respected.
    """

    def __init__(self, manual_opts_dict: dict | None = None) -> None:
        # Core options used by this minimal version
        self.opts_dict = {
            "interpolation_method": "linear",
            "ANTs_verbose": 1,
            "tempdir": None,
            "num_cores": os.environ.get("ANTs_use_threads", "1"),
            "ANTs_bin_path": os.environ["ANTs_bin_path"],
        }

        # Allow manual overrides
        if manual_opts_dict is not None:
            for key in manual_opts_dict:
                self.opts_dict[key] = manual_opts_dict[key]

    # ------------------------------------------------------------------
    # Utility: path normalization for WSL on Windows
    # ------------------------------------------------------------------

    def convert_path_to_linux(self, path_name: str | Path) -> str:
        """
        Convert a Windows path to a WSL-style Linux path if needed.

        On Linux/macOS this is a no-op.

        Parameters
        ----------
        path_name : str or Path
            Path to convert.

        Returns
        -------
        str
            Linux-style path (on Windows) or original path (elsewhere).
        """
        if platform.system() == "Windows":
            path_name_linux = "/mnt/" + str(path_name)

            path_name_linux = path_name_linux.replace("\\", "/")
            path_name_linux = path_name_linux.replace(" ", "\\ ")

            for drive in ["C", "D", "E", "F", "G", "X", "Y", "Z", "W", "V"]:
                path_name_linux = path_name_linux.replace(f"{drive}:", drive.lower())

            return path_name_linux
        else:
            return str(path_name)

    # ------------------------------------------------------------------
    # Utility: call an ANTs command line tool
    # ------------------------------------------------------------------

    def call_ANTs_command(
        self,
        command_list: list[str],
        stdin_file: str | None = None,
        stdout_file: str | None = None,
    ) -> None:
        """
        Execute an ANTs command (antsApplyTransformsToPoints) on the host OS.

        On Linux/macOS: run directly.
        On Windows: write a small shell script and invoke it via bash/Ubuntu.

        Parameters
        ----------
        command_list : list of str
            Command and arguments as a list, e.g.
            ["<ANTs_bin_path>/antsApplyTransformsToPoints", ...]
        stdin_file : str, optional
            Path to a file that will be used as stdin.
        stdout_file : str, optional
            Path to a file where stdout will be redirected.
        """
        if platform.system() in ["Linux", "Darwin"]:
            subprocess.run(
                command_list,
                stdin=open(stdin_file) if stdin_file is not None else None,
                stdout=open(stdout_file, "w") if stdout_file is not None else None,
                check=False,
            )

        elif platform.system() == "Windows":
            # On Windows, route the command via WSL / Ubuntu
            registration_commands = (
                f"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={self.opts_dict['num_cores']}\n"
                "export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS\n\n"
            )
            registration_commands += " ".join(command_list)

            if stdin_file is not None:
                registration_commands += f" < {stdin_file}"
            if stdout_file is not None:
                registration_commands += f" > {stdout_file}"

            registration_commands_path_temp = tempfile.NamedTemporaryFile(
                dir=self.opts_dict["tempdir"], suffix=".sh", delete=False
            )
            registration_commands_path_temp.close()

            registration_commands_path_linux = self.convert_path_to_linux(
                registration_commands_path_temp.name
            )

            with open(registration_commands_path_temp.name, "wb") as f:
                f.write((registration_commands + "\n").encode())

            print("Executing ANTs command inside Windows shell...")
            print(registration_commands)

            result = subprocess.run(["bash", "-c", registration_commands_path_linux])
            if result.returncode != 0:
                subprocess.run(["ubuntu", "run", registration_commands_path_linux])

            os.remove(registration_commands_path_temp.name)

        # Small sleep to give the OS a chance to release file handles
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # Core: apply transforms to point clouds
    # ------------------------------------------------------------------

    def ANTs_applytransform_to_points(
        self,
        data_points: np.ndarray,
        transformation_prefix_path: str | Path,
        use_forward_transformation: bool = True,
        ANTs_dim: int = 3,
        input_limit_x: float | None = None,
        input_limit_y: float | None = None,
        input_limit_z: float | None = None,
        input_shift_x: float = 0,
        input_scale_x: float = 1,
        input_shift_y: float = 0,
        input_scale_y: float = 1,
        input_shift_z: float = 0,
        input_scale_z: float = 1,
        input_swap_xy: bool = False,
        output_shift_x: float = 0,
        output_scale_x: float = 1,
        output_shift_y: float = 0,
        output_scale_y: float = 1,
        output_shift_z: float = 0,
        output_scale_z: float = 1,
        output_swap_xy: bool = False,
    ) -> np.ndarray:
        """
        Map 3D points through an ANTs deformation field using antsApplyTransformsToPoints.

        Parameters
        ----------
        data_points : (N, 3) array
            Input coordinates (x, y, z).
        transformation_prefix_path : str or Path
            Prefix of the deformation field, e.g. ".../ANTs_dfield".
            The method will internally use:
                - "<prefix>_inverse.nii.gz" if use_forward_transformation=True
                - "<prefix>.nii.gz" otherwise
        use_forward_transformation : bool, default True
            For historical reasons, "forward" for points corresponds to using
            the inverse warp (ANTS convention).
        ANTs_dim : int, default 3
            Dimensionality for antsApplyTransformsToPoints.
        input_limit_x, input_limit_y, input_limit_z : float, optional
            Clip points to these maxima to avoid ANTs getting confused by points
            far outside the volume.
        input_shift_* / input_scale_* : float
            Pre-scaling and shifting applied before passing points to ANTs
            (e.g. convert from nm to µm or voxel coordinates).
        input_swap_xy : bool
            Swap x and y before registration.
        output_shift_* / output_scale_* : float
            Post-scaling and shifting applied after ANTs.
        output_swap_xy : bool
            Swap x and y after ANTs.

        Returns
        -------
        transformed_points : (N, 3) array
            Mapped coordinates.
        """
        # Temp CSV files for antsApplyTransformsToPoints
        all_data_points_path = tempfile.NamedTemporaryFile(
            dir=self.opts_dict["tempdir"], suffix=".csv", delete=False
        )
        all_data_points_path.close()

        all_data_points_registered_path = tempfile.NamedTemporaryFile(
            dir=self.opts_dict["tempdir"], suffix=".csv", delete=False
        )
        all_data_points_registered_path.close()

        all_data_points_path_linux = self.convert_path_to_linux(
            all_data_points_path.name
        )
        all_data_points_registered_linux = self.convert_path_to_linux(
            all_data_points_registered_path.name
        )
        transformation_prefix_path_linux = self.convert_path_to_linux(
            transformation_prefix_path
        )

        # ANTs expects 4 columns: x, y, z, t
        data_points_ants = np.c_[data_points, np.zeros(data_points.shape[0])]

        # Clip points if requested
        if input_limit_x is not None:
            data_points_ants[data_points_ants[:, 0] > input_limit_x, 0] = input_limit_x
        if input_limit_y is not None:
            data_points_ants[data_points_ants[:, 1] > input_limit_y, 1] = input_limit_y
        if input_limit_z is not None:
            data_points_ants[data_points_ants[:, 2] > input_limit_z, 2] = input_limit_z

        # Input rescaling / shifting (e.g. nm -> µm)
        data_points_ants[:, 0] = input_shift_x + input_scale_x * data_points_ants[:, 0]
        data_points_ants[:, 1] = input_shift_y + input_scale_y * data_points_ants[:, 1]
        data_points_ants[:, 2] = input_shift_z + input_scale_z * data_points_ants[:, 2]

        # Optional x/y swap
        if input_swap_xy:
            data_points_ants = data_points_ants[:, [1, 0, 2]]

        # Write CSV for ANTs
        np.savetxt(
            all_data_points_path.name,
            data_points_ants,
            delimiter=",",
            header="x,y,z,t",
            comments="",
        )

        # Build antsApplyTransformsToPoints command
        cmd = [
            f"{self.opts_dict['ANTs_bin_path']}/antsApplyTransformsToPoints",
            "--precision",
            "0",  # float32
            "--dimensionality",
            f"{ANTs_dim}",
            "--input",
            f"{all_data_points_path_linux}",
            "--output",
            f"{all_data_points_registered_linux}",
        ]

        # For points, "forward" convention uses the inverse .nii.gz
        if use_forward_transformation:
            cmd += ["--transform", f"{transformation_prefix_path_linux}_inverse.nii.gz"]
        else:
            cmd += ["--transform", f"{transformation_prefix_path_linux}.nii.gz"]

        cmd += ["-o", all_data_points_registered_linux]

        self.call_ANTs_command(cmd)

        # Load transformed points (ensure 2D even if only one row)
        transformed_points = np.loadtxt(
            all_data_points_registered_path.name,
            delimiter=",",
            skiprows=1,
            usecols=(0, 1, 2),
            ndmin=2,
        )

        # Optional output swap
        if output_swap_xy:
            transformed_points = transformed_points[:, [1, 0, 2]]

        # Post-rescaling / shifting
        transformed_points[:, 0] = (
            output_shift_x + output_scale_x * transformed_points[:, 0]
        )
        transformed_points[:, 1] = (
            output_shift_y + output_scale_y * transformed_points[:, 1]
        )
        transformed_points[:, 2] = (
            output_shift_z + output_scale_z * transformed_points[:, 2]
        )

        # Cleanup temp files
        os.remove(all_data_points_path.name)
        os.remove(all_data_points_registered_path.name)

        return transformed_points

    # ------------------------------------------------------------------
    # High-level: map a cell and generate SWC
    # ------------------------------------------------------------------

    def map_and_skeletonize_cell(
        self,
        root_path: str | Path,
        cell_name: str,
        transformation_prefix_path: str | Path,
        use_forward_transformation: bool = True,
        input_limit_x: float | None = None,
        input_limit_y: float | None = None,
        input_limit_z: float | None = None,
        input_shift_x: float = 0,
        input_scale_x: float = 1,
        input_shift_y: float = 0,
        input_scale_y: float = 1,
        input_shift_z: float = 0,
        input_scale_z: float = 1,
        input_swap_xy: bool = False,
        output_shift_x: float = 0,
        output_scale_x: float = 1,
        output_shift_y: float = 0,
        output_scale_y: float = 1,
        output_shift_z: float = 0,
        output_scale_z: float = 1,
        output_swap_xy: bool = False,
    ) -> None:
        """
        Map one clem_zfish1 neuron/axon folder into reference brain space and
        produce SWC skeletons with synapse labels.

        Expected folder structure (per `cell_name`):
            <root_path>/<cell_name>/
                <cell_name>_metadata.txt
                <cell_name>_soma.obj          (optional for axon-only)
                <cell_name>_dendrite.obj      (optional for axon-only)
                <cell_name>_axon.obj
                <cell_name>_presynapses.csv   (optional, with header)
                <cell_name>_postsynapses.csv  (optional, with header)

        The method will:
            - map synapse CSVs into reference space,
            - map soma/axon/dendrite meshes,
            - create merged OBJ files (original + mapped),
            - skeletonize axon+dendrite,
            - label soma/axon/dendrite/presynapse/postsynapse nodes,
            - write SWC(s) into:
                <cell_name>.swc
                mapped/<cell_name>_mapped.swc

        SWC label convention:
            0 = undefined
            1 = soma
            2 = axon
            3 = dendrite
            4 = presynapse
            5 = postsynapse
        """
        import csv  # ensure available here
        root_path = Path(root_path)

        # ------------------------------------------------------------------
        # Small helpers for synapse I/O
        # ------------------------------------------------------------------
        def _detect_delimiter(path: Path, default: str = "\t") -> str:
            with open(path, "r", newline="") as f:
                sample = f.read(1024)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters="\t,; ")
                    return dialect.delimiter
                except csv.Error:
                    return default

        def _load_original_synapses(path: Path, synapse_type: str) -> pd.DataFrame:
            """
            Load original synapse CSV with header and nm-resolution columns.

            Returns a DataFrame with columns:
                partner_id, x_nm, y_nm, z_nm, radius_um

            - partner_id: postsynaptic_ID or presynaptic_ID, depending on file
            - x_nm, y_nm, z_nm: nm coordinates (from x_nm_res / y_nm_res / z_nm_res)
            - radius_um: currently a constant (0.5 µm) for all synapses
            """
            if not path.exists():
                return pd.DataFrame(
                    columns=["partner_id", "x_nm", "y_nm", "z_nm", "radius_um"]
                )

            delim = _detect_delimiter(path)
            df_raw = pd.read_csv(path, sep=delim)

            # Partner ID column: choose by name, fall back to first
            if "postsynaptic_ID" in df_raw.columns:
                partner_col = "postsynaptic_ID"
            elif "presynaptic_ID" in df_raw.columns:
                partner_col = "presynaptic_ID"
            else:
                partner_col = df_raw.columns[0]

            # Prefer nm-resolution columns if present
            if {"x_nm_res", "y_nm_res", "z_nm_res"}.issubset(df_raw.columns):
                x_nm = pd.to_numeric(df_raw["x_nm_res"], errors="coerce")
                y_nm = pd.to_numeric(df_raw["y_nm_res"], errors="coerce")
                z_nm = pd.to_numeric(df_raw["z_nm_res"], errors="coerce")
            else:
                # Fallback: use columns 1–3 (e.g. x (8 nm), y (8 nm), z (30 nm))
                if df_raw.shape[1] < 4:
                    raise ValueError(
                        f"{path} has fewer than 4 columns; cannot identify positions."
                    )
                x_nm = pd.to_numeric(df_raw.iloc[:, 1], errors="coerce")
                y_nm = pd.to_numeric(df_raw.iloc[:, 2], errors="coerce")
                z_nm = pd.to_numeric(df_raw.iloc[:, 3], errors="coerce")

            # For now use a fixed radius in µm; you can refine this later if needed
            radius_um = pd.Series(0.5, index=df_raw.index, dtype=float)

            df_out = pd.DataFrame(
                {
                    "partner_id": df_raw[partner_col].astype(str),
                    "x_nm": x_nm,
                    "y_nm": y_nm,
                    "z_nm": z_nm,
                    "radius_um": radius_um,
                }
            )
            # Drop rows with NaN coords
            df_out = df_out.dropna(subset=["x_nm", "y_nm", "z_nm"])
            return df_out

        def _load_mapped_synapses_simple(path: Path) -> pd.DataFrame:
            """
            Load mapped synapse CSV written by this function.

            Format (no header, space-separated):
                partner_id  x  y  z  radius

            All coordinates are in µm.
            """
            if not path.exists():
                return pd.DataFrame(
                    columns=["partner_id", "x", "y", "z", "radius"]
                )
            df = pd.read_csv(
                path,
                sep=r"\s+",
                header=None,
                names=["partner_id", "x", "y", "z", "radius"],
                engine="python",
            )
            return df

        # ------------------------------------------------------------------
        # Load metadata
        # ------------------------------------------------------------------
        with open(root_path / cell_name / f"{cell_name}_metadata.txt", mode="rb") as fp:
            metadata = tomllib.load(fp)

        print(f"Mapping and skeletonization of cell {cell_name}.")
        print("Meta data:", metadata)

        # Folder for mapped outputs (mapped meshes + mapped SWC + mapped CSVs)
        (root_path / cell_name / "mapped").mkdir(exist_ok=True)

        # ------------------------------------------------------------------
        # Step 1: Map synapse CSVs and create OBJ spheres
        # ------------------------------------------------------------------
        for synapse_type_str in ["presynapses", "postsynapses"]:
            src_csv = root_path / cell_name / f"{cell_name}_{synapse_type_str}.csv"
            if src_csv.exists():
                print(f"Mapping {synapse_type_str} for {cell_name}")

                # Load original (nm) synapse coordinates
                df_orig = _load_original_synapses(src_csv, synapse_type_str)

                if len(df_orig) > 0:
                    # Use nm coordinates as input to ANTs, with input_scale_* converting to µm
                    points_nm = df_orig[["x_nm", "y_nm", "z_nm"]].to_numpy(
                        dtype=float
                    ).reshape(-1, 3)

                    points_transformed = self.ANTs_applytransform_to_points(
                        points_nm,
                        transformation_prefix_path,
                        use_forward_transformation=use_forward_transformation,
                        ANTs_dim=3,
                        input_limit_x=input_limit_x,
                        input_limit_y=input_limit_y,
                        input_limit_z=input_limit_z,
                        input_shift_x=input_shift_x,
                        input_scale_x=input_scale_x,
                        input_shift_y=input_shift_y,
                        input_scale_y=input_scale_y,
                        input_shift_z=input_shift_z,
                        input_scale_z=input_scale_z,
                        input_swap_xy=input_swap_xy,
                        output_shift_x=output_shift_x,
                        output_scale_x=output_scale_x,
                        output_shift_y=output_shift_y,
                        output_scale_y=output_scale_y,
                        output_shift_z=output_shift_z,
                        output_scale_z=output_scale_z,
                        output_swap_xy=output_swap_xy,
                    )

                    # points_transformed are in reference space (µm)
                    df_mapped = pd.DataFrame(
                        {
                            "partner_id": df_orig["partner_id"],
                            "x": points_transformed[:, 0],
                            "y": points_transformed[:, 1],
                            "z": points_transformed[:, 2],
                            "radius": df_orig["radius_um"],
                        }
                    )
                else:
                    df_mapped = pd.DataFrame(
                        columns=["partner_id", "x", "y", "z", "radius"]
                    )

                # Write mapped synapses as a simple 5-column file (no original 8/30 nm coords)
                mapped_csv = (
                    root_path
                    / cell_name
                    / "mapped"
                    / f"{cell_name}_{synapse_type_str}_mapped.csv"
                )
                df_mapped.to_csv(
                    mapped_csv,
                    index=False,
                    sep=" ",
                    header=False,
                    float_format="%.8f",
                )

            # Draw synapses as small spheres in OBJ (original + mapped)
            for mapped_i in [0, 1]:
                suffix_str = "" if mapped_i == 0 else "_mapped"
                folder_str = "" if mapped_i == 0 else "mapped"

                if mapped_i == 0:
                    syn_csv = root_path / cell_name / f"{cell_name}_{synapse_type_str}.csv"
                    df_sphere = _load_original_synapses(syn_csv, synapse_type_str)
                    # For original OBJ, use µm coordinates by converting nm → µm
                    df_sphere = df_sphere.assign(
                        x=df_sphere["x_nm"] * input_scale_x,
                        y=df_sphere["y_nm"] * input_scale_y,
                        z=df_sphere["z_nm"] * input_scale_z,
                        radius=df_sphere["radius_um"],
                    )
                else:
                    syn_csv = (
                        root_path
                        / cell_name
                        / "mapped"
                        / f"{cell_name}_{synapse_type_str}_mapped.csv"
                    )
                    df_sphere = _load_mapped_synapses_simple(syn_csv)

                if len(df_sphere) == 0:
                    continue

                spheres = []
                for _, row in df_sphere.iterrows():
                    sphere = tm.creation.icosphere(
                        radius=row["radius"], subdivisions=2
                    )
                    sphere.apply_translation((row["x"], row["y"], row["z"]))
                    spheres.append(sphere)

                if spheres:
                    scene = tm.Scene(spheres)
                    out_obj = (
                        root_path
                        / cell_name
                        / folder_str
                        / f"{cell_name}_{synapse_type_str}{suffix_str}.obj"
                    )
                    scene.export(out_obj)

        # ------------------------------------------------------------------
        # Step 2: Map soma / dendrite / axon meshes  (unchanged)
        # ------------------------------------------------------------------
        meshes_original: dict[str, tm.Trimesh] = {}
        meshes_mapped: dict[str, tm.Trimesh] = {}

        for part_name in ["soma", "dendrite", "axon"]:
            print(f"Mapping mesh {part_name} for {cell_name}")

            part_path = root_path / cell_name / f"{cell_name}_{part_name}.obj"
            if not part_path.is_file():
                continue

            mesh = tm.load(part_path)
            meshes_original[part_name] = mesh.copy()

            # Original coordinates are assumed to be in nm → convert to µm for skeletonization
            meshes_original[part_name].vertices *= 0.001

            # Map vertices using ANTs
            mesh.vertices = self.ANTs_applytransform_to_points(
                mesh.vertices,
                transformation_prefix_path,
                use_forward_transformation=use_forward_transformation,
                ANTs_dim=3,
                input_limit_x=input_limit_x,
                input_limit_y=input_limit_y,
                input_limit_z=input_limit_z,
                input_shift_x=input_shift_x,
                input_scale_x=input_scale_x,
                input_shift_y=input_shift_y,
                input_scale_y=input_scale_y,
                input_shift_z=input_shift_z,
                input_scale_z=input_scale_z,
                input_swap_xy=input_swap_xy,
                output_shift_x=output_shift_x,
                output_scale_x=output_scale_x,
                output_shift_y=output_shift_y,
                output_scale_y=output_scale_y,
                output_shift_z=output_shift_z,
                output_scale_z=output_scale_z,
                output_swap_xy=output_swap_xy,
            )

            # Fix potential mesh issues and store mapped version
            meshes_mapped[part_name] = sk.pre.fix_mesh(
                mesh, fix_normals=True, inplace=False
            )

            # Save slightly simplified mapped mesh
            out_mapped = (
                root_path
                / cell_name
                / "mapped"
                / f"{cell_name}_{part_name}_mapped.obj"
            )
            sk.pre.simplify(meshes_mapped[part_name], 0.75).export(out_mapped)

        # ------------------------------------------------------------------
        # Step 3: Skeletonize (cell with soma+dendrite+axon OR axon-only)
        # ------------------------------------------------------------------
        for mapped_i in [0, 1]:
            neurite_radius = 0.5
            soma_radius = 2
            skeletonize_inv_dist = 1.5

            if mapped_i == 0:
                suffix_str = ""
                folder_str = ""
                meshes = meshes_original
            else:
                suffix_str = "_mapped"
                folder_str = "mapped"
                meshes = meshes_mapped

            # ----------------- (from here down, logic mostly unchanged) -----------------
            # Case 1: full neuron (soma + dendrite + axon)
            if "soma" in meshes and "dendrite" in meshes and "axon" in meshes:
                print(f"Making SWC (full cell) for {cell_name}{suffix_str}")

                # Save merged whole-neuron OBJ (original and mapped)
                if mapped_i == 0:
                    mesh_axon = tm.load(
                        root_path / cell_name / f"{cell_name}_axon.obj"
                    )
                    mesh_dendrite = tm.load(
                        root_path / cell_name / f"{cell_name}_dendrite.obj"
                    )
                    mesh_soma = tm.load(
                        root_path / cell_name / f"{cell_name}_soma.obj"
                    )
                    mesh_complete = tm.util.concatenate(
                        [mesh_axon, mesh_dendrite, mesh_soma]
                    )
                    sk.pre.simplify(mesh_complete, 0.75).export(
                        root_path / cell_name / f"{cell_name}.obj"
                    )
                else:
                    mesh_axon = tm.load(
                        root_path
                        / cell_name
                        / "mapped"
                        / f"{cell_name}_axon_mapped.obj"
                    )
                    mesh_dendrite = tm.load(
                        root_path
                        / cell_name
                        / "mapped"
                        / f"{cell_name}_dendrite_mapped.obj"
                    )
                    mesh_soma = tm.load(
                        root_path
                        / cell_name
                        / "mapped"
                        / f"{cell_name}_soma_mapped.obj"
                    )
                    mesh_complete = tm.util.concatenate(
                        [mesh_axon, mesh_dendrite, mesh_soma]
                    )
                    sk.pre.simplify(mesh_complete, 0.75).export(
                        root_path
                        / cell_name
                        / "mapped"
                        / f"{cell_name}_mapped.obj"
                    )

                # Concatenate axon+dendrite for skeletonization
                mesh_axon_dendrite = tm.util.concatenate(
                    [meshes["axon"], meshes["dendrite"]]
                )

                # Soma position: mean of soma triangle centers
                soma_x, soma_y, soma_z = np.mean(
                    meshes["soma"].triangles_center, axis=0
                )

                # Skeletonize axon+dendrite
                skel = sk.skeletonize.by_teasar(
                    mesh_axon_dendrite, inv_dist=skeletonize_inv_dist
                )
                skel = sk.post.clean_up(skel).reindex()

                df_swc = skel.swc
                df_swc["radius"] = neurite_radius
                df_swc["label"] = 0

                # Heal gaps using navis
                x = navis.read_swc(df_swc)
                x = navis.heal_skeleton(
                    x,
                    method="ALL",
                    max_dist=None,
                    min_size=None,
                    drop_disc=False,
                    mask=None,
                    inplace=False,
                )

                # Temporarily write to SWC to reimport as numeric DF
                f_swc_temp = tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".swc"
                )
                f_swc_temp.close()
                x.to_swc(f_swc_temp.name)

                df_swc = pd.read_csv(
                    f_swc_temp.name,
                    sep=" ",
                    names=["node_id", "label", "x", "y", "z", "radius", "parent_id"],
                    comment="#",
                    header=None,
                )
                os.remove(f_swc_temp.name)

                # Insert soma as node_id=1
                df_swc.loc[:, "node_id"] += 1
                df_swc.loc[df_swc["parent_id"] > -1, "parent_id"] += 1

                i_min = (
                    (df_swc["x"] - soma_x) ** 2
                    + (df_swc["y"] - soma_y) ** 2
                    + (df_swc["z"] - soma_z) ** 2
                ).argmin()

                if df_swc.loc[i_min, "parent_id"] == -1:
                    df_swc.loc[i_min, "parent_id"] = 0
                    soma_row = pd.DataFrame(
                        {
                            "node_id": 1,
                            "label": 1,
                            "x": soma_x,
                            "y": soma_y,
                            "z": soma_z,
                            "radius": soma_radius,
                            "parent_id": -1,
                        },
                        index=[0],
                    )
                else:
                    node_id = df_swc.loc[i_min, "node_id"]
                    soma_row = pd.DataFrame(
                        {
                            "node_id": 1,
                            "label": 1,
                            "x": soma_x,
                            "y": soma_y,
                            "z": soma_z,
                            "radius": soma_radius,
                            "parent_id": node_id,
                        },
                        index=[0],
                    )

                df_swc = pd.concat([soma_row, df_swc], ignore_index=True)

                # Label axon vs dendrite based on proximity to mapped meshes
                for i, row in df_swc.iterrows():
                    if row["node_id"] == 1:
                        continue  # soma already labeled

                    d_min_axon = np.sqrt(
                        (meshes["axon"].vertices[:, 0] - row["x"]) ** 2
                        + (meshes["axon"].vertices[:, 1] - row["y"]) ** 2
                        + (meshes["axon"].vertices[:, 2] - row["z"]) ** 2
                    ).min()

                    d_min_dendrite = np.sqrt(
                        (meshes["dendrite"].vertices[:, 0] - row["x"]) ** 2
                        + (meshes["dendrite"].vertices[:, 1] - row["y"]) ** 2
                        + (meshes["dendrite"].vertices[:, 2] - row["z"]) ** 2
                    ).min()

                    if d_min_axon < d_min_dendrite:
                        df_swc.loc[i, "label"] = 2  # axon
                    else:
                        df_swc.loc[i, "label"] = 3  # dendrite

                pre_synapses: list[list[int]] = []
                post_synapses: list[list[int]] = []

                # Attach presynapses
                if mapped_i == 0:
                    pres_df = _load_original_synapses(
                        root_path
                        / cell_name
                        / f"{cell_name}_presynapses.csv",
                        "presynapses",
                    )
                    pres_df = pres_df.assign(
                        x=pres_df["x_nm"] * input_scale_x,
                        y=pres_df["y_nm"] * input_scale_y,
                        z=pres_df["z_nm"] * input_scale_z,
                    )
                else:
                    pres_df = _load_mapped_synapses_simple(
                        root_path
                        / cell_name
                        / "mapped"
                        / f"{cell_name}_presynapses_mapped.csv"
                    )

                if len(pres_df) > 0:
                    print("Presynapses found for SWC labelling")
                    for _, row in pres_df.iterrows():
                        dist = np.sqrt(
                            (df_swc["x"] - row["x"]) ** 2
                            + (df_swc["y"] - row["y"]) ** 2
                            + (df_swc["z"] - row["z"]) ** 2
                        )
                        i_min = dist.argmin()

                        if dist[i_min] < 10 * neurite_radius:
                            df_swc.loc[i_min, "label"] = 4  # presynapse
                            pre_synapses.append(
                                [
                                    int(row["partner_id"]),
                                    int(df_swc.loc[i_min, "node_id"]),
                                ]
                            )
                        else:
                            print(
                                "Postsynaptic cell not connected to SWC. "
                                "Presynapse too far away:",
                                int(row["partner_id"]),
                                dist[i_min],
                            )
                            pre_synapses.append([int(row["partner_id"]), -1])

                # Attach postsynapses
                if mapped_i == 0:
                    post_df = _load_original_synapses(
                        root_path
                        / cell_name
                        / f"{cell_name}_postsynapses.csv",
                        "postsynapses",
                    )
                    post_df = post_df.assign(
                        x=post_df["x_nm"] * input_scale_x,
                        y=post_df["y_nm"] * input_scale_y,
                        z=post_df["z_nm"] * input_scale_z,
                    )
                else:
                    post_df = _load_mapped_synapses_simple(
                        root_path
                        / cell_name
                        / "mapped"
                        / f"{cell_name}_postsynapses_mapped.csv"
                    )

                if len(post_df) > 0:
                    print("Postsynapses found for SWC labelling")
                    for _, row in post_df.iterrows():
                        dist = np.sqrt(
                            (df_swc["x"] - row["x"]) ** 2
                            + (df_swc["y"] - row["y"]) ** 2
                            + (df_swc["z"] - row["z"]) ** 2
                        )
                        i_min = dist.argmin()

                        if dist[i_min] < 10 * neurite_radius:
                            df_swc.loc[i_min, "label"] = 5  # postsynapse
                            post_synapses.append(
                                [
                                    int(row["partner_id"]),
                                    int(df_swc.loc[i_min, "node_id"]),
                                ]
                            )
                        else:
                            print(
                                "Presynaptic cell not connected to SWC. "
                                "Postsynapse too far away:",
                                int(row["partner_id"]),
                                dist[i_min],
                            )
                            post_synapses.append([int(row["partner_id"]), -1])

            # Case 2: axon-only mapping
            elif "axon" in meshes and "soma" not in meshes and "dendrite" not in meshes:
                print(f"Making SWC (axon-only) for {cell_name}{suffix_str}")

                skel = sk.skeletonize.by_teasar(
                    meshes["axon"], inv_dist=skeletonize_inv_dist
                )
                skel = sk.post.clean_up(skel).reindex()

                df_swc = skel.swc
                df_swc["radius"] = neurite_radius
                df_swc["label"] = 0

                x = navis.read_swc(df_swc)
                x = navis.heal_skeleton(
                    x,
                    method="ALL",
                    max_dist=None,
                    min_size=None,
                    drop_disc=False,
                    mask=None,
                    inplace=False,
                )

                f_swc_temp = tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".swc"
                )
                f_swc_temp.close()
                x.to_swc(f_swc_temp.name)

                df_swc = pd.read_csv(
                    f_swc_temp.name,
                    sep=" ",
                    names=["node_id", "label", "x", "y", "z", "radius", "parent_id"],
                    comment="#",
                    header=None,
                )
                os.remove(f_swc_temp.name)

                # Label all nodes as axon
                df_swc["label"] = 2

                pre_synapses = []
                post_synapses = []

                # Presynapses
                if mapped_i == 0:
                    pres_df = _load_original_synapses(
                        root_path
                        / cell_name
                        / f"{cell_name}_presynapses.csv",
                        "presynapses",
                    )
                    pres_df = pres_df.assign(
                        x=pres_df["x_nm"] * input_scale_x,
                        y=pres_df["y_nm"] * input_scale_y,
                        z=pres_df["z_nm"] * input_scale_z,
                    )
                else:
                    pres_df = _load_mapped_synapses_simple(
                        root_path
                        / cell_name
                        / "mapped"
                        / f"{cell_name}_presynapses_mapped.csv"
                    )

                if len(pres_df) > 0:
                    print("Presynapses found for SWC labelling (axon-only)")
                    for _, row in pres_df.iterrows():
                        dist = np.sqrt(
                            (df_swc["x"] - row["x"]) ** 2
                            + (df_swc["y"] - row["y"]) ** 2
                            + (df_swc["z"] - row["z"]) ** 2
                        )
                        i_min = dist.argmin()
                        if dist[i_min] < 10 * neurite_radius:
                            df_swc.loc[i_min, "label"] = 4
                            pre_synapses.append(
                                [
                                    int(row["partner_id"]),
                                    int(df_swc.loc[i_min, "node_id"]),
                                ]
                            )
                        else:
                            print(
                                "Postsynaptic cell not connected to SWC. "
                                "Presynapse too far away:",
                                int(row["partner_id"]),
                                dist[i_min],
                            )
                            pre_synapses.append([int(row["partner_id"]), -1])

                # Postsynapses
                if mapped_i == 0:
                    post_df = _load_original_synapses(
                        root_path
                        / cell_name
                        / f"{cell_name}_postsynapses.csv",
                        "postsynapses",
                    )
                    post_df = post_df.assign(
                        x=post_df["x_nm"] * input_scale_x,
                        y=post_df["y_nm"] * input_scale_y,
                        z=post_df["z_nm"] * input_scale_z,
                    )
                else:
                    post_df = _load_mapped_synapses_simple(
                        root_path
                        / cell_name
                        / "mapped"
                        / f"{cell_name}_postsynapses_mapped.csv"
                    )

                if len(post_df) > 0:
                    print("Postsynapses found for SWC labelling (axon-only)")
                    for _, row in post_df.iterrows():
                        dist = np.sqrt(
                            (df_swc["x"] - row["x"]) ** 2
                            + (df_swc["y"] - row["y"]) ** 2
                            + (df_swc["z"] - row["z"]) ** 2
                        )
                        i_min = dist.argmin()
                        if dist[i_min] < 10 * neurite_radius:
                            df_swc.loc[i_min, "label"] = 5
                            post_synapses.append(
                                [
                                    int(row["partner_id"]),
                                    int(df_swc.loc[i_min, "node_id"]),
                                ]
                            )
                        else:
                            print(
                                "Presynaptic cell not connected to SWC. "
                                "Postsynapse too far away:",
                                int(row["partner_id"]),
                                dist[i_min],
                            )
                            post_synapses.append([int(row["partner_id"]), -1])

            else:
                # No valid combination of parts – nothing to skeletonize
                print(
                    f"Skipping SWC generation for {cell_name}{suffix_str}: "
                    f"no suitable combination of soma/dendrite/axon meshes."
                )
                continue

            # ------------------------------------------------------------------
            # Final SWC writeout (both full-cell and axon-only branches)
            # ------------------------------------------------------------------
            df_swc = df_swc.reindex(
                columns=["node_id", "label", "x", "y", "z", "radius", "parent_id"]
            )
            metadata["presynapses"] = pre_synapses
            metadata["postsynapses"] = post_synapses

            header = (
                "# SWC format file based on specifications at "
                "http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html\n"
                "# Generated by 'map_and_skeletonize_cell' of the ANTs registration helper library.\n"
                f"# Metadata: {str(metadata)}\n"
                "# Labels: 0 = undefined; 1 = soma; 2 = axon; 3 = dendrite; "
                "4 = Presynapse; 5 = Postsynapse\n"
            )

            out_swc = root_path / cell_name / folder_str / f"{cell_name}{suffix_str}.swc"
            with open(out_swc, "w") as fp:
                fp.write(header)
                df_swc.to_csv(fp, index=False, sep=" ", header=None)

    def convert_synapse_file(
        self,
        root_path: Path,
        cell_name: str,
        shift_x: float,
        shift_y: float,
        shift_z: float,
        scale_x: float,
        scale_y: float,
        scale_z: float,
    ) -> None:
        """
        Add nm-resolution coordinates to existing synapse CSVs by appending
        three new columns at the right:

            x_nm_res = shift_x + x * scale_x
            y_nm_res = shift_y + y * scale_y
            z_nm_res = shift_z + z * scale_z

        Assumes synapse CSVs already exist:
            <cell_name>_presynapses.csv
            <cell_name>_postsynapses.csv

        and that the 2nd–4th columns (index 1, 2, 3) are the raw x, y, z positions
        in index units (e.g. 8 nm, 8 nm, 30 nm steps).

        Parameters
        ----------
        root_path : Path
            Root folder containing per-cell subfolders.
        cell_name : str
            Name of the cell/axon folder.
        shift_x, shift_y, shift_z : float
            Offsets applied to the raw coordinates.
        scale_x, scale_y, scale_z : float
            Multipliers applied to the raw coordinates (e.g. 8, 8, 30 for nm).
        """
        root_path = Path(root_path)

        for synapse_type in ["presynapses", "postsynapses"]:
            csv_path = root_path / cell_name / f"{cell_name}_{synapse_type}.csv"

            if not csv_path.exists():
                print(f"{cell_name}: no {synapse_type} file found at {csv_path}")
                continue

            # Detect delimiter (tab/comma/space/etc.)
            with open(csv_path, "r", newline="") as f:
                sample = f.read(1024)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters="\t,; ")
                    delim = dialect.delimiter
                except csv.Error:
                    # Fallback: assume tab-separated
                    delim = "\t"

            # Read table with the detected delimiter
            df = pd.read_csv(csv_path, sep=delim)

            # If nm-res columns already exist → skip
            if {"x_nm_res", "y_nm_res", "z_nm_res"}.issubset(df.columns):
                print(f"{csv_path} already contains nm_res columns; skipping.")
                continue

            if df.shape[1] < 4:
                raise ValueError(
                    f"{csv_path} has fewer than 4 columns; cannot identify x/y/z "
                    f"as columns 1–3. Found columns: {list(df.columns)}"
                )

            # Use columns 1, 2, 3 as x, y, z (regardless of their names)
            x_raw = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            y_raw = pd.to_numeric(df.iloc[:, 2], errors="coerce")
            z_raw = pd.to_numeric(df.iloc[:, 3], errors="coerce")

            df["x_nm_res"] = shift_x + x_raw * scale_x
            df["y_nm_res"] = shift_y + y_raw * scale_y
            df["z_nm_res"] = shift_z + z_raw * scale_z

            # Write back using the same delimiter
            df.to_csv(csv_path, sep=delim, index=False)

            print(f"Updated {csv_path} with x_nm_res, y_nm_res, z_nm_res.")
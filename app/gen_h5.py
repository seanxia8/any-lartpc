#!/usr/bin/env python3
"""Main driver for training, validation, inference and analysis."""

import sys, os, argparse
import yaml
current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)

import torch
import numpy as np
import h5py
from algorithms import LArTPC_general, dot_steel_PMT, PMT_angle_general
from utils import place_hexa_opticsensors, place_photon_origin
def main(config):

    cfg_file = config
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(current_directory, 'config', config)
    if not os.path.isfile(cfg_file):
        raise FileNotFoundError(f"Configuration not found: {config}")

    # Load the configuration file
    with open(cfg_file, 'r', encoding='utf-8') as cfg_yaml:
        cfg = yaml.safe_load(cfg_yaml)

    assert 'detector' in cfg, "Must provide a `detector` block in the configuration."
    assert 'photon' in cfg, "Must provide a `photon` block in the configuration."
    assert 'io' in cfg, "Must provide an `io` block in the configuration."
    assert 'writer' in cfg['io'], "Must provide a `writer` block in the `io` block."

    # Load the detector configuration
    detector_cfg = cfg['detector']
    lartpc = LArTPC_general(detector_cfg)

    if 'PMT' in detector:
        pmt_model = dot_steel_PMT(detector['PMT'])
    else:
        pmt_model = dot_steel_PMT()

    pmt_coords = place_hexa_opticsensors(lartpc.lx, lartpc.ly, lartpc.lz, lartpc.spacing_y, lartpc.spacing_z)
    # Load the photon configuration
    photon_cfg = cfg['photon']
    n_photon = photon_cfg.get('n_photon', 10000)
    photon_reso = photon_cfg.get('resolution', 0.2)

    out_dir = cfg['io']['writer']['output_dir']
    out_prefix = cfg['io']['writer']['output_prefix']
    out_filename = f"{out_dir}/{out_prefix}_{lartpc.lx}x{lartpc.ly}x{lartpc.lz}_{n_photon}_{photon_reso}.h5"

    photon_origins = place_photon_origin(lartpc.lx, lartpc.ly, lartpc.lz, photon_reso)
    n_photon = torch.full((photon_origins.shape[1],), n_photon, dtype=torch.int16)

    visi_factor, angle_values, distance_values = lartpc.geometric_factor(photon_origins)
    time_of_flight_values = lartpc.tof
    pmt_model.compute_pmt_eff(angle_values, n_photon)
    efficiency_values = pmt_model.pmt_eff

    with h5py.File(out_filename, 'w') as f:
        geometry_group = f.create_group("geometry")
        data_group = f.create_group("data")

        pmt_group = geometry_group.create_group("pmt")
        photon_group = geometry_group.create_group("photon")

        pmt_group.create_dataset("positions", data=pmt_coords.numpy())
        photon_group.create_dataset("origins", data=photon_origins.numpy())
        data_group.create_dataset("n_photon", data=n_photon.numpy())
        data_group.create_dataset("efficiency", data=efficiency_values.numpy())
        data_group.create_dataset("angle", data=angle_values.numpy())
        data_group.create_dataset("distance", data=distance_values.numpy())
        data_group.create_dataset("time_of_flight", data=time_of_flight_values.numpy())

    print("HDF5 file written successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generates h5 files")

    parser.add_argument('--config', '-c',
                        help='Path to the configuration file',
                        type=str, required=True)

    args = parser.parse_arges()

    main(args.config)



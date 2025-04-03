import torch
import h5py
from algorithms import LArTPC_general, dot_steel_PMT, PMT_angle_general
from utils import place_photon_origin, get_unique_filename
import time

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
def gen_h5_photon(cfg:dict):

    assert 'detector' in cfg, "Must provide a `detector` block in the configuration."
    assert 'photon' in cfg, "Must provide a `photon` block in the configuration."
    assert 'io' in cfg, "Must provide an `io` block in the configuration."
    assert 'writer' in cfg['io'], "Must provide a `writer` block in the `io` block."

    # Load the detector configuration
    detector_cfg = cfg['detector']
    lartpc = LArTPC_general(detector_cfg)

    if 'PMT' in detector_cfg:
        pmt_model = dot_steel_PMT(detector_cfg['PMT'])
    else:
        pmt_model = dot_steel_PMT()

    # Load the photon configuration
    photon_cfg = cfg['photon']
    is_sampled = photon_cfg.get('sample', True)
    n_photon = photon_cfg.get('n_photon', 10000)
    photon_reso = photon_cfg.get('resolution', 0.2)

    out_dir = cfg['io']['writer']['output_dir']
    out_prefix = cfg['io']['writer']['output_prefix']

    if is_sampled:
        temp_filename = f"{out_dir}/{out_prefix}_{lartpc.lx}x{lartpc.ly}x{lartpc.lz}_{n_photon}_{lartpc.cathode_gap}.h5"
        scaling_factors = torch.tensor([lartpc.lx, lartpc.ly, lartpc.lz], dtype=torch.float32)
        offsets = scaling_factors / 2

        photon_origins = torch.rand((n_photon, 3), dtype=torch.float32) * scaling_factors - offsets
        n_photon = torch.full((photon_origins.shape[0],), 1000, dtype=torch.int16).unsqueeze(1)
    else:
        temp_filename = f"{out_dir}/{out_prefix}_{lartpc.lx}x{lartpc.ly}x{lartpc.lz}_{n_photon}_res_{photon_reso}_{lartpc.cathode_gap}.h5"
        photon_origins = place_photon_origin(lartpc.lx, lartpc.ly, lartpc.lz, photon_reso)
        n_photon = torch.full((photon_origins.shape[0],), n_photon, dtype=torch.int16).unsqueeze(1)

    t_photon = torch.rand(n_photon.shape, dtype=torch.float32)
    out_filename = get_unique_filename(temp_filename)

    tstart = time.time()
    visi_factor, angle_values, distance_values = lartpc.geometric_factor(photon_origins)
    time_of_flight_values = lartpc.tof
    print(f"Took {time.time() - tstart} to generate the PMT visibility and photon angles.")

    tstart = time.time()
    pmt_model.compute_pmt_eff(angle_values, n_photon)
    print(f"Took {time.time() - tstart} to generate the PMT efficiencies.")

    efficiency_values = pmt_model.pmt_eff

    with h5py.File(out_filename, 'w') as f:
        geometry_group = f.create_group("geometry")
        data_group = f.create_group("data")

        pmt_group = geometry_group.create_group("pmt")
        photon_group = geometry_group.create_group("photon")

        pmt_group.create_dataset("positions", data=lartpc.pmt_coords.detach().cpu().numpy())
        photon_group.create_dataset("origins", data=photon_origins.detach().cpu().numpy())
        photon_group.create_dataset("times", data = t_photon.detach().cpu().numpy())
        #photon_group.create_dataset("n_photon", data=n_photon.numpy())
        data_group.create_dataset("visibility", data=visi_factor.detach().cpu().numpy())
        data_group.create_dataset("pmt_efficiency", data=efficiency_values.detach().cpu().numpy())
        data_group.create_dataset("angle", data=angle_values.detach().cpu().numpy())
        data_group.create_dataset("distance", data=distance_values.detach().cpu().numpy())
        data_group.create_dataset("time_of_flight", data=time_of_flight_values.detach().cpu().numpy())

    print(f"HDF5 file {out_filename} written successfully.")

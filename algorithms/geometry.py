import torch
from utils import place_hexa_opticsensors

class LArTPC_general():
    def __init__(self, cfg: dict):
        self.lx = cfg.get('length', 4320)  # horizontal length as viewed from beam dir in mm
        self.ly = cfg.get('width', 4320)  # vertical length as viewed from beam dir in mm
        self.lz = cfg.get('height', 4320)  # depth as viewed from beam dir in mm

        self.gap_x = cfg.get('gap_x', 10)
        self.spacing_y = cfg.get('spacing_y', 500)
        self.spacing_z = cfg.get('spacing_z', 500)

        self.cathode_gap = cfg.get('cathode_gap', 10)

        self.pmt_coords = place_hexa_opticsensors(self.lx+2*self.gap_x, self.ly, self.lz, self.spacing_y, self.spacing_z)
        self.pmt_radius = cfg['PMT'].get('pmt_radius', 50)  # PMT radius in mm

        #self.n_opticsensor = self.pmt_coords.shape[0]
        #self.att = cfg.get("attenuation", 1095) # attenuation length in mm
        #self.k = 1 / self.att

        self.refra_index = cfg.get('refractive_index', 1.5)
        self.tof = None

        self.speed_of_light = 299.792 / self.refra_index # mm/ns

    def geometric_factor(self, coords, flip_coin):
        assert self.pmt_coords is not None, 'PMT coordinates not defined.'
        assert coords.shape[-1] == self.pmt_coords.shape[-1], ValueError("Position coordinates not correct.")
        self.pmt_coords = self.pmt_coords.to(coords.device)
        id = 1 - int(flip_coin)

        r = torch.cdist(coords, self.pmt_coords[id], p=2)
        r_sq = r ** 2
        displace = coords[:, None, 0] - self.pmt_coords[id][None,:,0]

        # Optimize the final return operation, in-place operations if possible
        sin_angle = torch.abs(displace) / r  # Sin angle calculation
        sin_sq_angle = sin_angle ** 2
        angle_rad = torch.asin(sin_angle)
        pmt_xsec = self.pmt_radius**2 * (1 - sin_sq_angle)
        visi_factor = pmt_xsec / r_sq / 4

        self.tof = r / self.speed_of_light
        return visi_factor, angle_rad

    @property
    def get_tof(self):
        return self.tof
    @property
    def get_pmt_coords(self):
        return self.pmt_coords

    @property
    def get_lengths(self):
        return torch.tensor([self.lx, self.ly, self.lz])
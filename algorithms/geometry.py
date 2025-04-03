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

        self.n_opticsensor = self.pmt_coords.shape[0]
        self.att = cfg.get("attenuation", 1095) # attenuation length in mm
        self.k = 1 / self.att

        self.refra_index = cfg.get('refractive_index', 1.5)
        self.tof = None

        self.speed_of_light = 299.792 / self.refra_index # mm/ns

    def geometric_factor(self, coords):
        assert self.pmt_coords is not None, 'PMT coordinates not defined.'
        assert coords.shape[1] == self.pmt_coords.shape[1], ValueError("Position coordinates not correct.")
        r = torch.cdist(coords, self.pmt_coords)
        mask_0 = r > 0
        displace = coords[:, None, 0] - self.pmt_coords[None,:,0]
        mask_cathode = torch.abs(displace) <= (0.5*self.lx + self.gap_x - 0.5*self.cathode_gap)
        sin_angle = displace / r

        visi_factor = torch.exp(-self.k * r) / r ** 2
        self.tof = r*mask_0*mask_cathode / self.speed_of_light

        return visi_factor*mask_0*mask_cathode, torch.rad2deg(torch.arcsin(sin_angle)), r
    @property
    def get_tof(self):
        return self.tof
    @property
    def get_pmt_coords(self):
        return self.pmt_coords

    @property
    def get_lengths(self):
        return torch.tensor([self.lx, self.ly, self.lz])
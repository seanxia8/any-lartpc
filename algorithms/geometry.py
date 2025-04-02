import torch
from utils import place_hexa_opticsensors

class LArTPC_general():
    def __init__(self, cfg: dict):
        self.lx = cfg.get('length', 4.32)  # horizontal length as viewed from beam dir
        self.ly = cfg.get('width', 4.32)  # vertical length as viewed from beam dir
        self.lz = cfg.get('height', 4.32)  # depth as viewed from beam dir

        self.spacing_y = cfg.get('spacing_y', 0.5)
        self.spacing_z = cfg.get('spacing_z', 0.5)

        self.pmt_coords = place_hexa_opticsensors(self.lx, self.ly, self.lz, self.spacing_y, self.spacing_z)

        self.n_opticsensor = self.pmt_coords.shape[0]
        self.att = cfg.get("attenuation", 1.095) # attenuation length in m
        self.k = 1 / self.att

        self.refra_index = cfg.get('refractive_index', 1.5)
        self.pmt_coords = None
        self.tof = None

        self.speed_of_light = 0.299792 / self.refra_index # m/ns

    def geometric_factor(self, coords):
        assert self.pmt_coords is not None, 'PMT coordinates not defined.'
        assert coords.shape[1] == self.pmt_coords.shape[1], ValueError("Position coordinates not correct.")
        r = torch.cdist(coords, self.pmt_coords)
        mask = r > 0
        displace = coords[:, None, 0] - self.pmt_coords[None,:,0]
        sin_angle = displace / r

        visi_factor = torch.exp(-self.k * r) / r ** 2
        self.tof = r / self.speed_of_light

        return visi_factor, torch.rad2deg(torch.arcsin(sin_angle)), r
    @property
    def get_tof(self):
        return self.tof
    @property
    def get_pmt_coords(self):
        return self.pmt_coords
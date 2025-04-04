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
        id = int(flip_coin)

        r = torch.cdist(coords, self.pmt_coords[id])
        r_sq = r ** 2
        displace = coords[:, None, 0] - self.pmt_coords[id][None,:,0]
        #mask_cathode = torch.abs(displace) <= (0.5*self.lx + self.gap_x - 0.5*self.cathode_gap)

        # Optimize the final return operation, in-place operations if possible
        sin_angle = displace / r  # Sin angle calculation
        angle_rad = torch.arcsin(sin_angle)  # Use pre-calculated sin_angle
        pmt_solid_angle = self.pmt_radius**2 * sin_angle**2 / r_sq
        # r in mm, multiply 1.E+6 to bring visibility to m^-2
        visi_factor = 1.E+6 * pmt_solid_angle / r_sq / 4
        self.tof = r / self.speed_of_light

        '''
        if flip_coin:
            visibility = torch.stack((visi_factor, torch.zeros_like(visi_factor)), dim=-1)
            angle_rad = torch.stack((ang_rad, torch.zeros_like(angle_rad)), dim=-1)
            self.tof = torch.stack((self.tof, torch.zeros_like(self.tof)), dim=-1)
        else:
            visibility = torch.stack((torch.zeros_like(visi_factor), visi_factor), dim=-1)
            angle_rad = torch.stack((torch.zeros_like(angle_rad), angle_rad), dim=-1)
            self.tof = torch.stack((torch.zeros_like(self.tof), self.tof), dim=-1)
        '''

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
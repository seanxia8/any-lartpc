import numpy as np
import torch
from torch.utils.data import Dataset
from optics.datatypes import cached_gamma_shotgun, gamma_shotgun, wcsim_db, simplesim_db
from optics.utils.profiler import Profiler

class MCset(Dataset):
    """
    Optic photon MC in the form of torch Dataset
    """
    def __init__(self, cfg: dict):
        """
        Creates a torch Dataset object for training opticSiren with MC
        that has a isotropic light or photon shotgun sources and PMT responses

        A h5 file is specified in the cfg.
        Configuration parameters:
        --------------------------

        Parameters:
        ------------
        cfg: dict
            model and dataset configurations
        """

        self._cfg = cfg.get('data')
        self.mc = None
        mctype = self._cfg.get('MC_type', 0)
        if mctype == 0:
            self.mc = cached_gamma_shotgun(cfg)
        elif mctype == 1:
            self.mc = wcsim_db(cfg)
        elif mctype == 2:
            self.mc = simplesim_db(cfg)
        elif mctype == 3:
            self.mc = wcsim_voxel(cfg)
        return

    def __len__(self):
        return len(self.mc)

    def __getitem__(self, idx:int) -> dict:
        """
        Used by torch Dataset to get a single track.

        Parameters
        ----------
        idx : int
            index of the event

        Returns
        -------
        dict
            a dictionary containing the event's n_gamma, dir_gamma, pos_gamma, pmtQ, and pmtT
        """

        dir_Gamma = None
        weights = None
        evID = self.mc._id[idx]
        if self.mc._dir is not None:
            dir_Gamma = self.mc._dir[evID].reshape(-1, 2) # (nevents, <phi, theta>)
            pos_Gamma = self.mc._pos[evID].reshape(-1, 3) # (nevents, <x, y, z>)
            pmt_Q = self.mc._pmtQ[evID][:] # (1, npmt)
            pmt_T = self.mc._pmtT[evID][:] # (1, npmt)
        if self.mc._weight is not None:
            weights = self.mc._weight[evID].reshape(-1, self.mc.npmt) # (nevents, n_pmt)
            
        Qmin = self.mc.Qmin
        Qmax = self.mc.Qmax
        output = {
            'evIdx': evID,
            'nGamma': self.mc._n_gamma,
            'dirGamma': dir_Gamma,
            'posGamma': pos_Gamma,
            'pmt_Q': pmt_Q,
            'pmt_T': pmt_T,
            'weights': weights,
            'Qmin': Qmin,
            'Qmax': Qmax
        }
        return output
    
    @staticmethod
    def collate_fn(batch: dict) -> dict:
        """
        Used by torch DataLoader to collate a batch of events into a single dictionary
        """
        app = lambda x: np.squeeze(x) if len(batch) < 1 else np.concatenate(x)  # no need to average
        output = {}

        output['evIdx'] = torch.as_tensor(
            [data['evIdx'] for data in batch], dtype=torch.int32
        )
        output['nGamma'] = torch.as_tensor(
            [data['nGamma'] for data in batch], dtype=torch.int32
        )
        if isinstance(batch[0]['dirGamma'], np.ndarray):
            output['dirGamma'] = torch.as_tensor(app([data['dirGamma'] for data in batch]),
                                                 dtype=torch.float32
            )
        output['posGamma'] = torch.as_tensor(app([data['posGamma'] for data in batch]),
                                             dtype=torch.float32
                                             )
        output['pmtQ'] = torch.as_tensor(app([data['pmt_Q'] for data in batch]),
                                         dtype=torch.float32
                                         )
        output['pmtT'] = torch.as_tensor(app([data['pmt_T'] for data in batch]),
                                         dtype=torch.float32
                                         )
        output['weights'] = torch.as_tensor(app([data['weights'] for data in batch]),
                                            dtype=torch.float32
                                            )
        output['Qmax'] = torch.as_tensor(batch[0]['Qmax'], dtype=torch.float32)
        output['Qmin'] = torch.as_tensor(batch[0]['Qmin'], dtype=torch.float32)

        return output

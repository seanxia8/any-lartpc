import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
from slar.transform import partial_xform_vis

class MCset(Dataset):
    """
    Optic photon MC in the form of torch Dataset
    """
    def __init__(self, cfg: dict):

        assert 'io' in cfg, 'The configuration file must contain the io section.'
        assert 'loader' in cfg['io'], 'The configuration file must contain the loader section.'

        self._pmt_coords = []
        self.photon_origins = []
        self.photon_times = []
        self.visi_factor = []
        self.eff_values = []
        self.angle_values = []
        self.distance_values = []
        self.time_of_flight_values = []

        #self.load(fname)

    def load(self, fname: str):
        """
        Load the photon info and pmt efficiency in the specified h5 file
        The h5 file must include these datasets:
            '/geometry/pmt/positions' (3, n_pmts): pmt coordinates
            '/geometry/photon/origins' (3, n_photon_origins): position of the photons
            '/data/visibility' (n_photon_origins, n_pmts, 1): visibility of the photons
            '/data/pmt_efficiency' (n_photon_origins, n_pmts, 1): efficiency of the pmts
            '/data/angle' (n_photon_origins, n_pmts, 1): angle of the photons
            '/data/distance' (n_photon_origins, n_pmts, 1): distance of the photons
            '/data/time_of_flight' (n_photon_origins, n_pmts, 1): time of flight of the photons

        Parameters:
        -----------
            fname: str
            Filename(s) / wildcard to laod the dataset from
        """

        hdf5_files = glob.glob(fname)
        print('[MCset] loading', fname)

        for file_index, file_name in enumerate(hdf5_files):
            print(f"Loading {file_name}.")
            with h5py.File(file_name, mode='r') as f:
                if file_index == 0:
                    self._pmt_coords.append(f['/geometry/pmt/positions'][:])
                self.photon_origins.append(f['/geometry/photon/origins'][:])
                #self._n_photon.append(f['/geometry/photon/n_photon'][:])
                self.photon_times.append(f['/geometry/photon/times'][:])
                self.visi_factor.append(f['/data/visibility'][:])
                self.eff_values.append(f['/data/pmt_efficiency'][:])
                self.angle_values.append(f['/data/angle'][:])
                #self.distance_values.append(f['/data/distance'][:])
                self.time_of_flight_values.append(f['/data/time_of_flight'][:])

        self._pmt_coords = np.array(self._pmt_coords)
        n_pmt = self._pmt_coords[0].shape[0]
        #n_photon_origins = self.photon_origins[0].shape[0]

        self.photon_origins = np.array(self.photon_origins).reshape(-1, 3)
        #self._n_photon = np.array(self._n_photon).reshape(-1, n_photon_origins)
        self.photon_times = np.array(self.photon_times).reshape(-1, 1)
        self.visi_factor = np.array(self.visi_factor).reshape(-1, n_pmt)
        self.eff_values = np.array(self.eff_values).reshape(-1, n_pmt)
        self.angle_values = np.array(self.angle_values).reshape(-1, n_pmt)
        #self.distance_values = np.array(self.distance_values).reshape(-1, n_pmt)
        self.time_of_flight_values = np.array(self.time_of_flight_values).reshape(-1, n_pmt)
    @property
    def pmts(self):
        return self._pmt_coords
    def __len__(self):
        return len(self.photon_origins)

    def to(self, device):
        self.photon_origins = torch.tensor(self.photon_origins).to(device)
        self.visi_factor = torch.tensor(self.visi_factor).to(device)
        self.eff_values = torch.tensor(self.eff_values).to(device)
        self.angle_values = torch.tensor(self.angle_values).to(device)
        torch.cuda.synchronize()
        return self

    @property
    def device(self):
        return self.photon_origins.device

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
        output = {
            'evIdx': idx,
            'photon_origins': self.photon_origins[idx],
            'photon_times': self.photon_times[idx],
            'visibility': self.visi_factor[idx],
            'pmt_efficiency': self.eff_values[idx],
            'angle': self.angle_values[idx],
            #'distance': self.distance_values[idx],
            'time_of_flight': self.time_of_flight_values[idx]
        }
        return output
    '''
    @staticmethod
    def collate_fn(batch: dict) -> dict:
        """
        Used by torch DataLoader to collate a batch of events into a single dictionary
        """
        output = {}

        output['evIdx'] = torch.as_tensor(
            [data['evIdx'] for data in batch], dtype=torch.int32
        )
        output['n_photons'] = torch.as_tensor(
            [data['n_photons'] for data in batch], dtype=torch.int32
        )
        output['photon_origins'] = torch.as_tensor(
            [data['photon_origins'] for data in batch], dtype=torch.int32
        )
        output['visibility'] = torch.as_tensor(
            [data['visibility'] for data in batch], dtype=torch.float32
        )
        output['pmt_efficiency'] = torch.as_tensor(
            [data['pmt_efficiency'] for data in batch], dtype=torch.float32
        )
        output['angle'] = torch.as_tensor(
            [data['angle'] for data in batch], dtype=torch.float32
        )
        output['distance'] = torch.as_tensor(
            [data['distance'] for data in batch], dtype=torch.float32
        )
        output['time_of_flight'] = torch.as_tensor(
            [data['time_of_flight'] for data in batch], dtype=torch.float32
        )
        return output
    '''

class MCDataLoader:
    '''
    A fast implementation of dataloader transcribed from Photonlib.
    '''

    def __init__(self, mc: MCset, cfg, device=None):
        '''
        Constructor.

        Arguments
        ---------
        cfg: dict
            Config dictionary. See "Examples" below.

        device: torch.device (optional)
            Device for the returned data. Default: None.

        Examples
        --------
        This is an example configuration in yaml format.

        ```
		photonlib:
			filepath: plib_file.h5

		data:
			dataset:
				weight:
					method: vis
					factor: 1000000.0
					threshold: 1.0e-08
			loader:
				batch_size: 500
				shuffle: true

        transform_vis:
            eps: 1.0e-05
            sin_out: false
            vmax: 1.0
		```

        The `photonlib` section provide the input file of `PhotonLib`.

        [Optional] The `weight` subsection is the weighting scheme. Supported
        schemes are:

        1. `vis`, where `weight ~ 1/vis * factor`.  Weights below `threshold`
        are set to one.
        2. To-be-implemented.

        [Optional] The `loader` subsection mimics pytorch's `DataLoader` class,
        however, only `batch_size` and `shuffle` options are implemented.  If
        `loader` subsection is absent, the data loader returns the whole photon
        lib in a single entry.

        [Optional] The `transform_vis` subsection uses `log(vis+eps)` in the
        training. The final output is scaled to `[0,1]`.
        '''

        # load plib to device
        self._mc = mc
        fname = cfg['photonlib']['file_path']
        self._mc.load(fname)
        self._mc.to(device)

        # get weighting scheme
        weight_cfg = cfg.get('data', {}).get('dataset', {}).get('weight')
        if weight_cfg:
            method = weight_cfg.get('method')
            if method == 'vis':
                self.get_weight = self.get_weight_by_vis
                print('[MCDataLoader] weighting using', method)
                print('[MCDataLoader] params:', weight_cfg)
            else:
                raise NotImplementedError(f'Weight method {method} is invalid')
            self._weight_cfg = weight_cfg
        else:
            print('[MCDataLoader] weight = 1')
            self.get_weight = lambda vis: torch.tensor(1., device=device)

        # tranform visiblity in pseudo-log scale (default: False)
        xform_params = cfg.get('transform_vis')
        if xform_params:
            print('[MCDataLoader] using log scale transformaion')
            print('[MCDataLoader] transformation params', xform_params)

        self.xform_vis, self.inv_xform_vis = partial_xform_vis(xform_params)

        # prepare dataloader
        loader_cfg = cfg.get('data', {}).get('loader')
        self._batch_mode = loader_cfg is not None

        if self._batch_mode:
            # dataloader in batches
            self._batch_size = loader_cfg.get('batch_size', 1)
            self._shuffle = loader_cfg.get('shuffle', False)
        else:
            # returns the whole plib in a single batch
            #n_voxels = len(self._mc)
            #vox_ids = torch.arange(n_voxels, device=device)

            #meta = self._plib.meta
            #pos = meta.norm_coord(meta.voxel_to_coord(vox_ids))

            vis = self._mc.visi_factor * self._mc.eff_values
            w = self.get_weight(vis)
            target = self.xform_vis(vis)

            self._cache = dict(position=self._mc.photon_origins, angle=self._mc.angle_values, value=vis, weight=w, target=target)

    @property
    def device(self):
        return self._mc.device

    def get_weight_by_vis(self, vis):
        '''
        Weight by inverse visibility, `weight  = 1/vis * factor`.
        Weights below `threshold` are set to 1.

        Arguments
        ---------
        vis: torch.Tensor
            Visibility values.

        Returns
        -------
        w: trorch.Tensor
            Weight values with `w.shape == vis.shape`.
        '''
        factor = self._weight_cfg.get('factor', 1.)
        threshold = self._weight_cfg.get('threshold', 1e-8)
        w = vis * factor
        w[w < threshold] = 1.
        return w

    def __len__(self):
        '''
        Number of batches.
        '''
        from math import ceil
        if self._batch_mode:
            return ceil(len(self._mc) / self._batch_size)

        return 1

    def __iter__(self):
        '''
        Generator of batch data.

        For non-batch mode, the whole photon lib is returned in a single entry
        from the cache.
        '''
        if self._batch_mode:
            #meta = self._plib.meta
            n_voxels = len(self._mc)
            if self._shuffle:
                vox_list = torch.randperm(n_voxels, device=self.device)
            else:
                vox_list = torch.arange(n_voxels, device=self.device)

            for b in range(len(self)):
                sel = slice(b * self._batch_size, (b + 1) * self._batch_size)
                vox_ids = vox_list[sel]
                pos = self._mc.photon_origins[vox_ids]
                angles = self._mc.angle_values[vox_ids]
                vis = self._mc.visi_factor[vox_ids] * self._mc.eff_values[vox_ids]
                w = self.get_weight(vis)
                target = self.xform_vis(vis)
                output = dict(position=pos, angle=angles, value=vis, weight=w, target=target)
                yield output
        else:
            yield self._cache
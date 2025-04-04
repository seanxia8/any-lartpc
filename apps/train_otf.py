import os
from tqdm import tqdm
import time
import yaml
import torch
from algorithms import LArTPC_general, dot_steel_PMT
from slar.nets import SirenVis, WeightedL2Loss
from slar.optimizers import optimizer_factory
from slar.utils import CSVLogger, get_device
from slar.transform import partial_xform_vis
from slar.analysis import vis_bias
from photonlib import AABox

def get_weight_by_vis(vis, factor=1., threshold=1e-8):
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
    w = vis * factor
    w[w < threshold] = 1.
    return w
def train_on_the_fly(train_cfg : dict, detector_cfg : dict, photon_cfg : dict):
    '''
    A function to run an optimization loop for SirenVis model.
    '''

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if train_cfg.get('device'):
        DEVICE = get_device(train_cfg['device']['type'])

    epoch_ctr = 0

    lartpc = LArTPC_general(detector_cfg)
    pmt = dot_steel_PMT(detector_cfg['PMT'])

    lengths = lartpc.get_lengths.to(DEVICE)
    ranges = torch.column_stack([-lengths/2,lengths/2])
    aabox = AABox(ranges)
    # Create necessary pieces: the model, optimizer, loss, logger.
    # Load the states if this is resuming.
    net = SirenVis(train_cfg,aabox).to(DEVICE)

    opt, sch, epoch = optimizer_factory(net.parameters(),train_cfg)
    if epoch >0:
        epoch_ctr = int(epoch)
        print('[train] resuming training from epoch',epoch_ctr)
    criterion = WeightedL2Loss() 
    logger = CSVLogger(train_cfg)
    logdir = logger.logdir
    bias_threshold = train_cfg['logger']['analysis']['vis_bias'].get('threshold', 4.5e-5)

    # Set the control parameters for the training loop
    train_sub_cfg = train_cfg.get('train',dict())
    epoch_max = train_sub_cfg.get('max_epochs',int(1e20))
    save_every_epochs = train_sub_cfg.get('save_every_epochs',10)
    print(f'[train] train for max epochs {epoch_max}')

    # tranform visiblity in pseudo-log scale (default: False)
    xform_params = train_cfg.get('transform_vis')
    if xform_params:
        print('[Transformation] using log scale transformaion')
        print('[Transformation] transformation params', xform_params)

    xform_vis, inv_xform_vis = partial_xform_vis(xform_params)

    data_cfg = train_cfg.get('data', {})
    weight_factor = data_cfg['dataset']['weight'].get('factor', 1.)
    weight_threshold = data_cfg['dataset']['weight'].get('threshold', 1e-8)

    # Store configuration
    with open(os.path.join(logdir,'train_cfg.yaml'), 'w') as f:
        yaml.safe_dump(train_cfg, f)
    
    # Start the training loop
    t0=time.time()
    twait=time.time()
    stop_training = False

    n_photon = photon_cfg.get('n_photon', 10000)
    n_photon_pos = photon_cfg.get('n_photon_pos', 50000)

    scaling_factors = torch.tensor([lartpc.lx / 2 - lartpc.cathode_gap / 2, lartpc.ly, lartpc.lz], dtype=torch.float32)
    offsets = torch.tensor([lartpc.cathode_gap / 2, -lartpc.ly / 2, -lartpc.lz / 2])
    n_photon = torch.full((n_photon_pos,), n_photon, dtype=torch.int16).unsqueeze(1)

    for epoch_ctr in tqdm(range(epoch_max)):
        opt.zero_grad()
        #for batch_idx, data in enumerate(tqdm(dl,desc='Epoch %-3d' % epoch_ctr)):
        photon_origins = torch.rand((n_photon_pos, 3), dtype=torch.float32) * scaling_factors + offsets
        photon_origins = photon_origins.to(DEVICE)
        flip_coin = torch.rand(1).item() < 0.5
        if flip_coin:
            photon_origins[:,0] *= -1.0

        visi_factor, angle_values = lartpc.geometric_factor(photon_origins, flip_coin)
        #time_of_flight_values = lartpc.tof.to(DEVICE)
        pmt.compute_pmt_eff(angle_values, n_photon)
        #efficiency_values = pmt.pmt_eff.to(DEVICE)
        efficiency_values = pmt.pmt_eff

        photon_origins = photon_origins.to(DEVICE)

        #visi_factor = visi_factor.to(DEVICE)
        #angle_values = (angle_values/torch.pi).to(DEVICE)

        if flip_coin:
            photon_origins = torch.stack((photon_origins, torch.zeros_like(photon_origins)), dim=0)
            visi_factor = torch.stack((visi_factor, torch.zeros_like(visi_factor)), dim=0)
            angle_values = torch.stack((angle_values, torch.zeros_like(angle_values)), dim=0)
            #time_of_flight_values = torch.stack((time_of_flight_values, torch.zeros_like(time_of_flight_values)), dim=0)
            efficiency_values = torch.stack((efficiency_values, torch.zeros_like(efficiency_values)), dim=0)
        else:
            photon_origins = torch.stack((torch.zeros_like(photon_origins), photon_origins), dim=0)
            visi_factor = torch.stack((torch.zeros_like(visi_factor), visi_factor), dim=0)
            angle_values = torch.stack((torch.zeros_like(angle_values), angle_values), dim=0)
            #time_of_flight_values = torch.stack((torch.zeros_like(time_of_flight_values), time_of_flight_values), dim=0)
            efficiency_values = torch.stack((torch.zeros_like(efficiency_values), efficiency_values), dim=0)

        acceptance = visi_factor * efficiency_values
        # Input data prep
        x              = torch.cat((photon_origins/lengths,angle_values/torch.pi),dim=-1)
        weights        = get_weight_by_vis(acceptance, factor=weight_factor, threshold=weight_threshold)
        target         = xform_vis(acceptance)
        target_linear  = acceptance

        twait = time.time()-twait
        # Running the model, compute the loss, back-prop gradients to optimize.
        ttrain = time.time()
        pred   = net(x)
        loss   = criterion(pred, target, weights)
        loss.backward()
        opt.step()
        ttrain = time.time()-ttrain
        # Log training parameters
        logger.record(['iter', 'epoch', 'loss', 'ttrain', 'twait'],
                      [epoch_ctr, epoch_ctr, loss.item(), ttrain, twait])
        twait = time.time()
        # Step the logger
        pred_linear = inv_xform_vis(pred)
        #pred_bias = vis_bias(pred_linear, target_linear, bias_threshold)
        logger.step(epoch_ctr, target_linear, pred_linear)
        #logger.write()

        if stop_training:
            break

        if sch is not None:
            sch.step()

        if (save_every_epochs*epoch_ctr) > 0 and epoch_ctr % save_every_epochs == 0:
            filename = os.path.join(logdir,'epoch-%04d.ckpt' % (epoch_ctr))
            net.save_state(filename,opt,sch,epoch_ctr)

            
    print('[train] Stopped training at epochs',epoch_max)
    logger.write()
    logger.close()

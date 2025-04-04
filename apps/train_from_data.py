import os
from tqdm import tqdm
import numpy as np
import time
import yaml
import torch
from data_io import MCset, MCDataLoader
from algorithms import LArTPC_general
from slar.nets import SirenVis, WeightedL2Loss
from slar.optimizers import optimizer_factory
from slar.utils import CSVLogger, get_device
from photonlib import AABox

def train(cfg : dict, lartpc: LArTPC_general, mc: MCset):
    '''
    A function to run an optimization loop for SirenVis model.
    Configuration specific to this function is "train" at the top level.

    Parameters
    ----------
    max_epochs : int
        The maximum number of epochs before stopping training

    max_iterations : int
        The maximum number of iterations before stopping training

    save_every_epochs : int
        A period in epochs to store the network state

    save_every_iterations : int
        A period in iterations to store the network state

    optimizer_class : str
        An optimizer class name to train SirenVis

    optimizer_param : dict
        Optimizer constructor arguments

    resume : bool
        If True, and if a checkopint file is provided for the model, resume training 
        with the optimizer state restored from the last checkpoint step.

    '''

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.get('device'):
        DEVICE = get_device(cfg['device']['type'])

    iteration_ctr = 0
    epoch_ctr = 0

    lengths = lartpc.get_lengths.to(DEVICE)
    ranges = torch.column_stack([-lengths/2,lengths/2])
    aabox = AABox(ranges)
    # Create necessary pieces: the model, optimizer, loss, logger.
    # Load the states if this is resuming.
    net = SirenVis(cfg,aabox).to(DEVICE)

    dl = MCDataLoader(mc, cfg, device=DEVICE)

    opt, sch, epoch = optimizer_factory(net.parameters(),cfg)
    if epoch >0:
        iteration_ctr = int(epoch * len(dl))
        epoch_ctr = int(epoch)
        print('[train] resuming training from iteration',iteration_ctr,'epoch',epoch_ctr)
    criterion = WeightedL2Loss() 
    logger = CSVLogger(cfg)
    logdir = logger.logdir

    # Set the control parameters for the training loop
    train_cfg = cfg.get('train',dict())
    epoch_max = train_cfg.get('max_epochs',int(1e20))
    iteration_max = train_cfg.get('max_iterations',int(1e20))
    save_every_iterations = train_cfg.get('save_every_iterations',-1)
    save_every_epochs = train_cfg.get('save_every_epochs',-1)  
    print(f'[train] train for max iterations {iteration_max} or max epochs {epoch_max}')

    # Store configuration
    with open(os.path.join(logdir,'train_cfg.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)
    
    # Start the training loop
    t0=time.time()
    twait=time.time()
    stop_training = False

    while iteration_ctr < iteration_max and epoch_ctr < epoch_max:
        opt.zero_grad()
        for batch_idx, data in enumerate(tqdm(dl,desc='Epoch %-3d' % epoch_ctr)):
            iteration_ctr += 1

            pos = data['position'].to(DEVICE)
            ang = (data['angle']/180.).to(DEVICE)
            # Input data prep
            x              = torch.cat((pos/lengths,ang),dim=1)
            weights        = data['weight'].to(DEVICE)
            target         = data['target'].to(DEVICE)
            target_linear  = data['value'].to(DEVICE)

            twait = time.time()-twait
            # Running the model, compute the loss, back-prop gradients to optimize.
            ttrain = time.time()
            pred   = net(x)
            loss   = criterion(pred, target, weights)
            loss.backward()
            ttrain = time.time()-ttrain
            
            # Log training parameters
            logger.record(['iter','epoch','loss','ttrain','twait'],
                          [iteration_ctr, epoch_ctr, loss.item(),ttrain,twait])
            twait = time.time()
            
            # Step the logger
            pred_linear = dl.inv_xform_vis(pred)
            logger.step(iteration_ctr, target_linear, pred_linear)
                
            # Save the model parameters if the condition is met
            if save_every_iterations > 0 and iteration_ctr % save_every_iterations == 0:
                filename = os.path.join(logdir,'iteration-%06d-epoch-%04d.ckpt' % (iteration_ctr,epoch_ctr))
                net.save_state(filename,opt,sch,iteration_ctr)
            
            if iteration_max <= iteration_ctr:
                stop_training=True
                break

        for param in net.parameters():
            param.grad /= len(dl)
        opt.step()
        if stop_training:
            break

        if sch is not None:
            sch.step()

        epoch_ctr += 1

        if (save_every_epochs*epoch_ctr) > 0 and epoch_ctr % save_every_epochs == 0:
            filename = os.path.join(logdir,'iteration-%06d-epoch-%04d.ckpt' % (iteration_ctr,epoch_ctr))
            net.save_state(filename,opt,sch,iteration_ctr/len(dl))

            
    print('[train] Stopped training at iteration',iteration_ctr,'epochs',epoch_ctr)
    logger.write()
    logger.close()

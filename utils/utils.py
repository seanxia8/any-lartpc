import torch
import yaml
import glob
import os
def place_hexa_opticsensors(lx, ly, lz, spacing_y, spacing_z):
    '''
    Generate hexagonal grid of PMTs on the y-z plane of a cubic detector
    :param lx: length of the cube in x direction
    :param ly: length of the cube in y direction
    :param lz: length of the cube in z direction
    :param spacing_y: spacing between PMTs in y direction
    :param spacing_z: spacing between PMTs in z direction
    :return:
    pmt_coords: coordinates of the PMTs
    '''

    grid_y = int(ly / spacing_y)
    grid_z = int(lz / spacing_z)
    print(f"Total PMT number is {2 * grid_y * grid_z}")

    # Generate hexagonal grid coordinates
    y_side, z_side = torch.meshgrid(torch.arange(grid_y), torch.arange(grid_z))
    y_side = y_side * spacing_y - ly / 2
    z_side = z_side * spacing_z - lz / 2
    y_side = y_side.to(torch.float32)
    z_side = z_side.to(torch.float32)
    for i in range(y_side.shape[1]):
        if i % 2 == 1:
            y_side[:, i] += spacing_y / 2

    y = torch.tile(y_side, (2, 1, 1)).reshape(2, -1)
    z = torch.tile(z_side, (2, 1, 1)).reshape(2, -1)
    x = torch.vstack((torch.full((y.shape[1],), -lx / 2), torch.full((y.shape[1],), lx / 2)))
    # print(x, y, z)
    # Shift every other row to create a hexagonal pattern

    #pmt_coords = torch.column_stack((x.flatten(), y.flatten(), z.flatten()))
    # change to swap between the sides
    pmt_coords = torch.stack((x,y,z), dim=-1)
    return pmt_coords

def place_photon_origin(lx, ly, lz, spacing_photon):
    '''
    Generate coordinates for photon origin
    :param lx: length of the cube in x direction
    :param ly: length of the cube in y direction
    :param lz: length of the cube in z direction
    :param n_photon: number of photons
    :param spacing_photon: spacing between photons
    :return:
    photon_coords: coordinates of the photons
    '''

    x_grid = int(lx / spacing_photon)
    y_grid = int(ly / spacing_photon)
    z_grid = int(lz / spacing_photon)
    print(f"Total photon position is {x_grid * y_grid * z_grid}")

    x, y, z = torch.meshgrid(torch.arange(x_grid), torch.arange(y_grid), torch.arange(z_grid))
    x = x * spacing_photon - lx / 2
    y = y * spacing_photon - ly / 2
    z = z * spacing_photon - lz / 2

    x = x.to(torch.float32)
    y = y.to(torch.float32)
    z = z.to(torch.float32)

    photon_coords = torch.column_stack((x.flatten(), y.flatten(), z.flatten()))
    return photon_coords

def get_config_dir():
    return os.path.join(os.path.dirname(__file__),'../config')

def list_config(full_path=False):

    fs = glob.glob(os.path.join(get_config_dir(), '*.yaml'))

    if full_path:
        return fs

    return [os.path.basename(f)[:-5] for f in fs]

def get_config(name):

    options = list_config()
    results = list_config(True)

    if name in options:
        return results[options.index(name)]

    alt_name = name + '.yaml'
    if alt_name in options:
        return results[options.index(alt_name)]

    print('No data found for config name:',name)
    raise NotImplementedError
def load_config(name:str):

    return yaml.safe_load(open(get_config(name),'r'))
def get_unique_filename(filename):
    base, ext = os.path.splitext(filename)  # Split the name and extension
    index = 1
    new_filename = filename

    while os.path.exists(new_filename):
        new_filename = f"{base}_{index}{ext}"
        index += 1

    return new_filename
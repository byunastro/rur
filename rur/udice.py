from rur.io_dice import io_dice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import minimize
from scipy.stats import binned_statistic
from scipy.stats import linregress
import warnings
import os
from PIL import Image
import glob

class DiceIC:
    def __init__(self, repo):
        self.repo = repo

    def get_part(self,repo=None, ptype=1):
        # Ptype: 0=Gas, 1=Halo, 2=Disk, 3=Bulge, 4=Stars
        # If ptype < 0, then all particles will return
        if(repo is None):
            repo = self.repo
        if not os.path.exists(repo):
            warnings.warn("File does not exist.", UserWarning)
            return
        
        io_dice.read_gadget(repo)
        dtype = np.dtype([('x', 'float'), ('y', 'float'), ('z', 'float'), ('vx', 'float'), ('vy', 'float'), ('vz', 'float'), ('id', 'float'), ('m', 'float')])

        pos=io_dice.pos
        vel=io_dice.vel
        mm=io_dice.mm
        id=io_dice.id
        types=io_dice.type
        x,y,z = pos[0], pos[1], pos[2]
        vx,vy,vz = vel[0], vel[1], vel[2]
        if ptype < 0:
            struct_array = np.zeros(len(types), dtype=dtype)
            # unit_l = 1.0kpc/h
            struct_array['x'] = x
            struct_array['y'] = y
            struct_array['z'] = z
            # unit_v = 1km/sec
            struct_array['vx'] = vx
            struct_array['vy'] = vy
            struct_array['vz'] = vz
            struct_array['id'] = id
            # unit_m = 1 solar mass /h
            struct_array['m'] = mm * 1e10
        else:
            ind = (types == ptype)
            struct_array = np.zeros(len(types[ind]), dtype=dtype)
            # unit_l = 1.0kpc/h
            struct_array['x'] = x[ind]
            struct_array['y'] = y[ind]
            struct_array['z'] = z[ind]
            # unit_v = 1km/sec
            struct_array['vx'] = vx[ind]
            struct_array['vy'] = vy[ind]
            struct_array['vz'] = vz[ind]
            struct_array['id'] = id[ind]
            # unit_m = 1 solar mass /h
            struct_array['m'] = mm[ind] * 1e10
        io_dice.read_gadget_deallocate()
        return struct_array
    
def plotimages(component, lbox, vmin1=None, vmax1=None, vmin2=None, vmax2=None, dpi=150):
    fig, axes = plt.subplots(ncols=2, figsize=(9,4))
    fig.set_dpi(dpi)

    ax1 = axes[0]
    ax1.hist2d(component['x'], component['y'], bins=300, norm=LogNorm(vmin=vmin1, vmax=vmax1)
                , cmap='CMRmap', range=[[-lbox,lbox],[-lbox,lbox]])
    ax1.set_facecolor('k')
    ax1.set_aspect(1)
    ax1.set_xlabel('x [kpc]')
    ax1.set_ylabel('y [kpc]')
    ax1.set_title('Face')

    ax2 = axes[1]
    ax2.hist2d(component['x'], component['z'], bins=300, norm=LogNorm(vmin=vmin2, vmax=vmax2)
                , cmap='CMRmap', range=[[-lbox,lbox],[-lbox,lbox]])
    ax2.set_facecolor('k')
    ax2.set_aspect(1)
    ax2.set_xlabel('x [kpc]')
    ax2.set_ylabel('z [kpc]')
    ax2.set_title('Edge')
    return fig, axes

def make_gif(frame_folder, fname, duration=150):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    # frame_one.save("Gas_fz5_lv10.gif", format="GIF", append_images=frames,
    frame_one.save("{}.gif".format(fname), format="GIF", append_images=frames,
               save_all=True, duration=150, loop=0)
    
def radial_density_profile(radius, weight, num_bins=20, symmetry='spherical', scale='log'):
    if np.min(radius) < 0.00001:
        r_min = 0.00001
    else:
        r_min = np.min(radius)
    if scale == 'log':
        bin_edges = np.logspace(np.log10(r_min)+1, np.log10(max(radius)), num_bins + 1)
    else:
        bin_edges = np.linspace(0, np.max(radius), num_bins + 1)
    if symmetry == 'spherical':
        volume = (4/3) * np.pi* bin_edges[1:]**3
    elif symmetry == 'cylindrical':
        volume = np.pi * bin_edges[1:]**2
    density, _ = np.histogram(radius, bins=bin_edges, weights=weight)
    average_density = np.cumsum(density) / volume
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_centers, average_density

def vertical_density_profile(z_data, weight, num_bins=20, scale='log'):
    if scale == 'log':
        bin_edges = np.logspace(-4, np.log10(max(z_data)), num_bins + 1)
    else:
        bin_edges = np.linspace(0, max(z_data), num_bins + 1)
    density, _ = np.histogram(z_data, bins=bin_edges, weights=weight)
    average_density = np.cumsum(density) / bin_edges[1:]
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_centers, average_density

def get_Re(radius, band, p=0.5):
    '''
    Get effective radius 
    Unit : kpc
    '''
    cum_lum = np.cumsum(band[np.argsort(radius)])
    sorted_rad = np.sort(radius)
    Re = sorted_rad[np.searchsorted(cum_lum, np.max(cum_lum) * p)]
    return Re

def get_Rvir(radius, mass, rho_crit=1.053e2):
    cum_mass = np.cumsum(mass[np.argsort(radius)])
    sorted_r = np.sort(radius)
    sorted_rho = np.flip(cum_mass / ((4*np.pi / 3) * (sorted_r**3)))
    r_vir = sorted_r[-np.searchsorted(sorted_rho, 200*rho_crit)]
    return r_vir

def rot(Lx,Ly,Lz,xs,ys,zs):
    #pi rotation
    cospi = Ly/np.sqrt(Lx**2+Ly**2)
    sinpi = np.sqrt(1-cospi**2) 
    if Lx>0:
        sinpi = -sinpi
    #theta rotaion
    costheta = Lz/np.sqrt(Lx**2+Ly**2+Lz**2)
    sintheta = -np.sqrt(1-costheta**2)

    R2 = np.array([
        1,  0,        0,
        0,  costheta, sintheta,
        0, -sintheta, costheta
    ]).reshape(3,3)

    R1 = np.array([
        cospi,  sinpi, 0,
        -sinpi, cospi, 0,
        0,      0,     1
    ]).reshape(3,3)

    R= np.matmul(R2,R1)
    IC = np.array(np.vstack((xs,ys,zs)))
    NC = np.matmul(R, IC)
    return NC[0],NC[1],NC[2]

def cartesian_to_cylindrical_velocity(vx, vy, x, y):
    r = np.sqrt(x**2 + y**2)
    
    vr = (x*vx + y*vy) / r
    vphi = (x*vy - y*vx) / r
    
    return vr, vphi

def cartesian_to_spherical_velocity(vx, vy, vz, x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    vr = (x * vx + y * vy + z * vz) / r 
    vtheta = (z * (x * vx + y * vy) - r**2 * vz) / (r * np.sqrt(x**2 + y**2))  # polar velocity
    vphi = (-y * vx + x * vy) / (x**2 + y**2)
    
    return vr, vtheta, vphi

def center_of_mass(x,y,z, mass):
    mass = np.array(mass)
    x,y,z = np.array(x), np.array(y), np.array(z)
    total_mass = np.sum(mass)
    center_x = np.sum(mass * x) / total_mass
    center_y = np.sum(mass * y) / total_mass
    center_z = np.sum(mass * z) / total_mass
    return np.array([center_x, center_y, center_z])
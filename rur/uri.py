
from os.path import join, exists
from parse import parse
from numpy.core.records import fromarrays as fromarrays

from scipy.integrate import cumtrapz

from rur.fortranfile import FortranFile
from rur.hilbert3d import hilbert3d
from rur.readr import readr
from rur.config import *
import numpy as np


def write_zoomparts_music(part_ini, cropped, filepath, reduce=None):
    """
    writes position table of particles in MUSIC format.

    :param part_ini: table of particles at start of the simulation
    :type part_ini: recarray or uri.RamsesSnapshot.Particle
    :param cropped: list particles in desired zoom region at certain redshift.
    :type cropped: recarray or uri.RamsesSnapshot.Particle
    :type filepath: str
    :param reduce: reduce the number of particles by this ratio by randomly sampling it.
    :type reduce: int
    :return: recarray or uri.RamsesSnapshot.Particle
    """
    cropped_ini = part_ini[np.isin(part_ini['id'], cropped['id'], True)]
    if reduce is not None:
        cropped_ini = np.random.choice(cropped_ini, cropped_ini.size//reduce, replace=False)
    np.savetxt(filepath, get_vector(cropped_ini))
    return cropped_ini

def write_parts_rockstar(part, snap, filepath):
    """
    writes particle data in ASCII format that can be read by Rockstar

    :param part:
    :param snap:
    :param filepath:
    """
    timer.start('Writing %d particles in %s... ' % (part.size, filepath), 1)

    pos = get_vector(part) * snap.params['boxsize']
    vel = get_vector(part, 'v') * snap.get_unit('v', 'km/s')
    table = fromarrays([*pos.T, *vel.T, part['id']], formats=['f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'i4'])
    np.savetxt(filepath, table, fmt=('%.16e',)*6 + ('%d',))

    timer.record()

def write_snaps_rockstar(repo, start, end, mode='none', path_in_repo='snapshots', ncpu=48, min_halo_particles=100):
    path = join(repo, 'rst')
    dm_flist = []
    star_flist = []
    for iout in np.arange(start, end):
        snap = RamsesSnapshot(repo, iout, mode, path_in_repo=path_in_repo)
        part = snap.get_part()

        filepath = join(path, 'dm_%05d.dat' % iout)
        write_parts_rockstar(part['dm'], snap, filepath)
        dm_flist.append(filepath)

        if(snap.params['star']):
            filepath = join(path, 'star_%05d.dat' % iout)
            write_parts_rockstar(part['star'], snap, filepath)
            star_flist.append(filepath)

    with open(join(path, 'dmlist.dat'), 'w') as opened:
        for fname in dm_flist:
            opened.write(fname+'\n')

    with open(join(path, 'dm.cfg')) as opened:
        opened.write('SNAPSHOT_NAMES = %s\n' % join(path, 'dmlist.dat'))
        opened.write('NUM_WRITERS = %d\n' % ncpu)
        opened.write('FILE_FORMAT = ASCII\n')

        opened.write('BOX_SIZE = %.3f\n' % snap.params['boxsize'])
        opened.write('PARTICLE_MASS = %.3f\n' % np.min(snap.part['m']) * snap.get_unit('m', 'Msun') * snap.params['h'])
        opened.write('h0 = %.4f\n' % snap.params['h'])
        opened.write('Ol = %.4f\n' % snap.params['omega_l'])
        opened.write('Om = %.4f\n' % snap.params['omega_m'])

        opened.write('MIN_HALO_PARTICLES = %d\n' % min_halo_particles)

    if (snap.params['star']):

        with open(join(path, 'starlist.dat'), 'w') as opened:
            for fname in star_flist:
                opened.write(fname+'\n')

        with open(join(path, 'star.cfg')) as opened:
            opened.write('SNAPSHOT_NAMES = %s\n' % join(path, 'starlist.dat'))
            opened.write('NUM_WRITERS = %d\n' % ncpu)
            opened.write('FILE_FORMAT = ASCII\n')

            opened.write('BOX_SIZE = %.3f\n' % snap.params['boxsize'])
            opened.write('PARTICLE_MASS = %.3f\n' % np.min(snap.part['m']) * snap.get_unit('m', 'Msun') * snap.params['h'])
            opened.write('h0 = %.4f\n' % snap.params['h'])
            opened.write('Ol = %.4f\n' % snap.params['omega_l'])
            opened.write('Om = %.4f\n' % snap.params['omega_m'])

            opened.write('MIN_HALO_PARTICLES = %d\n' % min_halo_particles)

def cut_spherical(table, center, radius, prefix='', ndim=3, inverse=False):
    distances = rss(center - get_vector(table, prefix, ndim))
    if(inverse):
        mask = distances > radius
    else:
        mask = distances <= radius
    return table[mask]

def cut_halo(table, halo, radius=1, code_unit=False, inverse=False, radius_name='rvir'):
    center = get_vector(halo)
    if(code_unit):
        radius = radius
    else:
        radius = halo[radius_name] * radius
    return cut_spherical(table, center, radius, inverse=inverse)

def classify_part(part, pname):
    # classity particles, if familty is given, use it.
    # if there is no field 'family', use id, mass and epoch instead.
    timer.start('Classifying %d particles... ' % part.size, 2)
    names = part.dtype.names
    if('family' in names):
        # Do a family-based classification
        mask = np.isin(part['family'], part_family[pname])

    elif('epoch' in names):
        # Do a parameter-based classification
        if(pname == 'dm'):
            mask = (part['epoch'] == 0) & (part['id'] > 0)
        elif(pname == 'star'):
            mask = ((part['epoch'] < 0) & (part['id'] > 0))\
                   | ((part['epoch'] != 0) & (part['id'] < 0))
        elif(pname == 'sink' or pname == 'cloud'):
            mask = (part['id'] < 0) & (part['m'] > 0) & (part['epoch'] == 0)
        elif(pname == 'tracer'):
            mask = (part['id'] < 0) & (part['m'] == 0)
        else:
            mask = False
    elif('id' in names):
        # DM-only simulation
        if(pname == 'dm'):
            mask =  part['id'] > 0
        elif(pname == 'tracer'):
            mask = (part['id'] < 0) & (part['m'] == 0)
        else:
            mask = False
    else:
        # No particle classification is possible
        raise ValueError('Particle data structure not classifiable.')
    output = part[mask]

    timer.record()
    return output

def find_smbh(part, verbose=None):
    # Find SMBHs by merging sink (cloud) particles
    if(verbose is not None):
        timer.verbose = verbose
    timer.start('Searching for SMBHs...', 1)
    sink = classify_part(part, 'cloud')
    names = part.dtype.names
    ids = np.unique(sink['id'])

    smbh = []
    for id in ids:
        smbh_cloud = sink[sink['id'] == id]
        row = ()
        for name in names:
            if (name == 'm'):
                val = np.sum(smbh_cloud[name])
            elif (name == 'family'):
                val = 4
            elif (name == 'tag'):
                val = 0
            else:
                val = np.average(smbh_cloud[name])
            row += (val,)
        smbh.append(row)
    smbh = np.array(smbh, dtype=part.dtype)
    timer.record()
    if(timer.verbose >=1):
        print('Found %d SMBHs.' % smbh.size)
    timer.verbose = verbose
    return smbh

def age(part, snap):
    # particle to age
    table = snap.cosmo_table
    ages = snap.params['age'] - np.interp(part['epoch'], table['u'], table['t'])
    return ages * snap.unit['Gyr']

def aform(part, snap):
    # particle to formation epoch (scale factor)
    table = snap.cosmo_table
    aexp = np.interp(part['epoch'], table['u'], table['aexp'])
    return aexp

def box_mask(coo, box, size=None, exclusive=False):
    # masking coordinates based on the box
    if size is not None:
        size = expand_shape(size, [0], 2)
    else:
        size = 0
    if(exclusive):
        size *= -1

    box_mask = np.all((box[:, 0] <= coo+size/2) & (coo-size/2 <= box[:, 1]), axis=-1)
    return box_mask

def interpolate_part(part1, part2, name, fraction=0.5):
    id1 = part1['id']
    id2 = part2['id']

    id1 = np.abs(id1)
    id2 = np.abs(id2)

    part_size = np.maximum(np.max(id1), np.max(id2))+1

    val1 = part1[name]
    val2 = part2[name]

    if(name == 'pos' or name == 'vel'):
        pool = np.zeros((part_size, 3), dtype='f8')
    else:
        pool = np.zeros(part_size, dtype=val1.dtype)

    mask1 = np.zeros(part_size, dtype='?')
    mask2 = np.zeros(part_size, dtype='?')

    mask1[id1] = True
    mask2[id2] = True

    active_mask = mask1 & mask2

    pool[id1] += val1 * (1. - fraction)
    pool[id2] += val2 * fraction
    val = pool[active_mask]

    if(timer.verbose>=2):
        print("Particle interpolation - part1[%d], part2[%d], result[%d]" % (id1.size, id2.size, np.sum(active_mask)))

    return val

def time_series(repo, iouts, halo_table, mode='none', extent=None, unit=None):
    # returns multiple snapshots from repository and array of iouts
    snaps = []
    snap = None
    for halo, iout in zip(halo_table, iouts):
        snap = RamsesSnapshot(repo, iout, mode, snap=snap)
        if(extent is None):
            extent_now = halo['rvir']*2
        else:
            extent_now = extent * snap.unit[unit]
        box = get_box(get_vector(halo), extent_now)
        snap.box = box
        snaps.append(snap)
    return snaps

def get_cpulist(box, binlvl, maxlvl, bound_key, ndim, n_divide):
    # get list of cpus involved in selected box.
    volume = np.prod([box[:, 1] - box[:, 0]])
    if (binlvl is None):
        binlvl = int(np.log2(1. / (volume + 1E-20)) / ndim) + n_divide
    if (binlvl > 64 // ndim):
        binlvl = 64 // ndim - 1
    lower, upper = np.floor(box[:, 0] * 2 ** binlvl).astype(int), np.ceil(box[:, 1] * 2 ** binlvl).astype(int)
    bbox = np.stack([lower, upper], axis=-1)

    bin_list = cartesian(
        np.arange(bbox[0, 0], bbox[0, 1]),
        np.arange(bbox[1, 0], bbox[1, 1]),
        np.arange(bbox[2, 0], bbox[2, 1]))  # TODO: generalize this

    if (timer.verbose >= 2):
        print("Setting bin level as %d..." % binlvl)
        print("Input box:", box)
        print("Bounding box:", bbox)
        ratio = np.prod([bbox[:, 1] / 2 ** binlvl - bbox[:, 0] / 2 ** binlvl]) / volume
        print("Volume ratio:", ratio)
        print("N. of Blocks:", bin_list.shape[0])

    keys = hilbert3d(*(bin_list.T), binlvl, bin_list.shape[0])
    keys = np.array(keys)
    key_range = np.stack([keys, keys + 1], axis=-1)
    key_range = key_range.astype('f8')

    involved_cpu = []
    for icpu_range, key in zip(
            np.searchsorted(bound_key / 2. ** (ndim * (maxlvl - binlvl + 1)), key_range),
            key_range):
        involved_cpu.append(np.arange(icpu_range[0], icpu_range[1] + 1))
    involved_cpu = np.unique(np.concatenate(involved_cpu)) + 1
    if (timer.verbose >= 2):
        print("List of involved CPUs: ", involved_cpu)
    return involved_cpu


class RamsesSnapshot(object):
    """A handy object to store RAMSES AMR/Particle snapshot data.

    One RamsesSnapshot represents single snapshot of the simulation. Efficient reads from RAMSES binary are guaranteed
    by only touching necessary files.

    Parameters
    ----------
    repo : string
        Repository path, path to directory where RAMSES raw data is stored.
    iout : integer
        RAMSES output number
    mode : 'none', 'yzics', 'nh', 'gem', etc.
        RAMSES version. Define it in config.py if necessary
    box : numpy.array with shape (3, 2)
        Bounding box of region to open in the simulation, in code unit, should be within (0, 1)
    path_in_repo : string
        Path to actual snapshot directories in the repository path, path where 'output_XXXXX' is stored.
    full_path : string
        Overrides repo and path_in_repo, full path to directory where 'output_XXXXX' is stored.
    snap : RamsesSnapshot object
        Previously created snapshot object from same simulation (but different output number) saves time to compute
        cosmology.

    Attributes
    ----------
    part : RamsesSnapshot.Particle object
        Loaded particle table in the box
    cell : RamsesSnapshot.Cell object
        Loaded particle table in the box
    box : numpy.array with shape (3, 2)
        Current bounding box. Change this value if bounding box need to be changed.
    part_data : numpy.recarray
        Currently cached particle data, if data requested by get_part() have already read, retrieve from this instead.
    cell_data : numpy.recarray
        Currently cached cell data, if data requested by get_cell() have already read, retrieve from this instead.

    Examples
    ----------
    To read data from ramses raw binary data:

    >>> from rur.rur import uri
    >>> repo = "path/to/your/ramses/outputs"
    >>> snap = uri.RamsesSnapshot(repo, iout=1, path_in_repo='')
    >>> snap.box = np.array([[0, 1], [0, 1], [0, 1]])

    To read particle and cell data:

    >>> snap.get_part()
    >>> snap.get_cell()
    >>> print(snap.part.dtype)
dtype((numpy.record, [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'), ('m', '<f8'), ('epoch', '<f8'), ('metal', '<f8'), ('id', '<i4'), ('level', 'u1'), ('cpu', '<i4')]))

    >>> print(snap.cell.dtype)
dtype((numpy.record, [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('rho', '<f8'), ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'), ('P', '<f8'), ('metal', '<f8'), ('zoom', '<f8'), ('level', '<i4'), ('cpu', '<i4')]))

    >>> snap.clear()

    """

    def __init__(self, repo, iout, mode='none', box=None, path_in_repo='snapshots', full_path=False, snap=None, longint=False):
        self.iout = iout
        if(full_path):
            self.snap_path = repo
        else:
            self.repo = repo
            self.path_in_repo = path_in_repo
            self.snap_path = join(repo, path_in_repo)
        self.path = join(self.snap_path, output_format.format(snap=self))

        self.params = {}

        self.mode = mode
        self.info_path = join(self.path, info_format[mode].format(snap=self))
        self.data_path = join(self.path, data_format[mode].format(snap=self))

        self.part_data = None
        self.cell_data = None

        self.part = None
        self.cell = None

        self.longint = longint

        if(mode == 'ng'):
            self.classic_format = False
        else:
            self.classic_format = True

        self.read_params(snap)
        if(box is not None):
            self.box = np.array(box)
        else:
            self.box = default_box
        self.box_cell = None
        self.box_part = None



    def __getitem__(self, item):
        return self.params[item]

    def __getattr__(self, item):
        return self.__getitem__(item)

    def get_path(self, type='amr', icpu=1):
        return self.data_path.format(type=type, icpu=icpu)

    def set_cosmology(self, params=None, n=5000, snap=None):
        if(params is None):
            params = self.params

        if(snap is None):
            # Integrate manually because astropy cosmology calculation is too slow...
            aarr = np.linspace(0, 1, n)[1:] ** 2
            aarr_st = (aarr[:-1] + aarr[1:])/2
            duda = 1. / (aarr_st ** 3 * np.sqrt(params['omega_m'] * aarr_st ** -3 + params['omega_l']))
            dtda = 1. / (params['H0'] * km * Gyr / Mpc * aarr_st * np.sqrt(params['omega_m'] * aarr_st ** -3 + params['omega_l']))
            aarr = aarr[1:]

            uarr = cumtrapz(duda[::-1], aarr[::-1], initial=0)[::-1]
            tarr = cumtrapz(dtda, aarr, initial=0)
            self.cosmo_table = fromarrays([aarr, tarr, uarr], dtype=[('aexp', 'f8'), ('t', 'f8'), ('u', 'f8')])
        else:
            self.cosmo_table = snap.cosmo_table

        self.params['age'] = np.interp(params['aexp'], self.cosmo_table['aexp'], self.cosmo_table['t'])

        if(timer.verbose>=1):
            print('Age of the universe (now/z=0): %.3f / %.3f Gyr, z = %.5f' % (self.params['age'], self.cosmo_table['t'][-1], params['z']))

    def set_extra_fields(self, params=None):
        custom_extra_fields(self)

    def set_unit(self):
        custom_units(self)

    def set_box(self, center, extent, code_unit=True):
        if(not code_unit):
            extent = extent/self['boxsize_physical']
        self.box = get_box(center, extent)

    def set_box_halo(self, halo, radius=1, code_unit=False, radius_name='rvir'):
        center = get_vector(halo)
        if(not code_unit):
            extent = halo[radius_name] * radius * 2
        else:
            extent = radius * 2
        self.set_box(center, extent)

    def read_params(self, snap):

        opened = open(self.info_path, mode='r')

        format1 = '{name:<12}={data:>11d}\n'
        format2 = '{name:<12}={data:>2.15e}\n'
        format3 = '{:14}{ordering:20}'
        format4 = '{domain:>8d} {ind_min:>2.15e} {ind_max:>2.15e}'

        params = {}
        for _ in range(6):
            parsed = parse(format1, opened.readline())
            params[parsed['name'].strip()] = parsed['data']

        opened.readline()

        for _ in range(11):
            parsed = parse(format2, opened.readline())
            params[parsed['name'].strip()] = parsed['data']

        # some cosmological calculations
        params['unit_m'] = params['unit_d'] * params['unit_l']**3
        params['h'] = params['H0']/100
        params['z'] = 1/params['aexp'] - 1
        params['boxsize'] = params['unit_l'] * params['h'] / Mpc / params['aexp']
        params['boxsize_physical'] = params['boxsize'] / (params['h']) * params['aexp']
        params['boxsize_comoving'] = params['boxsize'] / (params['h'])

        self.part_dtype = part_dtype[self.mode]
        self.hydro_names = hydro_names[self.mode]

        if(self.classic_format):
            opened.readline()
            params['ordering'] = parse(format3, opened.readline())['ordering'].strip()
            opened.readlines(2)
            if(params['ordering'] == 'hilbert'):
                bound_key = []
                for icpu in np.arange(1, params['ncpu']):
                    parsed = parse(format4, opened.readline())
                    bound_key.append(parsed['ind_max'])
                self.bound_key = np.array(bound_key)

            if(exists(self.get_path('part', 1))):
                self.params['star'] = self._read_nstar()>0
            else:
                self.params['star'] = False

            # check if star particle exists
            if(not self.params['star']):
                if(self.mode == 'nh'):
                    self.part_dtype = part_dtype['nh_dm_only']
                elif (self.mode == 'yzics'):
                    self.part_dtype = part_dtype['yzics_dm_only']
            if(self.longint):
                if(self.mode == 'iap' or self.mode == 'gem'):
                    self.part_dtype = part_dtype['gem_longint']
        else:
            self.params['star'] = True

        # initialize cpu list and boundaries
        self.cpulist_cell = np.array([], dtype='i4')
        self.cpulist_part = np.array([], dtype='i4')
        self.bound_cell = np.array([0], dtype='i4')
        self.bound_part = np.array([0], dtype='i4')
        self.params.update(params)

        self.set_cosmology(snap=snap)
        self.set_extra_fields()
        self.set_unit()

    def get_involved_cpu(self, box=None, binlvl=None, n_divide=5):
        # get the list of involved cpu domain for specific region.
        if(box is None):
            box = self.box
        if(self.classic_format and not box is None):
            box = np.array(box)
            maxlvl = self.params['levelmax']

            involved_cpu = get_cpulist(box, binlvl, maxlvl, self.bound_key, self.ndim, n_divide)
        else:
            involved_cpu = np.arange(self.params['ncpu']) + 1
        return involved_cpu


    def read_sink_table(self):
        if(self.mode=='nh'):
            table = np.genfromtxt(self.path+'/sink_%05d.csv' % self.iout, dtype=sink_table_format, delimiter=',')
        else:
            raise ValueError('This function works only for NH-version RAMSES')
        return table

    def read_part(self, target_fields=None, cpulist=None):
        """Reads particle data from current box.

        Parameters
        ----------
        target_fields: list of string
            target field name to read. If None is passed, read all fields.
        cpulist: list of integer
            target list of cpus, if specified, read selected cpus regardless of current box.
        Returns
        -------
        part : RamsesSnapshot.Particle object
            particle data, can be accessed as attributes also.

        """
        if(cpulist is None):
            cpulist = self.get_involved_cpu()
        if (self.part_data is not None):
            if(timer.verbose>=1): print('Searching for extra files...')
            cpulist = cpulist[np.isin(cpulist, self.cpulist_part, assume_unique=True, invert=True)]

        if (cpulist.size > 0):
            timer.start('Reading %d part files in %s... ' % (cpulist.size, self.path), 1)
            progress_bar = cpulist.size > progress_bar_limit and timer.verbose >= 1
            readr.read_part(self.snap_path, self.iout, cpulist, self.mode, progress_bar, self.longint)
            timer.record()

            if(self.longint):
                arr = [*readr.real_table.T, *readr.long_table.T, *readr.integer_table.T, *readr.byte_table.T]
            else:
                arr = [*readr.real_table.T, *readr.integer_table.T, *readr.byte_table.T]

            dtype = self.part_dtype

            if target_fields is not None:
                target_idx = np.where(np.isin(np.dtype(dtype).names, target_fields))[0]
                arr = [arr[idx] for idx in target_idx]
                dtype = [dtype[idx] for idx in target_idx]

            timer.start('Building table for %d particles... ' % readr.integer_table.shape[0], 1)
            part = fromarrays(arr, dtype=dtype)
            readr.close()

            bound = compute_boundary(part['cpu'], cpulist)
            if (self.part_data is None):
                self.part_data = part
            else:
                self.part_data = np.concatenate([self.part_data, part])

            self.bound_part = np.concatenate([self.bound_part[:-1], self.bound_part[-1] + bound])
            self.cpulist_part = np.concatenate([self.cpulist_part, cpulist])
            timer.record()

        else:
            if (timer.verbose >= 1):
                print('CPU list already satisfied.')

    def read_cell(self, target_fields=None, cpulist=None):
        """Reads amr data from current box.

        Parameters
        ----------
        target_fields: list of string
            target field name to read. If None is passed, read all fields.
        cpulist: list of integer
            target list of cpus, if specified, read selected cpus regardless of current box.
        Returns
        -------
        cell : RamsesSnapshot.Cell object
            amr data, can be accessed as attributes also.

        """
        if(cpulist is None):
            cpulist = self.get_involved_cpu()
        if(self.cell_data is not None):
            if(timer.verbose>=1):
                print('Searching for extra files...')
            cpulist = cpulist[np.isin(cpulist, self.cpulist_cell, assume_unique=True, invert=True)]

        if(cpulist.size > 0):
            timer.start('Reading %d AMR & hydro files in %s... ' % (cpulist.size, self.path), 1)

            progress_bar = cpulist.size > progress_bar_limit and timer.verbose >= 1
            readr.read_cell(self.snap_path, self.iout, cpulist, self.mode, progress_bar)
            self.params['nhvar'] = int(readr.nhvar)
            timer.record()

            formats = ['f8'] * self.params['ndim'] + ['f8'] * self.params['nhvar'] + ['i4'] * 2

            names = list(dim_keys[:self.params['ndim']]) + self.hydro_names + ['level', 'cpu']

            arr = [*readr.real_table.T, *readr.integer_table.T]

            if(len(arr) != len(names)):
                raise Warning(
                    "Number of fields and size of the hydro array does not match\n"
                    "This is likely due to wrong selection of RAMSES version.")

            if target_fields is not None:
                target_idx = np.where(np.isin(names, target_fields))[0]
                arr = [arr[idx] for idx in target_idx]
                formats = [formats[idx] for idx in target_idx]
                names = [names[idx] for idx in target_idx]

            timer.start('Building table for %d cells... ' % (arr[0].size), 1)
            cell = fromarrays(arr, formats=formats, names=names)
            readr.close()

            bound = compute_boundary(cell['cpu'], cpulist)
            if (self.cell_data is None):
                self.cell_data = cell
            else:
                self.cell_data = np.concatenate([self.cell_data, cell])

            self.bound_cell = np.concatenate([self.bound_cell[:-1], self.bound_cell[-1] + bound])
            self.cpulist_cell = np.concatenate([self.cpulist_cell, cpulist])
            timer.record()

        else:
            if(timer.verbose>=1):
                print('CPU list already satisfied.')

    def clear(self, part=True, cell=True):
        """Clear exsisting cache from snapshot data.

        Parameters
        ----------
        part : bool
            if True, clear particle cache
        cell : bool
            if True, clear amr cache

        Returns
        -------

        """
        self.box=None
        if(part):

            self.part_data = None
            self.part = None
            self.cpulist_part = np.array([], dtype='i4')
            self.bound_part = np.array([0], dtype='i4')
        if(cell):
            self.cell_data = None
            self.cell = None
            self.cpulist_cell = np.array([], dtype='i4')
            self.bound_cell = np.array([0], dtype='i4')

    def _read_nstar(self):
        part_file = FortranFile(self.get_path('part', 1))
        part_file.skip_records(4)
        if(not self.longint):
            return part_file.read_ints()
        else:
            return part_file.read_longs()

    def _read_amr_cpu(self, icpu):

        ncpu = self.params['ncpu']
        ndim = self.params['ndim']
        levelmax = self.params['levelmax']

        params = {}
        # opening amr file
        print(self.get_path('amr', icpu))
        amr_file = FortranFile(self.get_path('amr', icpu))

        amr_file.skip_records(2)  # skip ncpu, ndim
        amr_file.skip_records(1)  # skip nx, xy, nz
        amr_file.skip_records(2)  # skip nlevelmax, ngridmax

        nboundary = amr_file.read_ints()[0]

        amr_file.skip_records(1)
        # params['boxlen'] = file.read_reals()[0]  # known
        amr_file.skip_records(1)  # skip boxlen
        amr_file.skip_records(13)

        ngridlevel = np.reshape(amr_file.read_ints(), (levelmax, ncpu))

        amr_file.skip_records(1)

        if nboundary > 0:
            amr_file.skip_records(2)
            params['ngridbound'] = amr_file.read_ints()[0]
            # TODO something with boundary?

        amr_file.skip_records(2)

        if self.params['ordering'] == 'bisection':
            amr_file.skip_records(5)
        else:
            amr_file.skip_records(1)

        amr_file.skip_records(3)

        # opening hydro file
        hydro_file = FortranFile(self.get_path('hydro', icpu))
        hydro_file.skip_records(1)
        params['nvarh'] = hydro_file.read_ints()[0]
        hydro_file.skip_records(4)

        gridmap = ngridlevel > 0
        skip_block_size = 3 * (2 ** ndim + ndim) + 1

        xg_cpu = []
        ref_cpu = []
        var_cpu = []
        lmap_cpu = []

        for ilevel, jcpu in np.argwhere(gridmap) + 1:
            amr_file.skip_records(3)

            if (jcpu == icpu):
                xg = amr_file.read_arrays(ndim, dtype='f8').T
                xg_cpu.append(xg)

                amr_file.skip_records(1)
                amr_file.skip_records(2 * ndim)

                son = amr_file.read_arrays(2 ** ndim, dtype='i4').T
                ref_cpu.append(son>0)

                amr_file.skip_records(2 * 2 ** ndim)

                lmap_cpu.append(np.full(xg.shape[0], ilevel))

            else:
                amr_file.skip_records(skip_block_size)

        for ilevel in np.arange(1, levelmax + 1):
            for jcpu in np.arange(1, ncpu + 1):
                hydro_file.skip_records(2)

                if gridmap[ilevel - 1, jcpu - 1]:
                    if jcpu == icpu:
                        var = []
                        for _ in np.arange(2 ** ndim):
                            array = hydro_file.read_arrays(params['nvarh'], dtype='f8')
                            var.append(array)
                        var = np.array(var).T
                        var_cpu.append(var)

                    else:
                        hydro_file.skip_records(2 ** ndim * params['nvarh'])
        amr_file.close()
        hydro_file.close()

        xg_cpu = np.concatenate(xg_cpu)
        ref_cpu = np.concatenate(ref_cpu)
        var_cpu = np.concatenate(var_cpu)
        lmap_cpu = np.concatenate(lmap_cpu)

        return xg_cpu, ref_cpu, var_cpu, lmap_cpu

    def read_hydro_ng(self):
        hydro_uolds = []
        for icpu in np.arange(1, self.params['ncpu'] + 1):
            path = self.get_path(type='hydro', icpu=icpu)
            opened = open(path, mode='rb')

            header = np.fromfile(opened, dtype=np.int32, count=4)
            ndim, nvar, levelmin, nlevelmax = header # nothing to do

            nocts = np.fromfile(opened, dtype=np.int32, count=nlevelmax - levelmin + 1)
            hydro_uold = np.fromfile(opened, dtype=np.float64)

            hydro_uolds.append(hydro_uold)

            opened.close()

        hydro_uolds = np.concatenate(hydro_uolds)
        hydro_uolds = np.reshape(hydro_uolds, [-1, nvar, 2 ** ndim])
        hydro_uolds = np.swapaxes(hydro_uolds, 1, 2)

        self.params['nvar'] = nvar
        self.var = hydro_uolds

    def read_amr_ng(self):
        amr_poss = []
        amr_lvls = []
        amr_refs = []
        amr_cpus = []

        for icpu in np.arange(1, self.params['ncpu'] + 1):
            amr_path = self.get_path(type='amr', icpu=icpu)
            opened = open(amr_path, mode='rb')

            header = np.fromfile(opened, dtype=np.int32, count=3)
            ndim, levelmin, nlevelmax = header

            nocts = np.fromfile(opened, dtype=np.int32, count=nlevelmax - levelmin + 1)
            amr_size = np.sum(nocts)

            block_size = ndim + 2 ** ndim
            amr = np.fromfile(opened, dtype=np.int32)
            amr = np.reshape(amr, [-1, block_size])

            poss, lvls = ckey2idx(amr[:, :ndim], nocts, levelmin)
            amr_poss.append(poss)
            amr_lvls.append(lvls)
            amr_refs.append(amr[:, ndim:].astype(np.bool))
            amr_cpus.append(np.full(amr_size, icpu))

            opened.close()

        self.x = np.swapaxes(np.concatenate(amr_poss), 1, 2)
        self.lvl = np.concatenate(amr_lvls)
        self.ref = np.concatenate(amr_refs)
        self.cpu = np.concatenate(amr_cpus)

    def get_cell(self, box=None, target_fields=None, domain_slicing=True, exact_box=True, cpulist=None):
        if(box is not None):
            self.box = box
        if(self.box is None or np.array_equal(self.box, default_box)):
            domain_slicing = False
            exact_box = False
        if(self.box is None or not np.array_equal(self.box, self.box_cell)):
            if(cpulist is None):
                cpulist = self.get_involved_cpu()
            else:
                domain_slicing=False
                exact_box=False
            self.read_cell(target_fields=target_fields, cpulist=cpulist)
            if(domain_slicing):
                cell = domain_slice(self.cell_data, cpulist, self.cpulist_cell, self.bound_cell)
            else:
                cell = self.cell_data

            if(exact_box):
                mask = box_mask(get_vector(cell), self.box, size=self.cell_extra['dx'](cell))
                timer.start('Masking cells... %d / %d (%.4f)' % (np.sum(mask), mask.size, np.sum(mask)/mask.size), 1)
                cell = cell[mask]
                timer.record()

            cell = self.Cell(cell, self)
            self.box_cell = self.box
            self.cell = cell
        return self.cell

    class Cell(Table):
        def __getitem__(self, item):
            cell_extra = self.snap.cell_extra
            if isinstance(item, str):
                if item in cell_extra.keys():
                    return cell_extra[item](self)
                else:
                    return self.table[item]
            elif isinstance(item, tuple):
                letter, unit = item
                return self.__getitem__(letter) / self.snap.unit[unit]
            else:
                return RamsesSnapshot.Cell(self.table[item], self.snap)


    def get_part(self, box=None, target_fields=None, domain_slicing=True, exact_box=True, cpulist=None):
        if(box is not None):
            self.box = box
        if(self.box is None or np.array_equal(self.box, default_box)):
            domain_slicing = False
            exact_box = False
        if(self.box is None or not np.array_equal(self.box, self.box_part)):
            if(cpulist is None):
                cpulist = self.get_involved_cpu()
            else:
                domain_slicing=False
                exact_box=False
            self.read_part(target_fields=target_fields, cpulist=cpulist)
            if(domain_slicing):
                part = domain_slice(self.part_data, cpulist, self.cpulist_part, self.bound_part)
            else:
                part = self.part_data

            if(self.box is not None):
                if(exact_box):
                    mask = box_mask(get_vector(part), self.box)
                    timer.start('Masking parts... %d / %d (%.4f)' % (np.sum(mask), mask.size, np.sum(mask)/mask.size), 1)
                    part = part[mask]
                    timer.record()

            part = self.Particle(part, self)
            self.box_part = self.box
            self.part = part
        return self.part

    class Particle(Table):
        def __getitem__(self, item):
            part_extra = self.snap.part_extra
            if isinstance(item, str):
                if item in part_extra.keys():
                    return part_extra[item](self)
                elif item in part_family.keys():
                    return RamsesSnapshot.Particle(classify_part(self.table, item), self.snap)
                elif item == 'smbh':
                    return RamsesSnapshot.Particle(find_smbh(self.table, 0), self.snap)
                else:
                    return self.table[item]
            elif isinstance(item, tuple):
                letter, unit = item
                return self.__getitem__(letter) / self.snap.unit[unit]
            else:
                return RamsesSnapshot.Particle(self.table[item], self.snap)

    def get_halos_cpulist(self, halos, radius=3., radius_name='rvir', n_divide=4):
        cpulist = []
        for halo in halos:
            box = get_box(get_vector(halo), halo[radius_name]*radius*2)
            cpulist.append(get_cpulist(box, None, self.levelmax, self.bound_key, self.ndim, n_divide))
        return np.unique(np.concatenate(cpulist))

    def diag(self):
        dm_tot = 0
        star_tot = 0
        gas_tot = 0
        smbh_tot = 0
        if(self.box is not None):
            volume = np.prod(self.box[:, 1]-self.box[:, 0]) / (self.unit['Mpc']/self.params['h'])**3 / self.params['aexp']**3
            print('Redshift (z) = %.5f' % (self.z))
            print('Comoving volume of the box: %.3e (Mpc/h)^3' % (volume))
        if(self.part is not None):
            part = self.part
            part = part[box_mask(get_vector(part), self.box)]
            print('---------------------------------')
            print('Total  number of particles: %d' % part.size)
            dm = part['dm']
            tracer = part['tracer']

            dm_tot = np.sum(dm['m', 'Msol'])
            dm_min = np.min(dm['m', 'Msol'])

            tracer_tot = np.sum(tracer['m', 'Msol'])
            tracer_min = np.min(tracer['m', 'Msol'])

            print('Number of     DM particles: %d with total mass of %.3e Msol, Min. particle mass: %.3e Msol' % (dm.size, dm_tot, dm_min))
            print('Number of tracer particles: %d with total mass of %.3e Msol, Min. particle mass: %.3e Msol' % (tracer.size, tracer_tot, tracer_min))

            if(self.params['star']):
                star = part['star']
                smbh = part['smbh']

                star_tot = np.sum(star['m', 'Msol'])
                star_min = np.min(star['m', 'Msol'])

                smbh_tot = np.sum(smbh['m', 'Msol'])
                smbh_max = np.max(smbh['m', 'Msol'])
                print('---------------------------------------------')

                print('Number of       star particles: %d with total mass of %.3e Msol, Min. particle mass: %.3e Msol' % (star.size, star_tot, star_min))
                print('Number of       SMBH particles: %d with total mass of %.3e Msol, Max. SMBH mass: %.3e Msol' % (smbh.size, smbh_tot, smbh_max))
                print('DM/Stellar mass ratio is %.3f' % (dm_tot / star_tot))

                star_den = star_tot/volume
                print('Stellar Mass density is %.3e Msol / (Mpc/h)^3' % (star_den))

        if(self.cell is not None):
            cell = self.cell
            cell = cell[box_mask(get_vector(cell), self.box, size=cell['dx'])]
            print('---------------------------------------------')
            print('Min. spatial resolution = %.4f pc (%.4f pc/h in comoving)' % (np.min(self.cell['dx', 'pc']), self.boxsize_comoving*1E3*0.5**np.min(self.cell['level'])))
            print('Total number of cells: %d' % cell.size)
            gas_tot = np.sum(cell['rho'] * (cell['dx'])**3) / self.unit['Msol']
            print('Total gas mass: %.3e Msol' % gas_tot)


        if(self.cell is not None and self.part is not None):
            print('Baryonic fraction: %.3f' % ((gas_tot+star_tot+smbh_tot) / (dm_tot+gas_tot+star_tot+smbh_tot)))

########### functions ###########


def ckey2idx(amr_keys, nocts, levelmin, ndim=3):
    idx = 0
    poss = []
    lvls = []
    for noct, leveladd in zip(nocts, np.arange(0, nocts.size)):
        ckey = amr_keys[idx : idx+noct]
        idx += noct
        ckey = np.repeat(ckey[:,:,np.newaxis], 2**ndim, axis=-1)
        suboct_ind = np.arange(2**ndim)
        nstride = 2**np.arange(0, ndim)

        suboct_ind, nstride = np.meshgrid(suboct_ind, nstride)

        cart_key = 2*ckey+np.mod(suboct_ind//nstride, 2) + 0.5
        level = levelmin+leveladd
        poss.append(cart_key/2**level)
        lvls.append(np.full(noct, level))
    poss = np.concatenate(poss)
    #poss = np.mod(poss-0.5, 1)
    lvls = np.concatenate(lvls)
    return poss, lvls

def cartesian(*arrays):
    # cartesian product of arrays
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def domain_slice(array, cpulist, cpulist_all, bound):
    # array should already been aligned with bound
    idxs = np.where(np.isin(cpulist_all, cpulist))[0]
    doms = np.stack([bound[idxs], bound[idxs+1]], axis=-1)
    segs = doms[:, 1] - doms[:, 0]

    out = np.empty(np.sum(segs), dtype=array.dtype) # same performance with np.concatenate
    now = 0
    for dom, seg in zip(doms, segs):
        out[now:now + seg] = array[dom[0]:dom[1]]
        now += seg

    return out


def bulk_sort(array):
    # Sorts the array cpu-wise, not used for now
    cpumap = array['cpu']
    idxs = compute_boundary(cpumap)
    counts = np.diff(idxs)
    idxs = idxs[:-1]
    cpulist = cpumap[idxs]

    key = np.argsort(cpulist)
    cpulist = cpulist[key]
    idxs = idxs[key]
    counts = counts[key]

    new = np.empty(array.size, dtype=array.dtype)
    now = 0
    bound_new = [0]
    for icpu, idx, count in zip(cpulist, idxs, counts):
        new[now:now+count] = array[idx:idx+count]
        now += count
        bound_new.append(now)

    return new, np.array(bound_new)

def compute_boundary(cpumap, cpulist):
    bound = np.searchsorted(cpumap, cpulist)
    return np.concatenate([bound, [cpumap.size]])
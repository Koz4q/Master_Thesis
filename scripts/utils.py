import h5py

def load_h5_file(filepath):
    return h5py.File(filepath, 'r')

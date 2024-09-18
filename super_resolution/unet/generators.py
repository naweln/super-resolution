
# data generators

import numpy as np
import h5py
import skimage.transform

class Generator_Simple:

    def __init__(self, fname_h5):

        self.fname_h5 = fname_h5
        self.shapes = (None, None)
        self.l_inds = None
        self.parse_h5()

    def parse_h5(self):
        with h5py.File(self.fname_h5, 'r') as h5_fh:
            
            l_pids = h5_fh['patientID'][()]
            
            self.pIDs   = np.unique(l_pids)
            self.l_inds = list(range(len(l_pids)))

            self.num_images = len(self.l_inds)
            
            shpx = list(h5_fh['recon_linear'].shape[1:])
            shpy = list(h5_fh['recon_multisegment'].shape[1:])

            shape_x = shpx + [1]
            shape_y = shpy + [1]
            self.shapes = (shape_x, shape_y)

    def __iter__(self):
        with h5py.File(self.fname_h5, 'r') as h5_fh:
            while True:
                inds_data = np.copy(self.l_inds)
                
                for i in inds_data:
                    x = h5_fh['recon_linear'][i,...][()]
                    y = h5_fh['recon_multisegment'][i,...][()]

                    x = np.expand_dims(x, -1)
                    y = np.expand_dims(y, -1)

                    sample = (x, y)

                    yield sample



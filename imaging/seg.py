import logging
from collections import OrderedDict

import numpy as np
import vtk
import SimpleITK as sitk

from sonador.imaging.orthanc.base import ImagingSeries
from sonador.imaging.orthanc.seg import ImageSegmentData

logger = logging.getLogger(__name__)


def dcmseg2labelmap(dcmseries: ImagingSeries, dcmseg: ImageSegmentData, dtype=np.uint8):
    ''' Package a DICOM segmentation file as a labelmap array
        
        @input dcmseries (ImagingSeries): medical imaging series for which the labelmap should be created
        @input dcmseg (ImageSegmentData): segment from which the labelmap should be created
        
        @returns numpy.ndarray labelmap matching the dimensions (slices, rows, columns) of the input DICOM series 
    '''
    mask = np.zeros(shape=dcmseries.shape, dtype=dtype)
    
    # Iterate through frames and copy mask layers to appropriate level of the volume
    for i in range(dcmseg.data.frames.shape[0]):
        
        # Retrieve the mask layer and frame description to rebuild volume mask
        lmask = dcmseg.data.frames[i,:,:]
        ldesc = dcmseg.data.frame_descriptions[i]
        
        # Determine position mapping for mask layers
        for plane in getattr(ldesc, 'PlanePositionSequence', []):
            
            # Image position patient includes x,y,z coordinates of lower-left corner of slice
            # match with the slice location of the source image to determine position in the stack
            if hasattr(plane, 'ImagePositionPatient'):
                if len(plane.ImagePositionPatient) == 3:
                    
                    # Locate the slice based on vertical positioning
                    for s in dcmseries.slices_collection:
                        
                        # Add the slice layer to the mask
                        if s.slice_location == plane.ImagePositionPatient[2]:
                            mask[s.series_index-1,:,:] = lmask
                            break
                            
                # Unsupported position vector
                else: raise ValueError('Unsupported position vector, unable to locate correct slice')        
                
    return mask

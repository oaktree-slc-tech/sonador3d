import logging
from collections import OrderedDict

import numpy as np
import vtk
import SimpleITK as sitk

from sonador.imaging.orthanc.base import ImagingSeries
from sonador.imaging.orthanc.seg import ImageSegmentData

logger = logging.getLogger(__name__)


def dcmseg2labelmap(dcmseries: ImagingSeries, dcmseg: ImageSegmentData, dtype=None, z_idx=2):
    ''' Package a DICOM segmentation file as a labelmap array
        
        @input dcmseries (ImagingSeries): medical imaging series for which the labelmap should be created
        @input dcmseg (ImageSegmentData): segment from which the labelmap should be created
        @input dtype (numpy.ndarray data type): data type which should be used for the segmentation arrays
        @input z_idx (int, default=2): z-position index to be used for parsing segmentation files 
            to a labelmap.
        
        @returns numpy.ndarray labelmap matching the dimensions (slices, rows, columns) of the input DICOM series 
    '''
    dtype = dtype or dcmseg.data.frames.dtype
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

                    # Z location of segmentation
                    seg_z = plane.ImagePositionPatient[z_idx]
                    
                    # Locate the slice based on vertical positioning
                    for s in dcmseries.slices_collection:

                        # Pixel spacing
                        spacing = s.pixel_spacing

                        # Z location of slice
                        slice_z = s.slice_location if s.slice_location \
                            else s.image_position_patient.z if s.image_position_patient \
                            else None

                        # Thickness of the slice: use "thickness" if available otherwise use
                        # twice the width of the scan
                        slice_thickness = spacing.thickness if spacing.thickness is not None \
                            else 2*spacing.x if spacing else None
                        
                        # Add the slice layer to the mask: check to ensure that spacing, slice z-position,
                        # segmentation z-position, and spacing are all available. A segmentation plane
                        # is considered a match if it:
                        # 1. slice z-position and segmenation z-position match exactly
                        # 2. slice z-position falls within the segmentation z-position +/- half the thickness
                        if spacing and slice_z is not None and seg_z is not None \
                            and slice_thickness is not None and (
                                slice_z == seg_z or seg_z - slice_thickness/2 <= slice_z <= seg_z + slice_thickness/2):
                            mask[s.series_index-1,:,:] = lmask
                            break
                            
                # Unsupported position vector
                else: raise ValueError('Unsupported position vector, unable to locate correct slice')        
                
    return mask

import numpy as np
import SimpleITK as sitk

from sonador.imaging.orthanc.base import ImageCoord


def imagevolume_center(img: sitk.Image):
	'''	return the physical center point of the provided image
	'''
	width, height, depth = img.GetSize()
	return ImageCoord(*img.TransformIndexToPhysicalPoint((
		int(np.ceil(width/2)),
        int(np.ceil(height/2)),
		int(np.ceil(depth/2)))))
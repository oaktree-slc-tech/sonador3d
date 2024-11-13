import os, tempfile, logging, functools
from collections import OrderedDict, namedtuple

import numpy as np
import vtk
import SimpleITK as sitk
import pyvista as pv

from sonador.imaging.orthanc import ImagingSeries
from sonador.imaging.orthanc.base import ImageCoord
from sonador.tasks.images import download_imaging_filearchive
from sonador.imaging.orthanc.seg import ImageSegmentData

from sonador3d_imaging.apisettings import NUETRAL_3D_POSE, LABELMAP_DTYPE_DEFAULT, IDENTIY_3D
from sonador3d_imaging.volume import SonadorImagingVolume
from sonador3d_imaging.seg import dcmseg2labelmap

from .helpers import imagevolume_center, init_empty_image, sitk2vtk

logger = logging.getLogger(__name__)


class PyVistaImagingVolume(object):
	'''	Helper object used for working with 3D medical imaging data retrieved from Sonador.
		"volume" property is a PyVista volume.
	'''
	def __init__(self, series, *args, **kwargs):
		self.series = series
		self._volume = kwargs.get('volume')
		self._reader = kwargs.get('reader')

		# Ensure that an imaging series was provided
		if not isinstance(self.series, ImagingSeries):
			raise ValueError('Unable to initialize imaging volume, invalid DICOM resource. '
				+ 'Volumes may only be initialized using ImagingSeries resources.')

	def fectch_volume(self, *args, **kwargs):
		'''	Retrieve the voxel data associated with the image and return a point cloud

			@returns SimpleITK.Image
		'''
		if not len(self.series.slices_collection):
			raise ValueError(('Unable to retrieve imaging data for series %s, series does not contain any '
				+ 'DICOM slices.') % self.series.pk)

		with tempfile.TemporaryDirectory() as tmp:

			# Download and unpack imaging pixel data from Orthanc
			download_imaging_filearchive(self.series, tmp, extract=True)
			ipath = os.path.join(tmp, self.series.pk, 'IMAGES')

			if self._reader is None:
				self._reader = pv.DICOMReader(ipath)

			# Parse imaging volume
			return self._reader.read()

	@property
	def volume(self):
		'''	Voxel data for the imaging series

			@returns sitk.Image
		'''
		if getattr(self, '_volume', None) is None:
			setattr(self, '_volume', self.fectch_volume())

		return self._volume
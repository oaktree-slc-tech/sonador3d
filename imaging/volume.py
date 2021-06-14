import os, tempfile, logging, functools
from collections import OrderedDict

import numpy as np
import vtk
import SimpleITK as sitk

from sonador.imaging.orthanc import ImagingSeries
from sonador.tasks.images import download_imaging_filearchive
from sonador.imaging.orthanc.seg import ImageSegmentData

from .seg import dcmseg2labelmap

logger = logging.getLogger(__name__)


LABELMAP_DTYPE_DEFAULT = np.uint8


class SonadorImagingVolume(object):
	'''	Helper object used for working with medical imaging volumes retrieved from Sonador
	'''
	def __init__(self, series, *args, **kwargs):
		self.series = series
		self._volume = kwargs.get('volume')

		# Ensure that an imaging series was provided
		if not isinstance(self.series, ImagingSeries):
			raise ValueError('Unable to initialize imaging volume, invalid DICOM resource. '
				+ 'Volumes may only be initialized using ImagingSeries resources.')

		# Check the 
		if self._volume and not isinstance(self._volume, sitk.SimpleITK.Image):
			raise ValueError('Invalid volume type. SonadorImagingVolume instances '
				+ 'may only be initilized from SimpleITK.Image instances.')

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

			dcm_names = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(ipath)
			return sitk.ReadImage(dcm_names, *args, **kwargs)

	@property
	def center(self):
		'''	Return the physical center point of the volume
		'''
		width, height, depth = self.volume.GetSize()
		return self.volume.TransformIndexToPhysicalPoint((
			int(np.ceil(width/2)),
         	int(np.ceil(height/2)),
            int(np.ceil(depth/2))))

	@property
	def volume(self):
		'''	Voxel data for the imaging series

			@returns sitk.Image
		'''
		if getattr(self, '_volume', None) is None:
			setattr(self, '_volume', self.fectch_volume())

		return self._volume

	def labelmap_arrays(self, segments=None, dtype=LABELMAP_DTYPE_DEFAULT):
		'''	Create labelmap arrays from image segments for the image

			@input segments: iterable of segments (ImageSegmentData) to be included in the labelmap
				(default=segmentations associated with the imaging series)

			returns OrderedDict of labelmaps (numpy.ndarray) indexed to the segment metadata
		'''
		# Retrieve segments from Sonador segmentations (DICOM-SEG)
		if segments is None:
			segments = []
			for dcmseg in self.series.segmentations:
				segments.extend([s for s in dcmseg.segments])

		# Package segments as NumPy arrays
		return OrderedDict(
			(seg.meta, dcmseg2labelmap(self.series, seg, dtype=dtype)) for seg in segments)

	def labelmaps(self, segments=None, dtype=LABELMAP_DTYPE_DEFAULT):
		'''	Create a labelmap from image segments for the image. (Calls labelmap_arrays
				to package the labelmap volumes.)
			
			@returns OrderedDict of labelmaps (sitk.Image) indexed to the segment metadata
		'''
		labelmaps = OrderedDict()

		for smeta, sdata in self.labelmap_arrays(segments=segments, dtype=dtype).items():
			labelmaps[smeta] = sitk.GetImageFromArray(sdata[::-1])

		return labelmaps

	@property
	def volume_array(self):
		'''	numpy.ndarray representation of the imaging voxel data
		'''
		return sitk.GetArrayFromImage(self.volume)

	@functools.cached_property
	def transform_matrix(self):
		'''	Transformation matrix for the image series calculated from the ImageOrientationPatient
			headers of the series.
		'''
		o = self.series.image_orientation_patient

		# Package as x, y, z components
		x = np.array([o.row.x, o.row.y, o.row.z])
		y = np.array([o.col.x, o.col.y, o.col.z])
		z = np.cross(x, y)

		return np.array([x, y, z]).T

	def resample(self, transform, default_value=0, interpolator=sitk.sitkLinear, inplace=False):
		'''	Apply an image transform transform and resample the volume.

			@input transform (SimpleITK.Transform): SimpleITK transform to be performed.

			@returns SonadorImagingVolume with the transform applied
		'''
		volume = sitk.Resample(self.volume, self.volume, transform, interpolator, default_value)

		# Perform the transform inplace and update the internal volume
		if inplace:
			self._volume = volume
			return self

		# Return a new imaging volume with the changes applied
		return SonadorImagingVolume(self.series, volume=volume)

	def rotate(self, x_deg, y_deg, z_deg, center=None, transform_matrix=None, inplace=False):
		'''	Rotate the volume the specified number of degrees around the x, y, z axis

			@returns SonadorImagingVolume with a copy of the data
		'''
		# Set the center of rotation
		if center is None:
			center = self.center

		# Convert the degrees provided to radians
		theta_x = np.deg2rad(x_deg)
		theta_y = np.deg2rad(y_deg)
		theta_z = np.deg2rad(z_deg)

		# Initialize and set properties of the transform
		transform = sitk.Euler3DTransform()
		transform.SetCenter(center)
		
		if transform_matrix:
			transform.SetMatrix(transform_matrix)
		
		transform.SetRotation(theta_x, theta_y, theta_z)
		return self.resample(transform, inplace=inplace)


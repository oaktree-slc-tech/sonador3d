import os, tempfile, logging, functools
from collections import OrderedDict, namedtuple

import numpy as np
import vtk
import SimpleITK as sitk

from sonador.imaging.orthanc import ImagingSeries
from sonador.imaging.orthanc.base import ImageCoord
from sonador.tasks.images import download_imaging_filearchive
from sonador.imaging.orthanc.seg import ImageSegmentData

from .seg import dcmseg2labelmap

logger = logging.getLogger(__name__)


LABELMAP_DTYPE_DEFAULT = np.uint8


IDENTIY_3D = np.array([
	[1, 0, 0], 
	[0, 1, 0],
	[0, 1, 1]
])


class SonadorImagingVolume(object):
	'''	Helper object used for working with 3D medical imaging volumes retrieved from Sonador
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

			@returns ImageCoord
		'''
		width, height, depth = self.volume.GetSize()
		return ImageCoord(*self.volume.TransformIndexToPhysicalPoint((
			int(np.ceil(width/2)),
         	int(np.ceil(height/2)),
            int(np.ceil(depth/2)))))

	@property
	def origin(self):
		'''	Retrieve the origin point of the volume coordinate system.

			@returns ImageCoord
		'''
		return ImageCoord(*self.volume.GetOrigin())

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
	def rotation_matrix(self):
		'''	Rotation matrix for the image series calculated from the ImageOrientationPatient
			headers of the series.

			@returns numpy.Array
		'''
		o = self.series.image_orientation_patient

		# Package as x, y, z components
		x = np.array([o.row.x, o.row.y, o.row.z])
		y = np.array([o.col.x, o.col.y, o.col.z])
		z = np.cross(x, y)

		return np.array([x, y, z]).T

	def create_transform_bounding_volume(self, transform, volume=None, output_direction=IDENTIY_3D.flatten()):
		'''	EXPERIMENTAL: Compute the bounds of a resampling grid for ITK transformations in order to determine
			the size of an output image.

			1. Find the phyiscal points of the current bounding shape
			2. Create an inverse transform and apply the transform to the existing points
			3. Calculate the minimum and maximum of the transformed points for x, y, z
			4. Compute the grid size based on the physical size and spacing

			@returns simpleITK.Image
		'''
		volume = volume or self.volume

		# 1. Find the physical points of the current bounding shape
		bounding_shape = [
			volume.TransformIndexToPhysicalPoint([0, 0, 0]),
			volume.TransformIndexToPhysicalPoint([0, 0, self.volume.GetDepth()]),
			volume.TransformIndexToPhysicalPoint([self.volume.GetWidth(), 0, 0]),
			volume.TransformIndexToPhysicalPoint([self.volume.GetWidth(), 0, self.volume.GetDepth()]),
			volume.TransformIndexToPhysicalPoint([0, self.volume.GetHeight(), 0]),
			volume.TransformIndexToPhysicalPoint([0, self.volume.GetHeight(), self.volume.GetDepth()]),
			volume.TransformIndexToPhysicalPoint([self.volume.GetWidth(), self.volume.GetHeight(), 0]),
			volume.TransformIndexToPhysicalPoint([
				self.volume.GetWidth(), self.volume.GetHeight(), self.volume.GetDepth()]),
		]

		# 2. Create an inverse transfrm and apply the transform to the existing points
		tx_inverse = transform.GetInverse()
		bounding_tx = [tx_inverse.TransformPoint(p) for p in bounding_shape]

		# 3. Calculate the minimum and maximum of the transform points for each of of the dimensions
		min_x = min(bounding_shape)[0]
		min_y = min(bounding_shape, key=lambda p: p[1])[1]
		min_z = min(bounding_shape, key=lambda p: p[2])[2]
		max_x = max(bounding_shape)[0]
		max_y = max(bounding_shape, key=lambda p: p[1])[1]
		max_z = max(bounding_shape, key=lambda p: p[2])[2]

		# Spacing, direction, origin
		output_spacing = self.volume.GetSpacing()

		# Calculate the output size
		output_size = [
			int((max_x-min_x)/output_spacing[0]), 
			int((max_y-min_y)/output_spacing[1]), 
			int((max_z-min_z)/output_spacing[2])
		]

		# Output origin
		output_origin = [min_x, min_y, min_z]

		# Create output image
		output_image = sitk.Image(output_size[0], output_size[1], output_size[2], volume.GetPixelIDValue())
		output_image.SetOrigin(output_origin)
		output_image.SetSpacing(output_spacing)
		output_image.SetDirection(np.array(output_direction, dtype=np.double).flatten())

		return output_image

	def resample(self, transform, output_volume=None, default_value=0, interpolator=sitk.sitkLinear, inplace=False):
		'''	Apply an image transform transform and resample the volume.

			@input transform (SimpleITK.Transform): SimpleITK transform to be performed.
			@input output_volume (SimpleITK.Image, default=self.volume): Volume which should be used
				to bound the data after the transform has been applied.
			@input inplace (bool, default=False): Toggles whether the transform should return a 
				new SimpleITK.Image instance and new SonadorImagingVolume or should
				be applied to the current instance.

			@returns SonadorImagingVolume with the transform applied
		'''
		# Apply the transform and resample to a new output volume
		if output_volume:
			volume = sitk.Resample(self.volume, output_volume, transform, interpolator, default_value)
		
		# If an output size is not provided, use the current volume for the bounding shape
		else:
			volume = sitk.Resample(self.volume, self.volume, transform, interpolator, default_value)

		# Perform the transform inplace and update the internal volume
		if inplace:
			self._volume = volume
			return self

		# Return a new imaging volume with the changes applied
		return SonadorImagingVolume(self.series, volume=volume)

	def rotate(self, x_deg, y_deg, z_deg, center=None, inplace=False):
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
		transform.SetRotation(theta_x, theta_y, theta_z)

		return self.resample(transform, inplace=inplace)

	def transform(self, transform_matrix, translation_matrix=None, center=None, inplace=False):
		'''	EXPERIMENTAL: Perform an affine transform defined by the transformation matrix

			@returns SonadorImagingVolume with a copy of the data
		'''
		# Set the center of rotation
		if center is None:
			center = self.center

		# Initialize and set properties of the transform
		transform = sitk.AffineTransform(self.volume.GetDimension())
		transform.SetCenter(center)

		# Set rotation matrix and translation components of the transform
		transform.SetMatrix(transform_matrix)
		if translation_matrix is not None:
			transform.SetTranslation(translation_matrix)

		return self.resample(transform, 
			output_volume=self.create_transform_bounding_volume(transform), inplace=inplace)

	def translate(self, offset, inplace=False):
		'''	Translate the volume based on the provided offset

			@returns SonadorImagingVolume with a copy of the data
		'''
		transform = sitk.TranslationTransform(self.volume.GetDimension(), offset)
		return self.resample(transform, inplace=inplace)

	

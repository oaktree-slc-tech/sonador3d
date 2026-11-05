import numpy as np
import SimpleITK as sitk

import vtk
from vtk.util import numpy_support

from sonador.imaging.orthanc.base import ImageCoord
from sonador3d_imaging.apisettings import DCM_VOLUME_LEFT_POSTERIOR_SUPERIOR, \
	DCM_VOLUME_RIGHT_ANTERIOR_INFERIOR, DCM_VOLUME_RIGHT_ANTERIOR_SUPERIOR
from sonador3d_imaging.volume import SonadorImagingVolume
from sonador3d_imaging.helpers import imagevolume_center, init_empty_image, sitk_image, \
	DisplayImageGrid, SourceSlice, display_image_grid, image_grid, antialias_sigma, resample_display_image, \
	image_orientation_patient, image_slice_normal, image_slice_positions, \
	imaging_volume_source_slices, source_slice_pitch, source_slice_references, copy_private_blocks, \
	display_volume_size_label, display_volume_series_number, display_volume_series_description, \
	display_volume_derivation_description, \
	DisplayPixelRepresentation, image_intensity_range, display_pixel_representation, stored_pixel_container, stored_pixel_values, \
	lossy_compression_history, display_compression_transfer_syntax, display_compression_backends, \
	supported_display_compressions, check_compression_sop_class, check_compression_representation, encode_display_pixel_data, \
	dcm_encode_display_volume, dcm_encode_display_volumes

from ..spatial.helpers import tx_matrix


def sitk2vtk(img: sitk.Image, source_label='SimpleITK', sitk_origin=True, sitk_direction=True):
	'''	Convert a SimpleITK image to a VTK image (via numpy). Adapted from sitk2vtk from
		David T. Refer to https://github.com/dave3d/dicom2stl/blob/main/utils/sitk2vtk.py.

		@input sitk_origin (bool, default=True): copy SimpleITK origin
		@input sitk_direction (bool, default=True): copy SimpleITK direction vector
	'''
	size = list(img.GetSize())
	origin = list(img.GetOrigin())
	spacing = list(img.GetSpacing())
	components_n = img.GetNumberOfComponentsPerPixel()
	direction = img.GetDirection()

	# Convert SimpleITK image to a numpy array
	img_array = sitk.GetArrayFromImage(img)

	# Create VTK image data
	img_vtk = vtk.vtkImageData()
	
	# VTK images are three dimensional and treated as stacks. If a 2D image was provided, specify that there is
	# only a single image in the stack, add an additional value to the origin, and append a 0 spacing parameter.
	if len(size) == 2:
		size.append(1)
	if len(origin) == 2:
		origin.append(0.0)
	if len(spacing) == 2:
		spacing.append(spacing[0])

	# Update the direction vector for a 3D image
	if len(direction) == 4:
		direction = [ direction[0], direction[1], 0.0, direction[2], direction[3], 0.0, 0.0, 0.0, 1.0,]

	# Set dimensions, spacing, origin, extent, and direction of VTK image
	img_vtk.SetDimensions(size)
	img_vtk.SetSpacing(spacing)
	if sitk_origin:
		img_vtk.SetOrigin(origin)
	if sitk_direction:
		img_vtk.SetDirectionMatrix(direction)
	img_vtk.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

	# Convert pixel data to vtkImage	
	depth_array = numpy_support.numpy_to_vtk(img_array.ravel())
	depth_array.SetNumberOfComponents(components_n)
	depth_array.SetName(source_label)

	# Add the array to the image and mark as active
	img_vtk.GetPointData().SetScalars(depth_array)
	img_vtk.GetPointData().SetActiveScalars(source_label)

	# Mark VTK image as modified
	img_vtk.Modified()
	return img_vtk


def init_display_imagevolume(sx,
		volume_anatomic_orientation_default=DCM_VOLUME_LEFT_POSTERIOR_SUPERIOR, vol_direction=False,
		vol_dcm_attr='vol_dcm', vol_vtk_attr='vol_vtk', tx_inv_attr='tx_inv'):
	'''	Initialize an imaging volume that is intended to be used for visualization.

		There may be a mismatch in display settings between medical imaging, which often uses an 
		"right-anterior-inferior" or "right-anterior-superior" coordinate axis for volumetric data 
		and robotics systems, which favor a "left-posterior-superior" orientation.

		This method initializes a SonadorImagingVolume from the provided series, adds properties 
		for a "DICOM volume" (`vol_dcm`) with the desired anatomic orientation, a VTK version of the image
		data (`vol_vtk), and the reverse transform matrix so that it is possible to place other objects
		such as points and meshes within the same coordinate system as the imaging volume.

		@input sx (sonador.imaging.orthanc.ImagingSeries): imaging series for which the display volume
			should be initialized.
		@input volume_anatomic_orientation_default (default='LPS'): volume orientation to use if the 
			imaging volume instance does not include a display orientation.
		@input vol_direction (bool, default=False): copy SimpleITK direction vector to VTK (display) volume

		@returns SonadorImagingVolume: imaging volume instance with additional properties
			@attr vol_dcm (SimpleITK.Image): image instance which has been transformed to the specified
				volume anatomic orientation
			@attr vol_vtk (vtk.vtkImageData): display volume to be used in visualization
			@attr tx_inv (numpy.Array): numpy array containing the reverse transform for the display volume
	'''
	# Initialize imaging volume
	ivolume = SonadorImagingVolume(sx)

	# Create anatomic orientation, display volume, and inverse transform
	setattr(ivolume, vol_dcm_attr, sitk.DICOMOrient(
		ivolume.volume, getattr(sx, 'volume_anatomic_orientation', volume_anatomic_orientation_default)))
	setattr(ivolume, vol_vtk_attr, sitk2vtk(ivolume.vol_dcm, sitk_direction=vol_direction))
	setattr(ivolume, tx_inv_attr, tx_matrix(
		np.linalg.inv(np.array(ivolume.vol_dcm.GetDirection()).reshape((3,3))),
		ivolume.vol_dcm.GetOrigin(),
	))

	return ivolume

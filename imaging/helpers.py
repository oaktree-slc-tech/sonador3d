import numpy as np
import SimpleITK as sitk

import vtk
from vtk.util import numpy_support

from sonador.imaging.orthanc.base import ImageCoord


def imagevolume_center(img: sitk.Image):
	'''	return the physical center point of the provided image
	'''
	width, height, depth = img.GetSize()
	return ImageCoord(*img.TransformIndexToPhysicalPoint((
		int(np.ceil(width/2)),
        int(np.ceil(height/2)),
		int(np.ceil(depth/2)))))


def init_empty_image(img: sitk.Image, direction=None, origin=None, spacing=None, dtype=None):
	'''	Create an empty SimpleITK.Image instance with the same parameters as the provided input.
	'''
	# Retrieve size, direction, and origin
	isize = img.GetSize()
	direction = direction or img.GetDirection()
	origin = origin or img.GetOrigin()
	spacing = spacing or img.GetSpacing()

	# Initialize image
	empty_img = sitk.Image(isize[0], isize[1], isize[2], dtype or img.GetPixelIDValue())
	empty_img.SetDirection(direction)
	empty_img.SetOrigin(origin)
	empty_img.SetSpacing(spacing)

	return empty_img


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
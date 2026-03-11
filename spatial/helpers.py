import logging

import vtk
from vtk.util import numpy_support

import numpy as np
import SimpleITK as sitk
import vtk
import pyvista as pv

from sonador.imaging.orthanc.base import ImageCoord
from skimage.draw import polygon2mask

logger = logging.getLogger(__name__)


def slice2polypoints(poly, ordered_points=None, simplify=True, inplace=False):
	'''	Parse the provided intersection/slice to an ordered list of points
		to create a polygon/contour.

		@input poly (pyvista.PolyData): Shape to translate to points

		@returns iterable of points
	'''
	if not isinstance(poly, pv.PolyData):
		raise ValueError('slice2polypoints requires a valid pyvista.PolyData object as an input.')
	
	ordered_points = ordered_points or []

	# Create copy of the poly data to prevent any changes to the source
	poly = poly.copy()

	# Simplify geometry before beginning traversal
	if simplify:
		poly_stripper = vtk.vtkStripper()
		poly_stripper.SetInputData(poly)
		poly_stripper.Update()
		poly_ = poly_stripper.GetOutput()

	# Use source geometry 
	else: poly_ = poly

	# Create container for point indices
	source_ids = vtk.vtkIdList()

	# Retrieve references to points and lines
	source_points = poly_.GetPoints()
	source_lines = poly_.GetLines()

	# Initialize traversal across all source lines
	source_lines.InitTraversal()
	while source_lines.GetNextCell(source_ids):

		# Retrieve the number of points in the line segment
		line_pcount = source_ids.GetNumberOfIds()
		
		# Iterate through point indexes, retrieve points, and add to list
		for p_idx in range(line_pcount):

			# Retrieve point
			s_idx = source_ids.GetId(p_idx)
			ordered_points.append(ImageCoord(*source_points.GetPoint(s_idx)))

	return ordered_points


def tx_matrix(rotation_matrix, p0=None):
	''' Create a 4x4 transform matrix from the provided image and point.
	'''
	# Create 4x4 transform matrix, retrieve rotation component
	Mtx = np.identity(4)
	Mtx[:3,:3] = rotation_matrix

	if p0:
		# Translation component: determine the translation vector by taking
		# the difference of the provided point in the original coordinate system
		# and the point after rotation.
		vtx = ImageCoord(
			*np.subtract(p0, rotation_matrix.dot(p0)))
		Mtx[0,3] = vtx.x
		Mtx[1,3] = vtx.y
		Mtx[2,3] = vtx.z

	return Mtx


IMG2MESH_ALGORITHMS = {
	'flying_edges': vtk.vtkDiscreteFlyingEdges3D,
	'marching_cubes': vtk.vtkDiscreteMarchingCubes
}


def labelimg2mesh(img: sitk.Image, sitk_origin=True, sitk_direction=True, algorithm='flying_edges',
		generate_values=(1,1,1)):
	'''	Create a mesh from the provided label image. (The algorithm used for creating the image is controlled
		using via the "algorithm" parameter.)

		@input img (SimpleITK.Image): label image which should be used to create the mesh.
		@input sitk_origin (bool, default=True): when True the SimpleITK origin is preserved in the
			output mesh geometry.
		@input sitk_origin (bool, default=True): when True, the SimpleITK image direction is preserved in the
			output mesh geometry. If False, the mesh will be oriented relative to a neutral geometry.

		@returns pv.PolyData
	'''
	from ..imaging.helpers import sitk2vtk

	# Ensure that the requested algorithm is supported
	if not algorithm in IMG2MESH_ALGORITHMS:
		raise ValueError('Invalid algorithm "%s". Supported: %s'% (
			algorithm, ", ".join(IMG2MESH_ALGORITHMS.keys()))
		)

	# Convert SimpleITK image to vtkImageData (required to use VTK algorithm implementations)
	img_vtk = sitk2vtk(img, sitk_origin=sitk_origin, sitk_direction=sitk_direction)

	# Initialize surface extractor
	img_extractor = IMG2MESH_ALGORITHMS.get(algorithm)()
	img_extractor.SetInputData(img_vtk)
	img_extractor.GenerateValues(*generate_values)
	img_extractor.Update()

	return pv.PolyData(img_extractor.GetOutput())


AXES = {
	0: {'label': 'x', 'normal': 'y'},
	1: {'label': 'y', 'normal': 'z'},
	2: {'label': 'z', 'normal': 'x'},
}


def mesh2contours(poly, img: sitk.Image, *args, axis=2, **kwargs):
	'''	Create contoured slices from the mesh using the provided image volume.

		@input poly (pyvista.PolyData): mesh to be sliced
		@input img (SimpleITK.Image): volume associated with the mesh to be used for spatial coordinates
			and to determine along which axis the mesh should be sliced.
		@input axis (int, default=2): axis through which the mesh should be sliced
			(x=0, y=1, z=2).
	'''
	# Containers for contours, origins, and centers
	contours = kwargs.get('contours', [])
	cutting_planes = kwargs.get('cutting_planes', { 'origins': [], 'centers': [], '_init': False })

	# Iterate through the slices and aggregate the points which define the cutting plane
	if not cutting_planes.get('_init'):
		isize = img.GetSize()

		# Collect origins
		for i in range(isize[axis]):

			# Retrieve coordinates for the "origin" of the plane
			if axis == 0:
				o_ijk = [i,0,0]
				c_ijk = [i, int(isize[1]/2), int(isize[2]/2)]
			elif axis == 1:
				o_ijk = [0,i,0]
				c_ijk = [int(isize[0]/2), i, int(isize[2]/2)]
			elif axis == 2:
				o_ijk = [0,0,i]
				c_ijk = [int(isize[0]/2), int(isize[1]/2), i]

			o = ImageCoord(*img.TransformIndexToPhysicalPoint(o_ijk))
			c = ImageCoord(*img.TransformIndexToPhysicalPoint(c_ijk))

			# Collect origins and centers
			cutting_planes['origins'].append(o)
			cutting_planes['centers'].append(c)

		cutting_planes['_init'] = True

	# Slice mesh to extract contours
	if not len(cutting_planes['origins']) == len(cutting_planes['centers']):
		raise ValueError(
			'Invalid cutting planes. The number of provided origin points must match the number of provided centers.')

	for i,c in enumerate(cutting_planes['centers']):
		contours.append(poly.slice(origin=c, normal=AXES.get(axis)['normal'], 
			generate_triangles=kwargs.get('generate_triangles', True)))

	return contours


def mesh2labelmap(poly, img, axis=2, mask_array=None, mask_array_dtype=bool, 
		sitk_origin=True, sitk_direction=True, sitk_dtype=np.uint8, **kwargs):
	'''	Convert the provided poly to a labelmap image. This method slices the provided polydata
		using the image coordinate system as a guide and then creates

		@input poly (pyvista.PolyData): mesh to be converted to an image volume
		@input img (SimpleITK.Image): volume associated with the mesh. Used for slicing and spacing.
		@input axis (int, default=2): axis through which the mesh should be slided (x=0, y=1, z=2)

		@returns SimpleITK.Image sharing the same coordinate system as img
	'''
	# Create array container (numpy). The container array does not contain
	# spacing, direction/pose, or origin. Those will be copied from the source
	# imaging later in the method.
	mask_array = mask_array or np.zeros(shape=img.GetSize()[::-1], dtype=mask_array_dtype)

	# Slice mesh to contours
	contours = mesh2contours(poly, img, axis=axis, **kwargs)

	# Iterate through contours, generate slice, and unpack slice into volume array.
	# Take into account which axis is being un-packed.
	for i,g in enumerate(contours):

		def _slice_plane(_p):
			'''	Retrieve the size elements of the source image corresponding
				to the axis along which the labelmap is being sliced.
			'''
			if axis == 0:
				return _p[1:]
			if axis == 1:
				return _p[0],_p[2]
			elif axis == 2:
				return _p[:2]

			raise ValueError('Unable to retrieve points, invalid axis=%s' % axis)

		# Create a mask for pixels that fall within the bonds of the polygon
		if len(g.points):

			# polygon2mask takes the desired image size as an input alongside an ordered list of 
			# of the polygon points
			mask = polygon2mask(_slice_plane(img.GetSize()),
				[_slice_plane(img.TransformPhysicalPointToIndex(p))[::-1] for p in slice2polypoints(g)])

		# Create a fallback empty array for slices without contours
		else:
			mask = np.zeros(_slice_plane(img.GetSize()))

		# Add layer mask to container
		if axis == 0:
			mask_array[:,:,i] = mask
		elif axis == 1:
			mask_array[:,i,:] = mask
		elif axis == 2:
			mask_array[i,:,:] = mask
		else:
			raise ValueError('Unable to add slice labelmap to volume, invalid axis=%s' % axis)

	# Convert numpy array to SimpleITK.Image
	mask_img = sitk.GetImageFromArray(mask_array.astype(sitk_dtype))
	mask_img.SetSpacing(img.GetSpacing())

	# Set origin and direction
	if sitk_origin:
		mask_img.SetOrigin(img.GetOrigin())
	if sitk_direction:
		mask_img.SetDirection(img.GetDirection())

	return mask_img
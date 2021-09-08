import vtk
import pyvista as pv

from sonador.imaging.orthanc.base import ImageCoord


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


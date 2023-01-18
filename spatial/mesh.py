import os, logging, tempfile, mimetypes

import pyvista as pv
import meshio

from client.utils.object import pick, omit

from sonador.apisettings import DCMHEADER_ENCAPSULATED_DOCUMENT, \
	 DCMHEADER_ENCAPSULATED_DOCUMENT_MIMETYPE
from sonador.imaging.orthanc.base import ImagingSeries
from sonador.imaging.orthanc.m3d import DcmM3DSeries
from sonador.helpers import dcm_mimetype_guess_extension

from sonador.tasks.images import download_imaging_filearchive

logger = logging.getLogger(__name__)


class SonadorMesh(object):
	'''	Helper object used for working with 
	'''
	def __init__(self, series, *args, **kwargs):
		self.series = series

		# Ensure that an M3D series was provided
		if not isinstance(self.series, DcmM3DSeries):
			raise ValueError('Unabl to initialize mesh instance, invalid DICOM resource. '
				+ 'Mesh instances may only be initialized using DcmM3DSeries resources.')

	def parse_dcm_geometry(self, dcm, tmp=None, cleanup=True, *args, **kwargs):
		'''	Parse a DICOM mesh file to a PyVista geometry.

			@input dcm: DICOM file to be parsed
			@input tmp (tempfile.TemporaryDirectory): reference to a temporary directory
				to be used for file IO. If no tmp reference is provided, one will be created.
			@input cleanup (bool, default=True): toggles whether the `cleanup` method of
				tmp should be invoked after the mesh has been parsed from the DICOM file.

			@returns pyvisa.core.pointset.PolyData
		'''
		# Initialize temporary directory if one is not provided
		tmp = tmp or tempfile.TemporaryDirectory()

		try:

			# Retrieve DICOM file and determine file extension from the mimetype.
			# VTK (and by extension PyVista) requires a local file instance to parse
			# and uses file extensions to determine the data type.
			dcmfile = dcm.dcmfile(cache=kwargs.get('cache', False))		
			dcm_mtype = getattr(dcmfile, DCMHEADER_ENCAPSULATED_DOCUMENT_MIMETYPE)
			dcm_fext = mimetypes.guess_extension(dcm_mtype) or dcm_mimetype_guess_extension(dcm_mtype)

			# Ensure that the data type is supported by PyVista/Sonador 3D.
			if not dcm_fext:
				raise ValueError(
					'Unable to load data for M3D file %s, unsupported file mimetype: %s' % dcm_mtype)

			# Write file data to the local disk
			mesh_fpath = os.path.join(tmp, '%s%s' % (dcm.pk, dcm_fext))
			with open(mesh_fpath, 'wb') as f:
				f.write(getattr(dcmfile, DCMHEADER_ENCAPSULATED_DOCUMENT))

			# Parse file data with PyVista
			mesh = pv.read(mesh_fpath, *args, **omit(kwargs, ('cache',)))
			return mesh

		except Exception as err:
			raise err

		# Ensure that temporary files are cleaned up if an error was encountered
		# while downloading and attempting to load.
		finally:
			if cleanup:
				tmp.cleanup()

	def fetch_mesh_data(self, *args, ignore_errors=False, **kwargs):
		''' Retrieve mesh instances associated with the series
		'''
		if not len(self.series.instances_collection):
			raise ValueError('Unable to retrieve spatial data for series %s, no DICOM instances.')
		m3d_meshes = []

		with tempfile.TemporaryDirectory() as tmp:			

			# Download and unpack DICOM data from Orthanc
			for dcm in self.series.instances_collection:

				# Parse DICOM spatial data
				try: m3d_meshes.append(self.parse_dcm_geometry(dcm, *args, tmp=tmp, cleanup=False, **kwargs))
				
				# Unable to parse due to an error: log and (maybe) skip
				except Exception as err:
					logger.error('Unable to load data from DICOM instance %s due to an error. Error:\n%s'
						% (dcm.pk, err))
					if not ignore_errors:
						raise err

		return m3d_meshes

	@property
	def geometries(self):
		''' PyVista representation of the spatial data associated with the mesh series.
			Cached property.

			@returns iterable of pyvista.core.pointset.PolyData
		'''
		if getattr(self, '_geom', None) is None:
			setattr(self, '_geom', self.fetch_mesh_data())

		return self._geom

import os, itertools, logging, posixpath, yaml, copy, datetime, tempfile
from collections import OrderedDict
from io import BytesIO

from client.utils.general import first
from client.utils.urls import validate_url
from client.utils.conversion import str2bool
from client.utils.object import pick
from client.errors import ClientOperationError

from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset, FileDataset
from highdicom.color import CIELabColor

# Sonador headers
from sonador.apisettings import DicomMetaKey, DicomMeta, \
	IMAGING_SERVER_RESOURCE_IMAGE, IMAGING_SERVER_RESOURCE_SERIES, \
	DCMHEADER_PATIENT_ID, DCMHEADER_PATIENT_NAME, \
		DCMHEADER_PATIENT_SEX, DCMHEADER_PATIENT_WEIGHT, DCMHEADER_PATIENT_BIRTHDATE, \
	DCMHEADER_STUDY_INSTANCE_UID, DCMHEADER_STUDY_ID, DCMHEADER_STUDY_DATE, DCMHEADER_STUDY_TIME, \
		DCMHEADER_STUDY_DESCRIPTION, DCMHEADER_REFERRING_PHYSICIAN, \
	DCMHEADER_SOP_CLASS_UID, DCMHEADER_SERIES_INSTANCE_UID, DCMHEADER_SERIES_DATE, DCMHEADER_SERIES_TIME, \
		DCMHEADER_SERIES_DESCRIPTION, DCMHEADER_SERIES_NUMBER, DCMHEADER_MODALITY, DCM_MODALITY_MR, DCM_MODALITY_M3D, \
		DCMHEADER_FRAME_OF_REFERENCE_UID, DCMHEADER_PATIENT_POSITION, DCMHEADER_IMAGE_POSITION_PATIENT, \
		DCMHEADER_SLICE_THICKNESS, DCMHEADER_IMAGE_ORIENTATION_PATIENT, \
		DCMHEADER_SR_SOP_CLASS_UID, \
	DCMHEADER_CLINICAL_TRIAL_SERIES_ID, DCMHEADER_CLINICAL_TRIAL_PROTOCOLID, DCMHEADER_CLINICAL_TRIAL_SUBJECTID, \
		DCMHEADER_CLINICAL_TRIAL_SPONSOR, DCMHEADER_CLINICAL_TRIAL_COORDINATING_CENTER_NAME, \
		DCMHEADER_CLINICAL_TRIAL_TIMEPOINT_ID, \
	DCMHEADER_SOP_INSTANCE_UID, DCMHEADER_INSTANCE_NUMBER, DCMHEADER_PRIMARY_ANATOMIC_STRUCTURE_SEQUENCE, \
		DCMHEADER_CONTENT_DATE, DCMHEADER_CONTENT_TIME, DCMHEADER_SR_REF_SERIES_SEQ, \
	DCMHEADER_ENCAPSULATED_DOCUMENT_TITLE, DCMHEADER_CONTENT_DESCRIPTION
from sonador.apisettings.media import DCMHEADER_RECOMMENDED_DISPLAY_COLOR_VALUE,\
	DCM_MEDIA_PATIENT_TAGS, DCM_MEDIA_STUDY_TAGS, \
	DCM_MEDIA_CLINICAL_TRIAL_TAGS, DCMHEADER_SEGMENT_SEQUENCE, \
	DCMHEADER_SEGMENT_NUMBER, DCMHEADER_SEGMENT_LABEL

from sonador.apisettings import DCM_DATE_STRFORMAT, DCM_TIME_STRFORMAT
from sonador.apisettings.media import DCMEDIA_ENCAPSULATED_STL_SOP_CLASS_MRI
from sonador.helpers import srcode2dataset, dcm_encode_series_ref
from sonador.helpers.color import hex2rgb, rgb2cielab, dcm2hex
from sonador.tasks.media import dcm_encode_stl
from sonador.imaging.orthanc.seg import DcmSegmentationSeries

from .imaging import SonadorImagingVolume
from .spatial.helpers import labelimg2mesh

logger = logging.getLogger(__name__)


def dcm_encode_seg2m3d(sx_seg: DcmSegmentationSeries, sx_ref=None, m3d_series_uid=None, m3d_series_number=100, 
		ref_sopclass_uid=None, m3d_series_headers=None, m3d_instance_headers=None,
		stl_headers=[v for v in itertools.chain(DCM_MEDIA_PATIENT_TAGS, DCM_MEDIA_STUDY_TAGS, DCM_MEDIA_CLINICAL_TRIAL_TAGS)],
		stl_instance_numbers=None, stl_file_colors=None, post_mesh_create_callback=None, presave_callback=None, dry_run=False, hcache=None,
		**kwargs):
	'''	Create a mesh series from the provided segmentation series. Each segmentation layer
		will become a separate instance. Anatomic labels and display settings (such as color)
		will be preserved from the segmentation. The models will be added to the same study
		and, unless an override is provided, the imaging series from which the segmentation
		series was taken will be linked as the parent. Mesh generation is delegated to labelimg2mesh.

		@input sx_seg (sonador.imaging.orthanc.seg.DcmSegmentationSeries): Segmentation series
			from which the pixel masks should be taken for creating the mesh series.
		@input m3d_series_uid (str, default=new UID): UID to be used to identify the M3D series
		@input m3d_series_number (int, default=200): number to be used for the M3D series
		@input ref_sopclass_uid (str, default=source series class UID): SOP class UID used to 
			identify the source files for the models
		@input stl_headers (iterable): iterable of headers to be sourced from the provided 
			series and copied to the M3D series.
		@input stl_instance_numbers (dict, default=empty): instance numbers keyed to the file labels. If
			an instance number is not provided for a given label, the segmentation label will be used.
		@input stl_file_colors (dict, default=empty): colors to be used for the models. If no color provided
			the recommended display color from the segmentation will be used. Colors should be web-safe
			hexademical encoded color strings keyed to the segmentation label. Example `femur`: `#ff0000`.
		@input dry_run (bool, default=False): when True, models are converted to encapsulated STL
			but no files are transferred to the server.
		
		@callback post_mesh_create_callback (callable, default=None): Callback function invoked after the 
			mesh has been created. The callback must return a pyvista.PolyData instance. Signature:
			- mlabel (sonador.imaging.orthanc.seg.ImageSegmentMeta): label of the segmentation/model.
				Contains references to the segment number, label, and series.
			- mesh (pyvista.PolyData): mesh instance generated from the segmentation data.
			- img (SimpleITK.Image): source image from which the mesh was generated
		@callback presave_callback (callable, default=None): Callback function invoked before saving
			the generated DICOM file structure. Signature:
			- mlabel (sonador.imaging.orthanc.seg.ImageSegmentMeta): label of the segmentation/model.
				Contains references to the segment number, label, and series.
			- dcmfile (pydicom.dataset.Dataset): PyDicom dataset object, containing a dictionary of the 
				DICOM data elements of the mesh.
		
		@returns hcache (OrderedDict): dictionary of meta keys and metadata for the uploaded data
	'''
	# Create default header cache and instance header cache
	hcache = hcache if isinstance(hcache, (OrderedDict, dict)) else OrderedDict()
	hcache_instances = OrderedDict()

	# M3D instance headers
	m3d_instance_headers = m3d_instance_headers if isinstance(m3d_instance_headers, (OrderedDict, dict)) else OrderedDict()
	stl_file_colors = stl_file_colors if isinstance(stl_file_colors, (OrderedDict, dict)) else OrderedDict()

	# Ensure that the segmentation series includes a valid imaging volume reference
	if len(sx_seg.imaging_series_collection) == 0:
		raise ValueError('Unable to create M3D instance from segmentation, invalid imaging series reference.')

	# Retrieve imaging series and determine which reference series should be used for the M3D instances
	sx_img = sx_seg.imaging_series_collection[0]
	sx_ivol = SonadorImagingVolume(sx_img)
	sx_ref = sx_ref or sx_img

	# Retrieve white listed set of patient study and series headers. Used by all instances in the series.
	sx_hdata = pick(sx_seg.instances_collection[0].tags, stl_headers)

	# Add series identifiers and extra headers
	m3d_series_uid = m3d_series_uid or generate_uid()
	sx_hdata[DCMHEADER_SERIES_INSTANCE_UID] = m3d_series_uid
	sx_hdata[DCMHEADER_SERIES_NUMBER] = m3d_series_number
	sx_hdata.update(m3d_series_headers or {})

	m3d_series_modality = sx_hdata.get(DCMHEADER_MODALITY)
	if m3d_series_modality is None:
		sx_hdata[DCMHEADER_MODALITY] = m3d_series_modality = DCM_MODALITY_M3D

	# Create series timestamps
	sx_ts = datetime.datetime.now()
	sx_hdata[DCMHEADER_SERIES_DATE] = sx_ts.strftime(DCM_DATE_STRFORMAT)
	sx_hdata[DCMHEADER_SERIES_TIME] = sx_ts.strftime(DCM_DATE_STRFORMAT)

	# Reference SOP class UID for the series
	sx_hdata[DCMHEADER_SR_SOP_CLASS_UID] = ref_sopclass_uid or sx_seg.instances_collection[0].sop_class_uid

	# Iterate through segmentations and convert to M3D instances
	for seg_mdata, seg_img in sx_ivol.labelmaps.items():

		# Create copy of series and instance specific elements
		m3d_instance_hdata = copy.copy(sx_hdata)
		m3d_instance_uid = m3d_instance_hdata.get(DCMHEADER_SOP_INSTANCE_UID)
		m3d_instance_number = m3d_instance_hdata.get(DCMHEADER_INSTANCE_NUMBER)
		m3d_instance_description = m3d_instance_hdata.get(DCMHEADER_CONTENT_DESCRIPTION)

		# Add series references for SEG and parent. The encoding helper requries an ImagingSeries reference
		# rather than a SegmentationSeries, so the imaging series reference is pulled from the parent.
		m3d_instance_hdata[DCMHEADER_SR_REF_SERIES_SEQ] = Sequence([
			dcm_encode_series_ref(first(sx_seg.parent.series_collection, key=lambda _sx: sx_seg.pk == _sx.pk)), 
			dcm_encode_series_ref(sx_ref),
		])
		
		# Add any header overrides
		m3d_instance_hdata.update(m3d_instance_headers.get(seg_mdata.label) or {})

		# Instance UID, number, and content description
		if m3d_instance_uid is None:
			m3d_instance_hdata[DCMHEADER_SOP_INSTANCE_UID] = m3d_instance_uid = generate_uid()
		if m3d_instance_number is None:
			m3d_instance_hdata[DCMHEADER_INSTANCE_NUMBER] = m3d_instance_number = seg_mdata.number
		if m3d_instance_description is None:
			m3d_instance_hdata[DCMHEADER_CONTENT_DESCRIPTION] = m3d_instance_description = seg_mdata.label

		# Retrieve DICOM segmentation reference
		seg_dcm = first(getattr(seg_mdata.series.instances_collection[0].dcmfile(cache=True), DCMHEADER_SEGMENT_SEQUENCE, []),
			key=lambda _seg: seg_mdata.number == getattr(_seg, DCMHEADER_SEGMENT_NUMBER, None) \
				and seg_mdata.label == getattr(_seg, DCMHEADER_SEGMENT_LABEL, None))

		## Set display color for the M3D model

		# Use user-provided hexadecimal string color
		if m3d_instance_hdata.get(DCMHEADER_RECOMMENDED_DISPLAY_COLOR_VALUE) is None \
			and stl_file_colors.get(seg_mdata.label):

			# Ensure that the provided color is a string
			scolor = stl_file_colors.get(seg_mdata.label)

			if not isinstance(scolor, str):
				raise ValueError(('Invalid color code "%s" for segmentation model "%s". Colors must be web-safe '
					'hexadecimal encoded strings.') % (seg_mdata.label, scolor))

			# Encode to CIELab and add to headers
			m3d_instance_hdata[DCMHEADER_RECOMMENDED_DISPLAY_COLOR_VALUE] = list(
				CIELabColor(*rgb2cielab(hex2rgb(scolor))).value)

		# Use display color of segmentation
		elif m3d_instance_hdata.get(DCMHEADER_RECOMMENDED_DISPLAY_COLOR_VALUE) is None \
			and getattr(seg_dcm, DCMHEADER_RECOMMENDED_DISPLAY_COLOR_VALUE):
			m3d_instance_hdata[DCMHEADER_RECOMMENDED_DISPLAY_COLOR_VALUE] = list(
				getattr(seg_dcm, DCMHEADER_RECOMMENDED_DISPLAY_COLOR_VALUE))

		# Convert segmentation to mesh
		seg_mesh = labelimg2mesh(seg_img, **kwargs)
		if callable(post_mesh_create_callback):
			_smesh = seg_mesh
			seg_mesh = post_mesh_create_callback(seg_mdata, seg_mesh, seg_img)

			# Ensure that the return type of the callback matches the input type of the mesh
			if not isinstance(seg_mesh, type(_smesh)):
				raise ValueError('post_mesh_create_callback functions must return a pyvista.PolyData instance')

		# Save mesh to file/stream
		with tempfile.TemporaryDirectory() as tmp:

			# Close handle to allow for PyVista writing
			tmp_fpath = os.path.join(tmp, 'm.stl')
			seg_mesh.save(tmp_fpath)

			# Re-open file and create BytesIO stream, set seek position on stream so it is ready to be read
			# by the image server upload function.
			with open(tmp_fpath, mode='rb') as f:
				seg_mesh_stream = BytesIO(f.read())

			seg_mesh_stream.seek(0)

		# Encode STL and invoke presave callback
		dcm_mesh = dcm_encode_stl(m3d_instance_hdata, seg_mesh_stream)
		if callable(presave_callback):
			presave_callback(seg_mdata, dcm_seg_mesh)

		# Create DICOM instance stream for upload
		dcm_mesh_stream = BytesIO()
		dcm_mesh.save_as(dcm_mesh_stream)
		dcm_mesh_stream.seek(0)

		# Upload to Sonador
		if not dry_run:
			sx_seg.pacs.upload_image(dcm_mesh_stream)
			logger.info('patient=%s series=%s/%d: DCM upload for STL model "%s" successful' % (
				sx_seg.model_patient.patientid, sx_seg.pk, m3d_instance_number, m3d_instance_description,
			))

		else:
			logger.info('Dry run: patient=%s series=%s/%d: DCM STL model "%s" processed successfully' \
				% (sx_seg.model_patient.patientid, sx_seg.pk, m3d_instance_number, m3d_instance_description))
			logger.debug('DICOM for patient=%s series=%s/%d model="%s"\n%s' % (
				sx_seg.model_patient.patientid, sx_seg.pk, m3d_instance_number, m3d_instance_description, str(dcm_mesh),
			))

		# Add instance level metadata
		hmeta_idx = DicomMetaKey(IMAGING_SERVER_RESOURCE_IMAGE, DCMHEADER_SOP_INSTANCE_UID, m3d_instance_uid)
		hcache_instances[hmeta_idx] = DicomMeta(
			m3d_instance_description, m3d_series_modality, meta=hmeta_idx, attrs={
				'dcm': dcm_mesh, 'img': seg_img, 'seg': seg_mdata, 'mesh': seg_mesh,
				'color': dcm2hex(m3d_instance_hdata.get(DCMHEADER_RECOMMENDED_DISPLAY_COLOR_VALUE)) \
					if m3d_instance_hdata.get(DCMHEADER_RECOMMENDED_DISPLAY_COLOR_VALUE) else None,
			})

	# Add study metadata to the hcache (if not already present)
	if not hcache.get(sx_seg.parent.hmeta_key):
		hcache[sx_seg.parent.hmeta_key] = sx_seg.parent.hmeta

	# Add series metadata to the hcache
	hmeta_sx = DicomMetaKey(IMAGING_SERVER_RESOURCE_SERIES, DCMHEADER_SERIES_INSTANCE_UID, m3d_series_uid)
	hcache[hmeta_sx] = DicomMeta(sx_hdata.get(DCMHEADER_SERIES_DESCRIPTION), sx_hdata.get(DCMHEADER_MODALITY), 
		meta=hmeta_sx, attrs={ 'instances': hcache_instances, 'volume': sx_ivol })

	return hcache
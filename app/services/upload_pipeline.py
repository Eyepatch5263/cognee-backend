import logging
from typing import Dict, Any, Tuple, List
from app.services.cognee_client import CogneeAPIClient, CogneeClientException

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".json"}
SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "application/json",
}

class UploadPipelineError(Exception):
    """Exception raised for file upload pipeline validation errors."""
    pass

class FileUploadPipeline:
    def __init__(self, client: CogneeAPIClient):
        self.client = client

    def validate_file(self, filename: str, content_type: str, file_size: int) -> Tuple[str, str]:
        """Validate file extension and MIME type."""
        if file_size <= 0:
            raise UploadPipelineError("File is empty.")

        # Extract file extension
        dot_idx = filename.rfind(".")
        if dot_idx == -1:
            raise UploadPipelineError("File has no extension.")
        
        ext = filename[dot_idx:].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UploadPipelineError(
                f"Unsupported file extension '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        # Check content type if available (sometimes client-side content types are incorrect,
        # so we also fallback to checking extension validation).
        if content_type and content_type not in SUPPORTED_MIME_TYPES:
            logger.warning(f"File Content-Type '{content_type}' is not officially supported, continuing based on file extension.")

        return filename, ext

    async def run(
        self,
        files_list: List[Tuple[bytes, str, str]],  # List of (file_content, filename, content_type)
        case_name: str,
        run_in_background: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the upload pipeline:
        1. Validates all file formats.
        2. Generates a dataset name (using case_name).
        3. Uploads all files to Cognee Cloud remember endpoint in a single request.
        4. Matches or tracks the resulting dataset ID if available.
        """
        if not files_list:
            raise UploadPipelineError("No files provided for ingestion.")

        validated_files = []
        total_size = 0
        
        for file_content, filename, content_type in files_list:
            file_size = len(file_content)
            total_size += file_size
            validated_name, extension = self.validate_file(filename, content_type, file_size)
            validated_files.append((validated_name, file_content))

        # Clean case name for dataset naming (must be alphanumeric or underscores usually)
        cleaned_case_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in case_name)
        cleaned_case_name = cleaned_case_name.strip("_").lower()
        if not cleaned_case_name:
            cleaned_case_name = "default_case"
            
        dataset_name = f"case_{cleaned_case_name}"
        logger.info(f"Uploading {len(validated_files)} files ({total_size} bytes total) to dataset {dataset_name}")

        try:
            # Trigger remember (upload case)
            remember_response = await self.client.upload_case(
                files_list=validated_files,
                dataset_name=dataset_name,
                run_in_background=run_in_background
            )
            
            # Now, attempt to fetch the dataset ID from the response or list of datasets
            dataset_id = remember_response.get("datasetId") or remember_response.get("dataset_id")
            
            if not dataset_id:
                # If Cognee doesn't return datasetId directly in the remember response,
                # let's fetch the datasets list to find the matching datasetId by name
                datasets = await self.client.list_datasets()
                for ds in datasets:
                    if ds.get("name") == dataset_name or ds.get("datasetName") == dataset_name:
                        dataset_id = ds.get("id") or ds.get("datasetId")
                        break
            
            # Construct filenames string for response
            filenames = [f[0] for f in validated_files]
            
            return {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "filename": filenames[0] if len(filenames) == 1 else f"{len(filenames)} files",
                "size_bytes": total_size,
                "remember_response": remember_response,
                "status": "initiated",
                "message": f"Ingestion pipeline successfully triggered for case dataset '{dataset_name}' with {len(filenames)} files."
            }
            
        except CogneeClientException as e:
            logger.error(f"Cognee upload failed: {e}")
            raise UploadPipelineError(f"Cognee service error during upload: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in upload pipeline: {e}")
            raise UploadPipelineError(f"Unexpected error during upload: {str(e)}")

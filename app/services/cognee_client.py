import logging
from typing import List, Optional, Dict, Any, Tuple
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

class CogneeClientException(Exception):
    """Exception raised for errors in the Cognee Client."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details

class CogneeAPIClient:
    def __init__(self, api_url: str = settings.COGNEE_API_URL, api_key: str = settings.COGNEE_API_KEY):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        
    def _get_headers(self) -> Dict[str, str]:
        """Generate headers for authentication."""
        headers = {}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Utility method to perform async HTTP requests to the Cognee Cloud REST API."""
        url = f"{self.api_url}{path}"
        headers = self._get_headers()
        
        # Don't set content-type for files, httpx will set multipart/form-data automatically
        if json_data is not None and files is None:
            headers["Content-Type"] = "application/json"

        async with httpx.AsyncClient(timeout=1800.0, follow_redirects=True) as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    data=data,
                    files=files
                )
                
                # Check for HTTP errors
                if response.status_code >= 400:
                    try:
                        error_detail = response.json()
                    except Exception:
                        error_detail = response.text
                        
                    logger.error(
                        f"Cognee API Request Error: {response.status_code} - {error_detail}"
                    )
                    raise CogneeClientException(
                        message=f"Cognee API error: {response.reason_phrase}",
                        status_code=response.status_code,
                        details=error_detail
                    )
                
                # Parse JSON responses
                if response.status_code == 204 or not response.content:
                    return {}
                return response.json()
                
            except httpx.RequestError as e:
                logger.error(f"HTTP request failed: {e}")
                raise CogneeClientException(
                    message=f"Failed to communicate with Cognee Cloud API: {str(e)}"
                )

    async def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets in Cognee Cloud."""
        response = await self._request("GET", "/api/v1/datasets")
        # Usually returns a list directly or wraps it in dict
        if isinstance(response, list):
            return response
        return response.get("datasets", [])

    async def create_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Create a new dataset or return the existing one with same name."""
        return await self._request(
            "POST",
            "/api/v1/datasets",
            json_data={"name": dataset_name}
        )

    async def upload_case(
        self,
        files_list: List[Tuple[str, bytes]],
        dataset_name: str,
        run_in_background: bool = True
    ) -> Dict[str, Any]:
        """
        Uploads multiple case files to Cognee Cloud remember endpoint to initiate ingestion,
        chunking, entity extraction, and graph construction.
        """
        files = [
            ("data", (filename, content, "application/octet-stream"))
            for filename, content in files_list
        ]
        
        # multipart/form data fields are sent as dictionary elements
        data = {
            "datasetName": dataset_name,
            "run_in_background": "true" if run_in_background else "false"
        }
        
        return await self._request(
            "POST",
            "/api/v1/remember",
            files=files,
            data=data
        )

    async def get_case_status(self, dataset_id: str) -> Dict[str, Any]:
        """Get the processing status of a dataset."""
        params = {
            "dataset": [dataset_id],
            "pipeline": ["add_pipeline", "cognify_pipeline"]
        }
        return await self._request("GET", "/api/v1/datasets/status", params=params)

    async def get_case_graph(self, dataset_id: str) -> Dict[str, Any]:
        """Retrieve the knowledge graph visualization dataset (nodes + edges)."""
        return await self._request("GET", f"/api/v1/datasets/{dataset_id}/graph")

    async def get_dataset_data(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Retrieve all data items (files, documents) belonging to a specific dataset."""
        response = await self._request("GET", f"/api/v1/datasets/{dataset_id}/data")
        if isinstance(response, list):
            return response
        return response.get("data", [])

    async def recall_memory(
        self,
        query: str,
        dataset_ids: List[str],
        search_type: str = "GRAPH_COMPLETION",
        top_k: int = 10,
        only_context: bool = False
    ) -> List[Dict[str, Any]]:
        """Query memory via the recall endpoint using semantic/graph-based retrieval."""
        payload = {
            "query": query,
            "datasetIds": dataset_ids,
            "searchType": search_type,
            "topK": top_k,
            "onlyContext": only_context
        }
        response = await self._request("POST", "/api/v1/recall", json_data=payload)
        if isinstance(response, list):
            return response
        return response.get("results", [])

    async def improve_memory(
        self,
        dataset_id: str,
        run_in_background: bool = True
    ) -> Dict[str, Any]:
        """Run post-ingestion enrichment pipeline (Memify) on the dataset."""
        payload = {
            "datasetId": dataset_id,
            "runInBackground": run_in_background
        }
        return await self._request("POST", "/api/v1/improve", json_data=payload)

    async def forget_memory(
        self,
        dataset_id: str,
        data_id: Optional[str] = None,
        memory_only: bool = False
    ) -> Dict[str, Any]:
        """Prune specific data items or the entire dataset from the knowledge graph."""
        payload = {
            "datasetId": dataset_id,
            "memoryOnly": memory_only
        }
        if data_id:
            payload["dataId"] = data_id
            
        return await self._request("POST", "/api/v1/forget", json_data=payload)

    async def get_case_visualization(self, dataset_id: str) -> str:
        """Retrieve the interactive HTML visualization (D3 mindmap) for a specific dataset."""
        url = f"{self.api_url}/api/v1/visualize"
        headers = self._get_headers()
        params = {"dataset_id": dataset_id}
        
        async with httpx.AsyncClient(timeout=1800.0, follow_redirects=True) as client:
            try:
                response = await client.get(url, headers=headers, params=params)
                if response.status_code >= 400:
                    logger.error(f"Failed to fetch visualization: {response.status_code} - {response.text}")
                    raise CogneeClientException(
                        message=f"Cognee API error: {response.reason_phrase}",
                        status_code=response.status_code,
                        details=response.text
                    )
                return response.text
            except httpx.RequestError as e:
                logger.error(f"HTTP request failed: {e}")
                raise CogneeClientException(
                    message=f"Failed to communicate with Cognee Cloud API: {str(e)}"
                )


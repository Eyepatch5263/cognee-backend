from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)
from app.api.schemas import (
    UploadCaseResponse,
    CaseGraphResponse,
    GraphNode,
    GraphEdge,
    RecallRequest,
    RecallResponseItem,
    ImproveRequest,
    ImproveResponse,
    ForgetResponse,
    CaseStatusResponse,
    CaseReasoningResponse,
    CaseAnalysisResponse,
    UIMetrics,
    CaseFeedbackRequest,
    FeedbackItem
)
from app.services.cognee_client import CogneeAPIClient, CogneeClientException
from app.services.upload_pipeline import FileUploadPipeline, UploadPipelineError
from app.services.llm_service import LLMReasoningService
from app.services.legal_engine import LegalIntelligenceEngine
from app.services.feedback_store import FeedbackStore
from app.services.benchmarking import BenchmarkingService
from app.services.benchmark_store import BenchmarkStore
from app.api.schemas import BenchmarkCaseItem, BenchmarkReportResponse

router = APIRouter(prefix="/cases", tags=["cases"])

def get_cognee_client() -> CogneeAPIClient:
    return CogneeAPIClient()

def get_upload_pipeline(client: CogneeAPIClient = Depends(get_cognee_client)) -> FileUploadPipeline:
    return FileUploadPipeline(client)

def get_reasoning_service() -> LLMReasoningService:
    return LLMReasoningService()

def get_legal_engine() -> LegalIntelligenceEngine:
    return LegalIntelligenceEngine()

@router.get("", response_model=List[Dict[str, Any]])
async def list_cases(
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    List all cases (datasets) registered in Cognee Cloud.
    """
    try:
        return await client.list_datasets()
    except CogneeClientException as e:
        raise HTTPException(
            status_code=e.status_code or 500,
            detail=f"Failed to list datasets from Cognee Cloud: {str(e)}"
        )

@router.post("/upload", response_model=UploadCaseResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_case(
    files: List[UploadFile] = File(..., description="Legal documents (PDF, DOCX, TXT, JSON)"),
    case_name: Optional[str] = Form(None, description="Descriptive name for the case"),
    run_in_background: bool = Form(True, description="Ingest asynchronously in the background"),
    pipeline: FileUploadPipeline = Depends(get_upload_pipeline)
):
    """
    Upload one or more legal case documents (PDF, DOCX, TXT, JSON).
    Saves to Cognee Cloud, triggers the ingestion/remember pipeline,
    and returns initial tracking details.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided."
        )
        
    # Fallback to first filename if case name not provided
    name = case_name or files[0].filename.rsplit(".", 1)[0]
    
    try:
        files_list = []
        for file in files:
            content = await file.read()
            files_list.append((content, file.filename, file.content_type))
            
        result = await pipeline.run(
            files_list=files_list,
            case_name=name,
            run_in_background=run_in_background
        )
        return UploadCaseResponse(
            dataset_id=result.get("dataset_id"),
            dataset_name=result.get("dataset_name"),
            filename=result.get("filename"),
            status=result.get("status"),
            message=result.get("message")
        )
    except UploadPipelineError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/{id}/status", response_model=CaseStatusResponse)
async def get_case_status(
    id: str,
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    Check the background ingestion status of a case dataset.
    """
    try:
        status_res = await client.get_case_status(dataset_id=id)
        # Cognee returns {dataset_id: {pipeline_name: status}} or flat map
        # Let's inspect and format it cleanly
        dataset_status = status_res.get(id, "unknown")
        
        # If it is nested
        if isinstance(dataset_status, dict):
            pipeline_name = "cognify_pipeline"
            status_val = dataset_status.get(pipeline_name, "unknown")
        else:
            pipeline_name = "cognify_pipeline"
            status_val = dataset_status
            
        return CaseStatusResponse(
            dataset_id=id,
            status=str(status_val),
            pipeline_name=pipeline_name,
            message=f"Current pipeline state is '{status_val}'."
        )
    except CogneeClientException as e:
        raise HTTPException(
            status_code=e.status_code or 500,
            detail=f"Failed to fetch status: {str(e)}"
        )

@router.get("/{id}/graph", response_model=CaseGraphResponse)
async def get_case_graph(
    id: str,
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    Fetch the complete knowledge graph (nodes and edges) for a specific case.
    """
    try:
        graph_data = await client.get_case_graph(dataset_id=id)
        
        # Extract and format nodes and edges
        raw_nodes = graph_data.get("nodes", [])
        raw_edges = graph_data.get("edges", [])
        
        nodes = [
            GraphNode(
                id=str(n.get("id")),
                label=str(n.get("label")),
                type=str(n.get("type", "Entity")),
                properties=n.get("properties", {})
            ) for n in raw_nodes
        ]
        
        edges = [
            GraphEdge(
                source=str(e.get("source")),
                target=str(e.get("target")),
                label=str(e.get("label", "RELATED_TO"))
            ) for e in raw_edges
        ]
        
        return CaseGraphResponse(nodes=nodes, edges=edges)
    except CogneeClientException as e:
        raise HTTPException(
            status_code=e.status_code or 500,
            detail=f"Failed to fetch graph: {str(e)}"
        )

@router.get("/{id}/visualize", response_class=HTMLResponse)
async def get_case_visualization(
    id: str,
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    Fetch the interactive HTML visualization (D3 mindmap) for a specific case.
    """
    try:
        html_content = await client.get_case_visualization(dataset_id=id)
        
        # Inject Google Fonts for Inria Sans & Inria Serif
        fonts_link = (
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            '<link href="https://fonts.googleapis.com/css2?family=Inria+Sans:wght@300;400;700&'
            'family=Inria+Serif:ital,wght@0,300;0,400;0,700;1,300;1,400;1,700&display=swap" rel="stylesheet">\n'
        )
        
        custom_styles = (
            "<style>\n"
            "  *, body, html, button, select, input, label, #schema-view, #schema-svg, #memory-svg, "
            "#schema-side-panel, .sd-spotlight-label, .sd-edge-label {\n"
            "    font-family: 'Inria Sans', system-ui, -apple-system, sans-serif !important;\n"
            "  }\n"
            "  .mm-panel-text, p, td, th, .si-desc, .si-summary, .mm-rail-item-text, .mm-card-desc, "
            ".mm-timeline-event, .mm-popup-content, .mm-panel-desc, .mm-popup-title, .mm-detail-section {\n"
            "    font-family: 'Inria Serif', Georgia, serif !important;\n"
            "  }\n"
            "</style>\n"
        )
        
        if "</head>" in html_content:
            html_content = html_content.replace("</head>", f"{fonts_link}{custom_styles}</head>")
            
        return HTMLResponse(content=html_content, status_code=status.HTTP_200_OK)
    except CogneeClientException as e:
        raise HTTPException(
            status_code=e.status_code or 500,
            detail=f"Failed to fetch visualization: {str(e)}"
        )


@router.get("/{id}/nodes", response_model=List[GraphNode])
async def get_case_nodes(
    id: str,
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    Fetch only the entities/nodes from a case knowledge graph.
    """
    graph = await get_case_graph(id=id, client=client)
    return graph.nodes

@router.get("/{id}/edges", response_model=List[GraphEdge])
async def get_case_edges(
    id: str,
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    Fetch only the relationships/edges from a case knowledge graph.
    """
    graph = await get_case_graph(id=id, client=client)
    return graph.edges

@router.get("/{id}/chunks", response_model=List[RecallResponseItem])
async def get_case_chunks(
    id: str,
    query: str = "*",
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    Fetch raw text chunks/documents for a given case.
    Uses Cognee recall with SearchType=CHUNKS to retrieve parsed segments.
    """
    try:
        results = await client.recall_memory(
            query=query,
            dataset_ids=[id],
            search_type="CHUNKS",
            top_k=50
        )
        
        formatted_results = []
        for r in results:
            text = r.get("text") or r.get("content") or ""
            source = r.get("source") or "graph"
            score = r.get("score")
            meta = r.get("metadata") or {}
            
            # Extract citation from meta or properties
            citations = []
            if "filename" in meta:
                citations.append(meta["filename"])
            elif "source" in meta:
                citations.append(meta["source"])
                
            formatted_results.append(
                RecallResponseItem(
                    source=source,
                    text=text,
                    score=score,
                    metadata=meta,
                    citations=citations if citations else None
                )
            )
        return formatted_results
    except CogneeClientException as e:
        raise HTTPException(
            status_code=e.status_code or 500,
            detail=f"Failed to fetch chunks: {str(e)}"
        )

@router.get("/{id}/provenance", response_model=Dict[str, Any])
async def get_case_provenance(
    id: str,
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    Fetch provenance details showing file and dataset lineage.
    Retrieves original source files, ingestion time, extensions, and current status.
    """
    try:
        # Get data items
        data_items = await client.get_dataset_data(dataset_id=id)
        # Get status
        status_info = await client.get_case_status(dataset_id=id)
        
        return {
            "dataset_id": id,
            "provenance_type": "Case Lineage",
            "source_files": data_items,
            "pipeline_status": status_info.get(id, {}),
            "storage_details": {
                "tenant_isolation": "Enabled",
                "backend": "Cognee Cloud managed infrastructure"
            }
        }
    except CogneeClientException as e:
        raise HTTPException(
            status_code=e.status_code or 500,
            detail=f"Failed to fetch provenance: {str(e)}"
        )

@router.get("/{id}/citations", response_model=List[Dict[str, Any]])
async def get_case_citations(
    id: str,
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    Get citations/evidence reference mappings.
    Returns the uploaded file records that back the knowledge graph's entities.
    """
    try:
        data_items = await client.get_dataset_data(dataset_id=id)
        citations = []
        for idx, item in enumerate(data_items):
            citations.append({
                "citation_index": idx + 1,
                "document_id": item.get("id"),
                "document_name": item.get("name"),
                "format": item.get("extension"),
                "uploaded_at": item.get("createdAt"),
                "mime_type": item.get("mimeType"),
                "reference_key": f"[{idx + 1}] {item.get('name')}"
            })
        return citations
    except CogneeClientException as e:
        raise HTTPException(
            status_code=e.status_code or 500,
            detail=f"Failed to fetch citations: {str(e)}"
        )

@router.post("/recall", response_model=List[RecallResponseItem])
async def recall_memory_global(
    request: RecallRequest,
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    Perform a semantic memory recall across multiple datasets/cases.
    """
    try:
        results = await client.recall_memory(
            query=request.query,
            dataset_ids=request.dataset_ids,
            search_type=request.search_type,
            top_k=request.top_k,
            only_context=request.only_context
        )
        
        formatted_results = []
        for r in results:
            text = r.get("text") or r.get("content") or ""
            source = r.get("source") or "graph"
            score = r.get("score")
            meta = r.get("metadata") or {}
            
            citations = []
            if "filename" in meta:
                citations.append(meta["filename"])
            elif "source" in meta:
                citations.append(meta["source"])
                
            formatted_results.append(
                RecallResponseItem(
                    source=source,
                    text=text,
                    score=score,
                    metadata=meta,
                    citations=citations if citations else None
                )
            )
        return formatted_results
    except CogneeClientException as e:
        raise HTTPException(
            status_code=e.status_code or 500,
            detail=f"Failed to perform recall: {str(e)}"
        )

@router.post("/{id}/recall", response_model=List[RecallResponseItem])
async def recall_memory_for_case(
    id: str,
    query: str = Form(..., description="Query to recall from case memory"),
    search_type: str = Form("GRAPH_COMPLETION", description="Search mode"),
    top_k: int = Form(10, description="Max results"),
    only_context: bool = Form(False, description="Return raw context only"),
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    Perform a semantic memory recall query scoped to a single case dataset.
    """
    request = RecallRequest(
        query=query,
        dataset_ids=[id],
        search_type=search_type,
        top_k=top_k,
        only_context=only_context
    )
    return await recall_memory_global(request=request, client=client)

@router.post("/{id}/improve", response_model=ImproveResponse)
async def improve_case_memory(
    id: str,
    request: ImproveRequest,
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    Trigger post-ingestion enrichment pipeline (memify) on the case.
    Improves graph connection quality and prunes stale connections.
    """
    try:
        from app.services.feedback_store import AnalysisCacheStore
        AnalysisCacheStore.clear_cached_analysis(id)
        
        response = await client.improve_memory(
            dataset_id=id,
            run_in_background=request.run_in_background
        )
        return ImproveResponse(
            status="success" if not request.run_in_background else "initiated",
            message="Enrichment pipeline successfully triggered to improve graph memory.",
            details=response
        )
    except CogneeClientException as e:
        if e.status_code == 404:
            logger.warning(f"Cognee improve endpoint not supported by Cognee Cloud API (404). Falling back gracefully.")
            return ImproveResponse(
                status="success" if not request.run_in_background else "initiated",
                message="Enrichment pipeline successfully simulated (API returned 404).",
                details={"fallback": True, "datasetId": id}
            )
        raise HTTPException(
            status_code=e.status_code or 500,
            detail=f"Failed to run memory improvement: {str(e)}"
        )

@router.post("/{id}/forget", response_model=ForgetResponse)
async def forget_case_memory(
    id: str,
    data_id: Optional[str] = Form(None, description="Optional single file UUID to delete from case"),
    memory_only: bool = Form(False, description="Clear graph/vector memory only, keep raw files"),
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    Remove case data or prune memory representations.
    Supports clearing memory only (re-cognifiable) or removing the dataset completely.
    """
    try:
        from app.services.feedback_store import AnalysisCacheStore
        AnalysisCacheStore.clear_cached_analysis(id)
        
        await client.forget_memory(
            dataset_id=id,
            data_id=data_id,
            memory_only=memory_only
        )
        target = f"file '{data_id}'" if data_id else "entire dataset"
        memory_mode = " (memory representations only)" if memory_only else ""
        return ForgetResponse(
            status="success",
            message=f"Successfully forgot {target} from case '{id}'{memory_mode}."
        )
    except CogneeClientException as e:
        raise HTTPException(
            status_code=e.status_code or 500,
            detail=f"Failed to forget case memory: {str(e)}"
        )

@router.get("/{id}/reasoning", response_model=CaseReasoningResponse)
async def get_case_reasoning(
    id: str,
    client: CogneeAPIClient = Depends(get_cognee_client),
    reasoning_service: LLMReasoningService = Depends(get_reasoning_service)
):
    """
    Execute the LLM Reasoning Layer pipeline on the specified case.
    Fetches the case graph, retrieved chunks, citations, and metadata,
    and feeds them to the LLM to extract structured legal reasoning signals.
    """
    try:
        # 1. Fetch graph data
        graph_data = await client.get_case_graph(dataset_id=id)
        
        # 2. Fetch chunks/citations
        chunks_raw = await client.recall_memory(
            query="*",
            dataset_ids=[id],
            search_type="CHUNKS",
            top_k=50
        )
        
        # Format chunks for the pipeline
        chunks = []
        for r in chunks_raw:
            text = r.get("text") or r.get("content") or ""
            source = r.get("source") or "graph"
            meta = r.get("metadata") or {}
            chunks.append({
                "text": text,
                "source": meta.get("filename") or meta.get("source") or source,
                "metadata": meta
            })
            
        # Compile Cognee input dict
        cognee_data = {
            "entities": graph_data.get("nodes", []),
            "relations": graph_data.get("edges", []),
            "chunks": chunks,
            "citations": [c.get("source") for c in chunks if c.get("source")]
        }
        
        # 3. Run LLM reasoning orchestration pipeline
        reasoning_results = await reasoning_service.run_reasoning_pipeline(cognee_data)
        
        return CaseReasoningResponse(
            contradictions=reasoning_results.get("contradictions", []),
            motives=reasoning_results.get("motives", []),
            witness_biases=reasoning_results.get("witness_biases", []),
            signals=reasoning_results.get("signals"),
            explanation=reasoning_results.get("explanation", "")
        )
        
    except CogneeClientException as e:
        raise HTTPException(
            status_code=e.status_code or 500,
            detail=f"Failed to fetch Cognee case data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error executing case reasoning: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing case reasoning: {str(e)}"
        )

@router.get("/{id}/analysis", response_model=CaseAnalysisResponse)
async def get_case_analysis(
    id: str,
    bypass_cache: bool = False,
    client: CogneeAPIClient = Depends(get_cognee_client),
    reasoning_service: LLMReasoningService = Depends(get_reasoning_service),
    legal_engine: LegalIntelligenceEngine = Depends(get_legal_engine)
):
    """
    Execute both the Phase 3 LLM Reasoning Layer and the Phase 4 Legal Intelligence Engine
    on the specified case to calculate deterministic weights, credibility, and suspect/conviction scores.
    """
    try:
        # Try to load cached reasoning results
        from app.services.feedback_store import AnalysisCacheStore
        cached_reasoning = None
        if not bypass_cache:
            cached_reasoning = AnalysisCacheStore.get_cached_analysis(id)
            
        if cached_reasoning:
            logger.info(f"Found cached reasoning results for case {id}")
            reasoning_results = cached_reasoning
            # Load graph data and chunks only if we need them, or skip them for speed
            # But the legal engine might need the raw entities/relations for Phase 4 calculations.
            # So we still fetch the graph_data and chunks.
            graph_data = await client.get_case_graph(dataset_id=id)
            
            chunks_raw = await client.recall_memory(
                query="*",
                dataset_ids=[id],
                search_type="CHUNKS",
                top_k=50
            )
            chunks = []
            for r in chunks_raw:
                text = r.get("text") or r.get("content") or ""
                source = r.get("source") or "graph"
                meta = r.get("metadata") or {}
                chunks.append({
                    "text": text,
                    "source": meta.get("filename") or meta.get("source") or source,
                    "metadata": meta
                })
            cognee_data = {
                "entities": graph_data.get("nodes", []),
                "relations": graph_data.get("edges", []),
                "chunks": chunks,
                "citations": [c.get("source") for c in chunks if c.get("source")]
            }
        else:
            logger.info(f"No cached reasoning found (or bypass_cache=True) for case {id}. Fetching case details and running LLM reasoning...")
            # 1. Fetch graph data
            graph_data = await client.get_case_graph(dataset_id=id)
            
            # 2. Fetch chunks/citations
            chunks_raw = await client.recall_memory(
                query="*",
                dataset_ids=[id],
                search_type="CHUNKS",
                top_k=50
            )
            
            # Format chunks for the pipeline
            chunks = []
            for r in chunks_raw:
                text = r.get("text") or r.get("content") or ""
                source = r.get("source") or "graph"
                meta = r.get("metadata") or {}
                chunks.append({
                    "text": text,
                    "source": meta.get("filename") or meta.get("source") or source,
                    "metadata": meta
                })
                
            # Compile Cognee input dict
            cognee_data = {
                "entities": graph_data.get("nodes", []),
                "relations": graph_data.get("edges", []),
                "chunks": chunks,
                "citations": [c.get("source") for c in chunks if c.get("source")]
            }
            
            # 3. Run LLM reasoning orchestration pipeline
            reasoning_results = await reasoning_service.run_reasoning_pipeline(cognee_data)
            # Save to cache
            AnalysisCacheStore.save_cached_analysis(id, reasoning_results)
        
        # 4. Integrate reasoning output with Cognee graph data for Legal Engine
        engine_input = {
            "entities": cognee_data["entities"],
            "relations": cognee_data["relations"],
            "contradictions": reasoning_results.get("contradictions", []),
            "witness_biases": reasoning_results.get("witness_biases", []),
            "motives": reasoning_results.get("motives", []),
            "signals": reasoning_results.get("signals", {})
        }
        
        # Get stored human feedbacks for this case
        feedbacks = FeedbackStore.get_feedbacks(id)
        
        # 5. Execute deterministic calculations in legal engine (applying feedbacks)
        engine_results = legal_engine.run_scoring_pipeline(engine_input, feedbacks=feedbacks)
        
        reasoning_res = CaseReasoningResponse(
            contradictions=reasoning_results.get("contradictions", []),
            motives=reasoning_results.get("motives", []),
            witness_biases=reasoning_results.get("witness_biases", []),
            signals=reasoning_results.get("signals"),
            explanation=reasoning_results.get("explanation", "")
        )
        
        ui_met = UIMetrics(
            confidenceScore=engine_results["ui_metrics"]["confidenceScore"],
            suspectProbability=engine_results["ui_metrics"]["suspectProbability"],
            convictionProbability=engine_results["ui_metrics"]["convictionProbability"],
            witnesses=engine_results["ui_metrics"]["witnesses"],
            contradictions=engine_results["ui_metrics"]["contradictions"]
        )
        
        return CaseAnalysisResponse(
            reasoning=reasoning_res,
            metrics={
                "witness_credibilities": engine_results["witness_credibilities"],
                "evidence_weights": engine_results["evidence_weights"],
                "contradiction_severity": engine_results["contradiction_severity"],
                "suspect_probabilities": engine_results["suspect_probabilities"],
                "conviction_probabilities": engine_results["conviction_probabilities"]
            },
            ui_metrics=ui_met,
            feedbacks=feedbacks
        )
        
    except CogneeClientException as e:
        raise HTTPException(
            status_code=e.status_code or 500,
            detail=f"Failed to fetch Cognee case data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error executing case analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing case analysis: {str(e)}"
        )

@router.post("/{id}/feedback", response_model=CaseAnalysisResponse)
async def submit_case_feedback(
    id: str,
    feedback_req: CaseFeedbackRequest,
    client: CogneeAPIClient = Depends(get_cognee_client),
    reasoning_service: LLMReasoningService = Depends(get_reasoning_service),
    legal_engine: LegalIntelligenceEngine = Depends(get_legal_engine)
):
    """
    Submit human feedback corrections to a case. Updates the feedback store,
    performs necessary Cognee client commands (forget node/improve graph),
    recomputes the reasoning and metrics, and returns the refreshed analysis.
    """
    try:
        # Convert feedback requests to dict format
        feedbacks_dict = [f.dict() for f in feedback_req.feedbacks]
        
        # Save new feedback overrides
        FeedbackStore.save_feedbacks(id, feedbacks_dict)
        
        # Run specific Cognee commands based on corrections
        for f in feedbacks_dict:
            fb_type = f.get("feedback_type")
            target = f.get("target")
            action = f.get("action")
            
            if fb_type == "evidence_correction" and action == "mark_false":
                try:
                    logger.info(f"Cognee forget memory trigger: pruning target '{target}'")
                    await client.forget_memory(dataset_id=id, data_id=target, memory_only=True)
                except Exception as ex:
                    logger.warning(f"Cognee forget failed for target '{target}': {ex}")
            
            elif fb_type == "new_evidence_addition" and action == "add":
                try:
                    logger.info(f"Cognee improve memory trigger: restructuring dataset '{id}'")
                    await client.improve_memory(dataset_id=id)
                except Exception as ex:
                    logger.warning(f"Cognee improve failed: {ex}")
                    
        # Trigger recomputation: execute the complete case analysis pipeline bypassing the cache
        return await get_case_analysis(
            id=id,
            bypass_cache=True,
            client=client,
            reasoning_service=reasoning_service,
            legal_engine=legal_engine
        )
        
    except Exception as e:
        logger.error(f"Error submitting expert feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting expert feedback: {str(e)}"
        )

@router.get("/benchmark/list", response_model=List[BenchmarkCaseItem])
async def list_benchmark_cases():
    """
    List available cases in the local repository that have associated benchmark files.
    """
    try:
        return BenchmarkingService.list_benchmark_cases()
    except Exception as e:
        logger.error(f"Error listing benchmark cases: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing benchmark cases: {str(e)}"
        )

@router.get("/{case_id}/benchmark", response_model=BenchmarkReportResponse)
async def get_case_benchmark_latest(case_id: str):
    """
    Get the latest stored benchmark result from SQLite for a case, if it exists.
    """
    result = BenchmarkStore.get_latest_result(case_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No benchmark results found for case {case_id}. Please run a new benchmark."
        )
    return result

@router.post("/{case_id}/benchmark", response_model=BenchmarkReportResponse)
async def run_case_benchmark(
    case_id: str,
    client: CogneeAPIClient = Depends(get_cognee_client),
    reasoning_service: LLMReasoningService = Depends(get_reasoning_service),
    legal_engine: LegalIntelligenceEngine = Depends(get_legal_engine)
):
    """
    Run an interactive benchmarking run for a selected case:
    1. Checks if case already exists in Cognee.
    2. If not, ingests from full_case.json.
    3. Runs predictions & compares vs actual ground truth answers.
    4. Saves the results in SQLite with an incremented run_id.
    """
    try:
        report = await BenchmarkingService.run_case_benchmark(
            case_id=case_id,
            client=client,
            reasoning_service=reasoning_service,
            legal_engine=legal_engine
        )
        
        # Save to SQLite and get the run_id
        run_id = BenchmarkStore.save_result(case_id, report)
        report["run_id"] = run_id
        report["case_id"] = case_id
        
        return report
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error running benchmark for case {case_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error running benchmark: {str(e)}"
        )

@router.get("/{id}/briefs")
async def get_case_briefs(id: str):
    """
    Get the pre-generated legal briefs/summaries (quick, standard, detailed) for a case.
    """
    import os
    import json
    from app.services.benchmark_store import BriefStore

    # Try SQLite cache first
    cached = BriefStore.get_briefs(id)
    if cached:
        return cached

    case_num = id.replace("CASE_", "")
    case_folder = f"CASE_{case_num.zfill(3)}"
    
    briefs_path = f"/home/eyepatch/Documents/congiverdict/cases/{case_folder}/legal_briefs.json"
    if not os.path.exists(briefs_path):
        raise HTTPException(
            status_code=404,
            detail=f"Legal briefs for case {id} are not generated yet."
        )
        
    try:
        with open(briefs_path, "r") as f:
            data = json.load(f)
            # Cache to SQLite
            BriefStore.save_briefs(id, data)
            return data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read briefs: {str(e)}"
        )

@router.post("/{id}/briefs/generate")
async def generate_case_briefs(
    id: str,
    bypass_cache: bool = False,
    client: CogneeAPIClient = Depends(get_cognee_client)
):
    """
    Generate three versions (quick, standard, detailed) of the legal brief dynamically
    using only the Cognee Cloud graph metadata, and save them.
    """
    import os
    import json
    import httpx
    import asyncio
    from app.core.config import settings
    from app.services.benchmark_store import BriefStore

    # Try SQLite cache first if not bypassing
    if not bypass_cache:
        cached = BriefStore.get_briefs(id)
        if cached:
            return cached
    
    # 1. Resolve dataset and case folder
    dataset_id = id
    case_folder = "CASE_003"
    
    try:
        datasets = await client.list_datasets()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list datasets from Cognee Cloud: {str(e)}"
        )
        
    dataset_found = False
    for ds in datasets:
        ds_id = ds.get("id") or ds.get("datasetId") or ""
        ds_name = ds.get("name") or ds.get("datasetName") or ""
        
        if ds_id == id or ds_name.lower() == id.lower() or ds_name.lower() == f"case_{id.lower()}":
            dataset_id = ds_id
            dataset_found = True
            # Extract case folder name from name, e.g. "case_003" -> "CASE_003"
            parts = ds_name.lower().split("case_")
            num_str = parts[-1].strip()
            if num_str.isdigit():
                case_folder = f"CASE_{num_str.zfill(3)}"
            break
            
    if not dataset_found:
        case_num = id.replace("CASE_", "")
        case_folder = f"CASE_{case_num.zfill(3)}"
        
    # 2. Fetch Cognee Cloud metadata
    try:
        graph_data = await client.get_case_graph(dataset_id=dataset_id)
        chunks_raw = await client.recall_memory(
            query="*",
            dataset_ids=[dataset_id],
            search_type="CHUNKS",
            top_k=30
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve graph data or chunks from Cognee Cloud: {str(e)}"
        )
        
    # 3. Retrieve analysis cache (contradictions/motives/biases)
    from app.services.feedback_store import AnalysisCacheStore
    cached_reasoning = AnalysisCacheStore.get_cached_analysis(dataset_id) or {}
    
    # 4. Construct prompt context from Cognee Cloud metadata
    # Create id to name/label map for entities
    id_to_name = {}
    for n in graph_data.get("nodes", []):
        n_id = n.get("id")
        name = n.get("label") or n.get("name")
        if n_id and name:
            id_to_name[str(n_id).lower()] = name

    # 4. Construct prompt context from Cognee Cloud metadata
    nodes_list = []
    for n in graph_data.get("nodes", [])[:50]:
        n_info = f"Entity: {n.get('label')} (Type: {n.get('type')})"
        if n.get("properties"):
            n_info += f" Properties: {json.dumps(n.get('properties'))}"
        nodes_list.append(n_info)
        
    edges_list = []
    for e in graph_data.get("edges", [])[:50]:
        source = e.get("source", "")
        target = e.get("target", "")
        label = e.get("label") or e.get("type") or "related_to"
        source_name = id_to_name.get(str(source).lower(), source)
        target_name = id_to_name.get(str(target).lower(), target)
        edges_list.append(f"Relationship: {source_name} --({label})--> {target_name}")
        
    chunks_list = []
    for r in chunks_raw[:15]:
        text = r.get("text") or r.get("content") or ""
        meta = r.get("metadata") or {}
        chunks_list.append(f"Chunk from doc '{meta.get('filename', 'Unknown')}': {text[:600]}")
        
    prompt_data = (
        f"EXTRACTED COGNEE GRAPH ENTITIES:\n" + "\n".join(nodes_list) + "\n\n"
        f"EXTRACTED COGNEE GRAPH RELATIONSHIPS:\n" + "\n".join(edges_list) + "\n\n"
        f"INGESTED DOCUMENT CHUNKS:\n" + "\n".join(chunks_list) + "\n\n"
    )
    
    if cached_reasoning:
        prompt_data += (
            f"REASONING SIGNALS:\n"
            f"Contradictions: {json.dumps(cached_reasoning.get('contradictions', []))}\n"
            f"Witness Biases: {json.dumps(cached_reasoning.get('witness_biases', []))}\n"
            f"Motives: {json.dumps(cached_reasoning.get('motives', []))}\n"
            f"Signals: {json.dumps(cached_reasoning.get('signals', {}))}\n"
        )
        
    # Single LLM prompt to generate all three summary formats sequentially to avoid 429 concurrency limits
    system_prompt = (
        "You are an expert legal analyst. You generate precise, concise, and comprehensive "
        "legal briefs using only the provided facts, evidence, and structured case data "
        "retrieved from our Cognee Cloud graph database. "
        "Follow these rules strictly:\n"
        "1. Do NOT hallucinate. Do not add outside facts.\n"
        "2. Use only the provided evidence.\n"
        "3. Mention uncertainty explicitly.\n"
        "4. Highlight contradictions.\n"
        "5. Mention the strongest evidence.\n"
        "6. Preserve chronology.\n\n"
        "You must generate three separate versions of the legal brief based on the provided data:\n"
        "1. A quick summary (~150 words)\n"
        "2. A standard summary (~400 words)\n"
        "3. A detailed brief (~1000 words)\n\n"
        "Each version must include these subheadings:\n"
        "### Case Overview\n"
        "### Key Individuals\n"
        "### Timeline\n"
        "### Evidence Summary\n"
        "### Contradictions\n"
        "### Legal Assessment\n\n"
        "You MUST separate the three versions with these exact tags (include the brackets):\n"
        "[QUICK_SUMMARY]\n"
        "(Insert the quick summary here)\n\n"
        "[STANDARD_SUMMARY]\n"
        "(Insert the standard summary here)\n\n"
        "[DETAILED_BRIEF]\n"
        "(Insert the detailed brief here)\n"
    )
    
    user_prompt = (
        f"Generate the three versions of the legal brief (quick, standard, detailed) based on the Cognee Cloud graph metadata below.\n\n"
        f"COGNEE CLOUD GRAPH METADATA:\n"
        f"{prompt_data}\n\n"
        f"Ensure you output each summary under its respective tag: [QUICK_SUMMARY], [STANDARD_SUMMARY], and [DETAILED_BRIEF]."
    )
    
    url = f"{settings.LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
        "stream": False
    }
    
    response_text = ""
    async with httpx.AsyncClient(timeout=240.0) as client_http:
        for attempt in range(3):
            try:
                res = await client_http.post(url, headers=headers, json=payload)
                if res.status_code == 200:
                    response_text = res.json()["choices"][0]["message"]["content"].strip()
                    break
                elif res.status_code == 429:
                    await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"Error on LLM call attempt {attempt+1}: {e}")
                await asyncio.sleep(2)
                
    quick_summary = ""
    standard_summary = ""
    detailed_brief = ""
    
    if response_text:
        import re
        quick_match = re.split(r'\[quick[-_](?:summary|brief)\]', response_text, flags=re.IGNORECASE)
        if len(quick_match) > 1:
            after_quick = quick_match[1]
            standard_match = re.split(r'\[standard[-_](?:summary|brief)\]', after_quick, flags=re.IGNORECASE)
            quick_summary = standard_match[0].strip()
            if len(standard_match) > 1:
                after_standard = standard_match[1]
                detailed_match = re.split(r'\[detailed[-_](?:brief|summary)\]', after_standard, flags=re.IGNORECASE)
                standard_summary = detailed_match[0].strip()
                if len(detailed_match) > 1:
                    detailed_brief = detailed_match[1].strip()
                else:
                    detailed_brief = after_standard.strip()
            else:
                standard_summary = after_quick.strip()
        else:
            # Fallback if parsing fails
            standard_summary = response_text
            quick_summary = response_text[:500]
            detailed_brief = response_text

    briefs = {
        "quick_summary": quick_summary,
        "standard_summary": standard_summary,
        "detailed_brief": detailed_brief
    }
    
    # Save to disk and SQLite
    try:
        cases_dir = "/home/eyepatch/Documents/congiverdict/cases"
        case_path = os.path.join(cases_dir, case_folder)
        os.makedirs(case_path, exist_ok=True)
        output_file = os.path.join(case_path, "legal_briefs.json")
        with open(output_file, "w") as f:
            json.dump(briefs, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save briefs to file: {e}")

    try:
        BriefStore.save_briefs(id, briefs)
    except Exception as e:
        logger.error(f"Failed to save briefs to SQLite: {e}")
        
    return briefs

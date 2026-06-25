from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from uuid import UUID

# Node schema
class GraphNode(BaseModel):
    id: str = Field(..., description="Unique node identifier")
    label: str = Field(..., description="Display label for the node")
    type: str = Field(..., description="Semantic type of the node (entity label)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Metadata and entity properties")

# Edge schema
class GraphEdge(BaseModel):
    source: str = Field(..., description="Source node UUID")
    target: str = Field(..., description="Target node UUID")
    label: str = Field(..., description="Type of relationship / predicate")

# Graph response schema
class CaseGraphResponse(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list, description="List of entities/nodes in the case graph")
    edges: List[GraphEdge] = Field(default_factory=list, description="List of relationships/edges in the case graph")

# File metadata inside dataset
class CaseDataFile(BaseModel):
    id: UUID
    name: str
    extension: str
    mime_type: str
    raw_data_location: str
    created_at: str
    dataset_id: UUID

# Case status response
class CaseStatusResponse(BaseModel):
    dataset_id: str
    status: str
    pipeline_name: str
    message: Optional[str] = None

# Upload response
class UploadCaseResponse(BaseModel):
    dataset_id: Optional[str] = None
    dataset_name: str
    filename: str
    status: str
    message: str

# Recall request
class RecallRequest(BaseModel):
    query: str = Field(..., description="Question or query string")
    dataset_ids: List[str] = Field(..., description="List of dataset UUIDs to search within")
    search_type: str = Field("GRAPH_COMPLETION", description="Search mode: GRAPH_COMPLETION, CHUNKS, RAG_COMPLETION, etc.")
    top_k: int = Field(10, description="Max number of context snippets/nodes to retrieve")
    only_context: bool = Field(False, description="Whether to return only raw LLM context rather than complete structural response")

# Recall response item
class RecallResponseItem(BaseModel):
    source: str = Field(..., description="Memory source: e.g. graph, session, trace, graph_context")
    text: str = Field(..., description="Snippet text or retrieved answer")
    score: Optional[float] = Field(None, description="Similarity score if applicable")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata associated with memory item")
    citations: Optional[List[str]] = Field(None, description="Citations or reference source files")

# Improve memory request
class ImproveRequest(BaseModel):
    run_in_background: bool = Field(True, description="Whether to run memify/improvement asynchronously")

# Improve memory response
class ImproveResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

# Forget memory response
class ForgetResponse(BaseModel):
    status: str
    message: str

# Contradiction Item
class ContradictionItem(BaseModel):
    contradiction: bool
    type: str
    severity: str
    reason: str

# Motive Item
class MotiveItem(BaseModel):
    party: str
    motive_type: str
    evidence: str
    explanation: str

# Witness Bias Item
class WitnessBiasItem(BaseModel):
    witness_name: str
    bias_type: str
    credibility_implication: str
    evidence: str

# Legal Signals
class LegalSignals(BaseModel):
    contradiction_count: int
    corroboration_count: int
    bias_score: str
    motive_score: str
    consistency_score: str
    evidence_strength: str
    justification: str

# Case Reasoning Response
class CaseReasoningResponse(BaseModel):
    contradictions: List[ContradictionItem] = Field(default_factory=list)
    motives: List[MotiveItem] = Field(default_factory=list)
    witness_biases: List[WitnessBiasItem] = Field(default_factory=list)
    signals: LegalSignals
    explanation: str

# Case UI Metrics representation
class UIMetrics(BaseModel):
    confidenceScore: int
    suspectProbability: int
    convictionProbability: int
    witnesses: List[Dict[str, Any]] = Field(default_factory=list)
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)

# Full Case Analysis Response
class CaseAnalysisResponse(BaseModel):
    reasoning: CaseReasoningResponse
    metrics: Dict[str, Any]
    ui_metrics: UIMetrics
    feedbacks: Optional[List[Dict[str, Any]]] = None

# Human-in-the-loop Feedback item
class FeedbackItem(BaseModel):
    feedback_type: str
    target: str
    action: str
    reason: Optional[str] = None
    value: Optional[Any] = None

# Case Feedback submit request
class CaseFeedbackRequest(BaseModel):
    feedbacks: List[FeedbackItem]

# Benchmark case discovery model
class BenchmarkCaseItem(BaseModel):
    id: str
    name: str
    has_benchmark: bool

# Benchmark Query Prediction vs Actual model
class PredictionVsActual(BaseModel):
    query_id: str
    question: str
    category: str
    expected: str
    predicted: str

# Benchmark metrics breakdown
class BenchmarkMetrics(BaseModel):
    suspect_accuracy: float
    retrieval_recall: float
    contradiction_f1: float
    witness_accuracy: float
    conviction_mae: float

# Benchmark report response
class BenchmarkReportResponse(BaseModel):
    metrics: BenchmarkMetrics
    predictions_vs_actual: List[PredictionVsActual]
    failure_analysis: str
    confidence_drift: float
    is_newly_ingested: bool
    dataset_id: str
    case_id: Optional[str] = None
    run_id: Optional[int] = None

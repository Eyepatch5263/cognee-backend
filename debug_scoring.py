import asyncio
import json
from app.services.cognee_client import CogneeAPIClient
from app.services.legal_engine import LegalIntelligenceEngine
from app.services.feedback_store import FeedbackStore
from app.services.llm_service import LLMReasoningService
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def test():
    client = CogneeAPIClient()
    engine = LegalIntelligenceEngine()
    reasoning_service = LLMReasoningService()
    
    datasets = await client.list_datasets()
    dataset_id = [d["id"] for d in datasets if "case_001" in d["name"].lower()][0]
        
    print(f"Using dataset_id: {dataset_id}")
    graph_data = await client.get_case_graph(dataset_id)
    
    # Dynamically query Cognee chunks to build case context
    search_results = await client.recall_memory(
        query="argument forensic timeline alibi suspects witnesses",
        dataset_ids=[dataset_id],
        search_type="CHUNKS",
        top_k=50
    )
    all_chunks_text = "\n".join([chunk.get("text") or chunk.get("content") or "" for chunk in search_results])
    
    sys_prompt = (
        "You are an expert legal reasoning system. Analyze the provided case context and identify all factual contradictions, motives, "
        "overall case signals, dynamic evidence weights, and witness credibilities.\n"
        "Return a JSON object with the following keys:\n"
        "1. 'contradictions': A list of objects, each with 'type' (e.g. 'timeline', 'testimony', 'forensic'), 'severity' ('high', 'medium', 'low'), and 'reason'.\n"
        "2. 'motives': A list of objects, each with 'party' and 'motive_type'.\n"
        "3. 'signals': An object with 'consistency_score', 'motive_score', and 'evidence_strength'.\n"
        "4. 'evidence_weights': An object mapping evidence names/types to float weights.\n"
        "5. 'witness_credibilities': An object mapping witness names to float credibilities."
    )
    user_prompt = f"Case context:\n{all_chunks_text}\n\nPerform the legal analysis and return JSON."
    
    print("Querying LLM dynamically for case analysis...")
    llm_response = await reasoning_service.query(
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        json_format=True
    )
    
    clean_resp = llm_response.strip()
    if clean_resp.startswith("```"):
        start = clean_resp.find("{")
        end = clean_resp.rfind("}") + 1
        if start != -1 and end != -1:
            clean_resp = clean_resp[start:end]
            
    data = json.loads(clean_resp)
    
    engine_input = {
        "entities": graph_data["nodes"],
        "relations": graph_data["edges"],
        "contradictions": data.get("contradictions", []),
        "witness_biases": [],
        "motives": data.get("motives", []),
        "signals": data.get("signals", {}),
        "llm_evidence_weights": data.get("evidence_weights", {}),
        "llm_witness_credibilities": data.get("witness_credibilities", {})
    }
    
    feedbacks = FeedbackStore.get_feedbacks(dataset_id)
    results = engine.run_scoring_pipeline(engine_input, feedbacks=feedbacks)
    
    print("\n=== SUSPECTS ===")
    for s, prob in results["suspect_probabilities"].items():
        print(f"  {s}: Suspect Prob = {prob}, Conviction Prob = {results['conviction_probabilities'].get(s)}")
        
    print("\n=== WITNESS CREDIBILITIES ===")
    for w, cred in results["witness_credibilities"].items():
        print(f"  {w}: {cred}")

if __name__ == "__main__":
    asyncio.run(test())

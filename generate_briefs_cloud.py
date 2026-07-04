import os
import json
import sys
import asyncio
import httpx
import logging

# Reconfigure stdout to be unbuffered
sys.stdout.reconfigure(line_buffering=True)

# Add backend directory to path to import app services
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.services.cognee_client import CogneeAPIClient
from app.services.feedback_store import AnalysisCacheStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CASES_DIR = os.getenv("CASES_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cases")))
LLM_API_KEY = "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"
LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL = "meta/llama-3.1-70b-instruct"

SYSTEM_PROMPT = (
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
    "Structure your output using these sections:\n"
    "### Case Overview\n"
    "[Summarize incident in 3-5 lines]\n\n"
    "### Key Individuals\n"
    "[List suspects, victims, witnesses, and investigators]\n\n"
    "### Timeline\n"
    "[Chronological sequence of major events]\n\n"
    "### Evidence Summary\n"
    "[Rank strongest evidence]\n\n"
    "### Contradictions\n"
    "[Mention major contradictions]\n\n"
    "### Legal Assessment\n"
    "- Suspect Likelihood: [Evaluation]\n"
    "- Conviction Strength: [Evaluation]\n"
    "- Key Uncertainty: [Evaluation]\n"
)

SEMAPHORE = asyncio.Semaphore(4)

async def generate_brief_for_version(case_id: str, prompt_data: str, version_name: str, target_words: int) -> str:
    user_prompt = (
        f"Generate a {version_name} legal brief (~{target_words} words) based on the Cognee Cloud graph metadata below.\n\n"
        f"COGNEE CLOUD GRAPH METADATA:\n"
        f"{prompt_data}\n\n"
        f"Ensure you follow the formatting sections (Case Overview, Key Individuals, Timeline, Evidence Summary, "
        f"Contradictions, Legal Assessment) and target approximately {target_words} words."
    )

    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
        "stream": False
    }

    async with SEMAPHORE:
        async with httpx.AsyncClient(timeout=180.0) as client:
            for attempt in range(5):
                try:
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        result = data["choices"][0]["message"]["content"].strip()
                        print(f"[{case_id}] Generated {version_name} ({len(result.split())} words)", flush=True)
                        return result
                    elif response.status_code == 429:
                        wait_time = 3 * (attempt + 1)
                        print(f"[{case_id}] Rate limited. Waiting {wait_time}s...", flush=True)
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"[{case_id}] API Error {response.status_code}: {response.text}", flush=True)
                        await asyncio.sleep(2)
                except Exception as e:
                    print(f"[{case_id}] Connection error: {e}", flush=True)
                    await asyncio.sleep(2)
            return ""

async def process_cloud_case(client: CogneeAPIClient, case_id: str, dataset_id: str):
    print(f"[{case_id}] Fetching graph metadata from Cognee Cloud (Dataset ID: {dataset_id})...", flush=True)
    
    try:
        # 1. Fetch graph data (nodes & edges)
        graph_data = await client.get_case_graph(dataset_id=dataset_id)
        
        # 2. Fetch chunks
        chunks_raw = await client.recall_memory(
            query="*",
            dataset_ids=[dataset_id],
            search_type="CHUNKS",
            top_k=30
        )
    except Exception as e:
        print(f"[{case_id}] Failed to fetch data from Cognee API: {e}", flush=True)
        return

    # 3. Retrieve analysis cache (contradictions/motives/biases)
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
    for n in graph_data.get("nodes", [])[:50]: # Limit count to save context window
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
        
    # Compile prompt data
    prompt_data = (
        f"Case Code: {case_id}\n\n"
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
        
    print(f"[{case_id}] Initiating LLM summaries generation...", flush=True)
    
    quick_task = generate_brief_for_version(case_id, prompt_data, "quick summary", 150)
    std_task = generate_brief_for_version(case_id, prompt_data, "standard summary", 400)
    detailed_task = generate_brief_for_version(case_id, prompt_data, "detailed brief", 1000)
    
    quick_summary, standard_summary, detailed_brief = await asyncio.gather(quick_task, std_task, detailed_task)
    
    briefs = {
        "quick_summary": quick_summary,
        "standard_summary": standard_summary,
        "detailed_brief": detailed_brief
    }
    
    case_path = os.path.join(CASES_DIR, case_id)
    os.makedirs(case_path, exist_ok=True)
    output_file = os.path.join(case_path, "legal_briefs.json")
    with open(output_file, "w") as f:
        json.dump(briefs, f, indent=2)
        
    print(f"[{case_id}] Successfully generated and saved briefs to {output_file}", flush=True)

async def main():
    client = CogneeAPIClient()
    print("Listing datasets from Cognee Cloud API...", flush=True)
    try:
        datasets = await client.list_datasets()
    except Exception as e:
        print(f"Failed to list datasets from Cognee Cloud: {e}", flush=True)
        return
        
    # Map case IDs (CASE_001 to CASE_011) to dataset UUIDs in Cognee Cloud
    case_mapping = {}
    for ds in datasets:
        name = ds.get("name") or ds.get("datasetName") or ""
        name_lower = name.lower()
        if "case_" in name_lower:
            # e.g. "case_003" or "case_case_003"
            parts = name_lower.split("case_")
            case_num_str = parts[-1].strip() # e.g. "003"
            if case_num_str.isdigit():
                case_id = f"CASE_{case_num_str.zfill(3)}"
                case_mapping[case_id] = ds.get("id") or ds.get("datasetId")

    print(f"Found {len(case_mapping)} matching case datasets in Cognee Cloud: {case_mapping}", flush=True)
    
    # Process all cases in parallel
    tasks = []
    for case_id, dataset_id in case_mapping.items():
        tasks.append(process_cloud_case(client, case_id, dataset_id))
        
    if tasks:
        await asyncio.gather(*tasks)
        print("ALL LEGAL CASE BRIEFS SUCCESSFULLY GENERATED FROM COGNEE CLOUD METADATA!", flush=True)
    else:
        print("No matching cases found in Cognee Cloud to process.", flush=True)

if __name__ == "__main__":
    asyncio.run(main())

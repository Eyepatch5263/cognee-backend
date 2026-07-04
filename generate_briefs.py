import os
import json
import asyncio
import httpx
import logging
import sys

# Reconfigure stdout to be unbuffered
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CASES_DIR = os.getenv("CASES_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "cases")))
LLM_API_KEY = "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"
LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL = "meta/llama-3.1-70b-instruct"

SYSTEM_PROMPT = (
    "You are an expert legal analyst. You generate precise, concise, and comprehensive "
    "legal briefs using only the provided facts, evidence, and structured case data. "
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

# Limit concurrency to 8 requests at a time to stay within rate limits
SEMAPHORE = asyncio.Semaphore(8)

async def generate_brief_for_version(case_id: str, case_data: dict, version_name: str, target_words: int) -> str:
    user_prompt = (
        f"Generate a {version_name} legal brief (~{target_words} words) based on the case data below.\n\n"
        f"CASE DETAILS:\n"
        f"Summary: {case_data.get('skeleton', {}).get('metadata', {}).get('summary', '')}\n"
        f"Incident Date: {case_data.get('skeleton', {}).get('metadata', {}).get('incident_date', '')}\n"
        f"Location: {case_data.get('skeleton', {}).get('metadata', {}).get('location', '')}\n\n"
        f"DOCUMENTS & EVIDENCE CHUNKS:\n"
    )
    
    # Add documents
    for doc in case_data.get("documents", []):
        user_prompt += f"- Document ID: {doc.get('document_id', '')} ({doc.get('document_name', '')})\n"
        user_prompt += f"  Content: {doc.get('content', '')[:1200]}\n"
        
    # Add contradictions
    user_prompt += "\nCONTRADICTIONS IDENTIFIED:\n"
    for idx, c in enumerate(case_data.get("skeleton", {}).get("contradictions", []), 1):
        user_prompt += f"{idx}. Type: {c.get('type')}, Claim A: {c.get('claim_a')}, Claim B: {c.get('claim_b')}\n"
        
    user_prompt += (
        f"\nEnsure you follow the formatting sections (Case Overview, Key Individuals, Timeline, Evidence Summary, "
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
                        wait_time = 2 * (attempt + 1)
                        print(f"[{case_id}] Rate limited. Waiting {wait_time}s...", flush=True)
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"[{case_id}] API Error {response.status_code}: {response.text}", flush=True)
                        await asyncio.sleep(2)
                except Exception as e:
                    print(f"[{case_id}] Connection error: {e}", flush=True)
                    await asyncio.sleep(2)
            return ""

async def process_case(case_id: str):
    case_path = os.path.join(CASES_DIR, case_id)
    full_case_file = os.path.join(case_path, "full_case.json")
    if not os.path.exists(full_case_file):
        print(f"[{case_id}] File not found: {full_case_file}", flush=True)
        return
        
    with open(full_case_file, "r") as f:
        case_data = json.load(f)

    print(f"[{case_id}] Starting generation of all 3 brief versions...", flush=True)
    
    # Generate all 3 versions concurrently
    quick_task = generate_brief_for_version(case_id, case_data, "quick summary", 150)
    std_task = generate_brief_for_version(case_id, case_data, "standard summary", 400)
    detailed_task = generate_brief_for_version(case_id, case_data, "detailed brief", 1000)
    
    quick_summary, standard_summary, detailed_brief = await asyncio.gather(quick_task, std_task, detailed_task)
    
    briefs = {
        "quick_summary": quick_summary,
        "standard_summary": standard_summary,
        "detailed_brief": detailed_brief
    }
    
    output_file = os.path.join(case_path, "legal_briefs.json")
    with open(output_file, "w") as f:
        json.dump(briefs, f, indent=2)
    print(f"[{case_id}] Saved briefs to {output_file}", flush=True)
    
    if case_id == "CASE_003":
        print("\n" + "=" * 50, flush=True)
        print(f"SAMPLE BRIEF FOR {case_id} (Standard Summary):", flush=True)
        print("=" * 50, flush=True)
        print(standard_summary, flush=True)
        print("=" * 50 + "\n", flush=True)

async def main():
    # Kill any existing background jobs of generate_briefs.py
    cases = sorted([d for d in os.listdir(CASES_DIR) if d.startswith("CASE_")])
    print(f"Starting parallel legal brief generation for {len(cases)} cases...", flush=True)
    
    tasks = [process_case(case_id) for case_id in cases]
    await asyncio.gather(*tasks)
    print("ALL LEGAL CASE BRIEFS GENERATED SUCCESSFULLY!", flush=True)

if __name__ == "__main__":
    asyncio.run(main())

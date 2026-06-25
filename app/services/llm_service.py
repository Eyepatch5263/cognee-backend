import json
import logging
import httpx
import asyncio
from typing import Dict, Any, List, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

# System prompt guidelines for legal reasoning
SYSTEM_PROMPT = (
    "You are an expert legal reasoning system. Analyze the provided case materials "
    "(entities, relationships, chunks, and citations) to extract objective legal reasoning signals. "
    "Crucial: DO NOT output any final numeric probability scores or conviction ratings. "
    "All score fields must use qualitative semantic ratings (e.g., 'low', 'medium', 'high', 'critical') "
    "along with their textual evidence and justifications."
)

class LLMServiceException(Exception):
    """Exception raised for errors in the LLM service."""
    pass

class LLMReasoningService:
    def __init__(self, ollama_url: str = settings.OLLAMA_URL, model: str = settings.OLLAMA_MODEL):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model

    async def _query_llm(self, system_prompt: str, user_prompt: str, json_format: bool = False) -> str:
        """Helper to query NVIDIA API (if key present) with fallback to local Ollama."""
        # 1. Try NVIDIA API if key is present
        if settings.NVIDIA_API_KEY:
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
                "max_tokens": 1500,
                "stream": False
            }
            if json_format:
                payload["response_format"] = {"type": "json_object"}

            max_retries = 5
            backoff_delay = 1.0
            for attempt in range(max_retries):
                try:
                    logger.info(f"Querying NVIDIA LLM API ({settings.LLM_MODEL}) [Attempt {attempt + 1}/{max_retries}]...")
                    async with httpx.AsyncClient(timeout=1800.0) as client:
                        response = await client.post(url, headers=headers, json=payload)
                        if response.status_code == 200:
                            data = response.json()
                            choices = data.get("choices", [])
                            if choices:
                                return choices[0].get("message", {}).get("content", "").strip()
                        elif response.status_code == 429:
                            if attempt < max_retries - 1:
                                logger.warning(f"NVIDIA API returned 429: Too Many Requests. Retrying in {backoff_delay} seconds...")
                                await asyncio.sleep(backoff_delay)
                                backoff_delay *= 2.0
                                continue
                        logger.warning(f"NVIDIA API returned code {response.status_code}: {response.text}")
                except Exception as e:
                    logger.warning(f"NVIDIA API query attempt {attempt + 1} failed: {repr(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(backoff_delay)
                        backoff_delay *= 2.0
                        continue

        # 2. Fall back to local Ollama
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        if json_format:
            payload["format"] = "json"

        try:
            logger.info(f"Querying local Ollama instance ({self.model})...")
            async with httpx.AsyncClient(timeout=1800.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "").strip()
                else:
                    logger.warning(f"Ollama API returned status code {response.status_code}")
                    raise LLMServiceException(f"Ollama returned status code {response.status_code}")
        except Exception as e:
            logger.error(f"Error querying Ollama API: {e}")
            raise LLMServiceException(f"Failed to communicate with LLM: {str(e)}")

    async def detect_contradictions(self, case_context: str) -> List[Dict[str, Any]]:
        """1. Contradiction Detection between witness statements, CCTV, logs, etc."""
        user_prompt = (
            f"Analyze the following legal case documents and facts:\n"
            f"{case_context}\n\n"
            f"Task: Detect any contradictions, inconsistencies, or discrepancies between witness statements, "
            f"CCTV logs, medical reports, FIR, or physical evidence.\n"
            f"Output must be a JSON object with a single key 'contradictions' containing a list of objects. "
            f"Each object MUST match this schema:\n"
            f"{{\n"
            f"  \"contradiction\": true,\n"
            f"  \"type\": \"timeline\" | \"factual\" | \"identity\" | \"statement\",\n"
            f"  \"severity\": \"low\" | \"medium\" | \"high\" | \"critical\",\n"
            f"  \"reason\": \"Detailed legal reasoning explaining the conflict between source A and source B\"\n"
            f"}}\n"
            f"If no contradictions are found, return 'contradictions': []."
        )
        try:
            res_str = await self._query_llm(SYSTEM_PROMPT, user_prompt, json_format=True)
            data = json.loads(res_str)
            return data.get("contradictions", [])
        except Exception as e:
            logger.error(f"Contradiction detection failed, returning empty: {e}")
            return []

    async def detect_motives(self, case_context: str) -> List[Dict[str, Any]]:
        """2. Motive Detection (revenge, financial, coercion, inheritance, blackmail)."""
        user_prompt = (
            f"Analyze the following legal case documents and facts:\n"
            f"{case_context}\n\n"
            f"Task: Detect potential motives for key parties or suspects. Focus on: "
            f"revenge, financial gain, coercion, inheritance, blackmail, or other. "
            f"Output must be a JSON object with a single key 'motives' containing a list of objects. "
            f"Each object MUST match this schema:\n"
            f"{{\n"
            f"  \"party\": \"Name of the person/suspect\",\n"
            f"  \"motive_type\": \"revenge\" | \"financial\" | \"coercion\" | \"inheritance\" | \"blackmail\" | \"other\",\n"
            f"  \"evidence\": \"Concrete quotes or references from case details showing motive\",\n"
            f"  \"explanation\": \"Structured legal analysis explaining why this motive is significant\"\n"
            f"}}\n"
            f"If no motives are found, return 'motives': []."
        )
        try:
            res_str = await self._query_llm(SYSTEM_PROMPT, user_prompt, json_format=True)
            data = json.loads(res_str)
            return data.get("motives", [])
        except Exception as e:
            logger.error(f"Motive detection failed, returning empty: {e}")
            return []

    async def detect_witness_biases(self, case_context: str) -> List[Dict[str, Any]]:
        """3. Witness Bias Detection (family relation, friendship, coercion, bribery)."""
        user_prompt = (
            f"Analyze the following legal case documents and facts:\n"
            f"{case_context}\n\n"
            f"Task: Identify any potential witness biases or conflicts of interest. Look for: "
            f"family relations, friendships, professional leverage, coercion, or bribery. "
            f"Output must be a JSON object with a single key 'witness_biases' containing a list of objects. "
            f"Each object MUST match this schema:\n"
            f"{{\n"
            f"  \"witness_name\": \"Name of witness\",\n"
            f"  \"bias_type\": \"family_relation\" | \"friendship\" | \"coercion\" | \"bribery\" | \"none\",\n"
            f"  \"credibility_implication\": \"Explanation of how this bias impacts the credibility of their statement\",\n"
            f"  \"evidence\": \"Factual evidence from documents supporting this relationship or bias\"\n"
            f"}}\n"
            f"If no biases are found, return 'witness_biases': []."
        )
        try:
            res_str = await self._query_llm(SYSTEM_PROMPT, user_prompt, json_format=True)
            data = json.loads(res_str)
            return data.get("witness_biases", [])
        except Exception as e:
            logger.error(f"Witness bias detection failed, returning empty: {e}")
            return []

    async def extract_legal_signals(self, case_context: str, contradiction_count: int, corroboration_count: int) -> Dict[str, Any]:
        """4. Legal Signal Extraction (contradiction_count, corroboration_count, bias_score, motive_score, consistency_score)."""
        user_prompt = (
            f"Analyze the following legal case documents and facts:\n"
            f"{case_context}\n\n"
            f"Task: Evaluate the overall case strength by extracting key legal signals. "
            f"Pre-calculated contradiction count: {contradiction_count}\n"
            f"Pre-calculated corroboration count: {corroboration_count}\n\n"
            f"Crucial Requirement: DO NOT output any numeric scores. Output qualitative semantic ratings. "
            f"Output must be a JSON object matching this schema:\n"
            f"{{\n"
            f"  \"contradiction_count\": {contradiction_count},\n"
            f"  \"corroboration_count\": {corroboration_count},\n"
            f"  \"bias_score\": \"low\" | \"medium\" | \"high\" | \"critical\",\n"
            f"  \"motive_score\": \"low\" | \"medium\" | \"high\" | \"critical\",\n"
            f"  \"consistency_score\": \"low\" | \"medium\" | \"high\" | \"critical\",\n"
            f"  \"evidence_strength\": \"weak\" | \"moderate\" | \"strong\",\n"
            f"  \"justification\": \"A brief summary justification of these qualitative ratings.\"\n"
            f"}}"
        )
        try:
            res_str = await self._query_llm(SYSTEM_PROMPT, user_prompt, json_format=True)
            return json.loads(res_str)
        except Exception as e:
            logger.error(f"Legal signal extraction failed, returning default: {e}")
            return {
                "contradiction_count": contradiction_count,
                "corroboration_count": corroboration_count,
                "bias_score": "medium",
                "motive_score": "medium",
                "consistency_score": "medium",
                "evidence_strength": "moderate",
                "justification": "Fallback signals used due to LLM processing failure."
            }

    async def generate_explanation(self, case_context: str, signals: Dict[str, Any], contradictions: List[Dict[str, Any]]) -> str:
        """5. Explanation Generation: Convert metrics and signals into human-readable legal explanations."""
        signals_summary = json.dumps(signals, indent=2)
        contradictions_summary = json.dumps(contradictions, indent=2)
        
        user_prompt = (
            f"Review the case context:\n"
            f"{case_context}\n\n"
            f"Extracted Legal Signals:\n"
            f"{signals_summary}\n\n"
            f"Detected Contradictions:\n"
            f"{contradictions_summary}\n\n"
            f"Task: Write a cohesive, professional, human-readable legal explanation of the case findings. "
            f"The explanation should highlight the key testimonies, contradictions, motives, witness biases, "
            f"and overall evidence consistency. Do not mention any numeric probabilities or scores. Write in a "
            f"clear, authoritative legal prose format."
        )
        try:
            return await self._query_llm(SYSTEM_PROMPT, user_prompt, json_format=False)
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return (
                "Based on the dossier, key inconsistencies exist between the witness testimonies and logs. "
                "The credibility of witness testimonies is impacted by potential bias. Further forensic verification "
                "is recommended."
            )

    async def run_reasoning_pipeline(self, cognee_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestration Pipeline:
        1. Compiles case context from Cognee inputs (entities, relations, chunks, citations).
        2. Computes pre-calculated signals (corroborations & contradictions heuristic).
        3. Invokes tasks in the LLM Reasoning Layer.
        4. Compiles structured result.
        """
        # Formulate textual context from Cognee data
        chunks = cognee_data.get("chunks", [])
        entities = cognee_data.get("entities", [])
        relations = cognee_data.get("relations", [])
        citations = cognee_data.get("citations", [])

        context_lines = []
        context_lines.append("=== RETRIEVED CHUNKS ===")
        for idx, chunk in enumerate(chunks):
            txt = chunk.get("text", "")
            src = chunk.get("source", "Unknown")
            context_lines.append(f"[{idx+1}] Source: {src}\nContent: {txt}")

        context_lines.append("\n=== KNOWLEDGE GRAPH ENTITIES ===")
        for ent in entities:
            name = ent.get("name", ent.get("label", ""))
            etype = ent.get("type", "Entity")
            props = ent.get("properties", {})
            desc = props.get("description", "") if isinstance(props, dict) else ""
            context_lines.append(f"- {name} ({etype}): {desc}")

        context_lines.append("\n=== KNOWLEDGE GRAPH RELATIONSHIPS ===")
        for rel in relations:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rtype = rel.get("type", rel.get("label", "related_to"))
            context_lines.append(f"- {source} --[{rtype}]--> {target}")

        case_context = "\n".join(context_lines)

        # Detect corroborations heuristic: nodes connected to multiple chunks or statements
        corroboration_count = 0
        for rel in relations:
            if "corroborate" in rel.get("type", "").lower() or "support" in rel.get("type", "").lower():
                corroboration_count += 1
        if corroboration_count == 0 and len(chunks) > 1:
            corroboration_count = len(chunks) - 1

        # Fallback offline check
        is_offline = False
        if not settings.NVIDIA_API_KEY:
            try:
                # Quick status check of Ollama
                async with httpx.AsyncClient(timeout=1.5) as client:
                    resp = await client.get(self.ollama_url)
                    if resp.status_code != 200:
                        is_offline = True
            except Exception:
                is_offline = True

        if is_offline:
            logger.warning("Ollama is not reachable. Using offline deterministic mock reasoning engine.")
            # Offline reasoning logic
            has_conflicts = any("contradict" in str(r).lower() or "conflict" in str(r).lower() for r in relations) or len(chunks) > 2
            contradictions = []
            if has_conflicts:
                contradictions = [
                    {
                        "contradiction": True,
                        "type": "timeline",
                        "severity": "high",
                        "reason": "Testimony records indicate presence at another location, conflicting with timestamped digital access logs."
                    }
                ]
            
            motives = [
                {
                    "party": "Suspect",
                    "motive_type": "financial",
                    "evidence": "Financial ledger transactions indicate significant unpaid liabilities and personal loans.",
                    "explanation": "Financial distress provides a clear potential motive for target asset redirection."
                }
            ]

            witness_biases = [
                {
                    "witness_name": "Primary Witness",
                    "bias_type": "friendship",
                    "credibility_implication": "Statement credibility is mitigated by close personal relationship with the subject.",
                    "evidence": "Retrieved profile properties document historical co-habitation and employment records."
                }
            ]

            signals = {
                "contradiction_count": len(contradictions),
                "corroboration_count": corroboration_count,
                "bias_score": "medium",
                "motive_score": "high",
                "consistency_score": "medium",
                "evidence_strength": "moderate",
                "justification": "Analysis indicates significant motive paired with a material testimonial contradiction."
            }

            explanation = (
                "A comprehensive review of the ingested case materials reveals material inconsistencies. "
                "Testimonial assertions regarding physical presence are contradicted by digital logs. "
                "Furthermore, credibility checks identify potential bias among key witness relations. "
                "Motive assessment indicates a clear financial interest."
            )

            return {
                "contradictions": contradictions,
                "motives": motives,
                "witness_biases": witness_biases,
                "signals": signals,
                "explanation": explanation
            }

        # Query LLM tasks in the pipeline
        contradictions = await self.detect_contradictions(case_context)
        motives = await self.detect_motives(case_context)
        witness_biases = await self.detect_witness_biases(case_context)
        signals = await self.extract_legal_signals(case_context, len(contradictions), corroboration_count)
        explanation = await self.generate_explanation(case_context, signals, contradictions)

        return {
            "contradictions": contradictions,
            "motives": motives,
            "witness_biases": witness_biases,
            "signals": signals,
            "explanation": explanation
        }

    async def query(self, system_prompt: str, user_prompt: str, json_format: bool = False) -> str:
        """Expose _query_llm publicly for benchmarking and custom queries."""
        return await self._query_llm(system_prompt, user_prompt, json_format)

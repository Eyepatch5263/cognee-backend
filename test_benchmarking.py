import asyncio
import sys
import os
import time
import json
import logging

# Ensure stdout is unbuffered so print statements show in real time
sys.stdout.reconfigure(line_buffering=True)

# Add backend directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.benchmarking import BenchmarkingService
from app.services.cognee_client import CogneeAPIClient
from app.services.llm_service import LLMReasoningService
from app.services.legal_engine import LegalIntelligenceEngine

# Configure logging to print to console with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

async def main():
    print("=" * 70, flush=True)
    print("COGNIVERDICT LEGAL RACHMARK PIPELINE TESTER", flush=True)
    print("=" * 70, flush=True)
    
    print("\nInitializing clients and legal reasoning engines...", flush=True)
    client = CogneeAPIClient()
    reasoning_service = LLMReasoningService()
    legal_engine = LegalIntelligenceEngine()
    
    case_id = "CASE_003"
    print(f"\nStarting benchmark suite for case: {case_id}...", flush=True)
    start_time = time.time()
    
    try:
        report = await BenchmarkingService.run_case_benchmark(
            case_id=case_id,
            client=client,
            reasoning_service=reasoning_service,
            legal_engine=legal_engine
        )
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 70, flush=True)
        print(f"BENCHMARK COMPLETE (Total time: {elapsed:.2f} seconds)", flush=True)
        print("=" * 70, flush=True)
        
        print("\n--- METRICS REPORT ---", flush=True)
        for metric, value in report["metrics"].items():
            print(f"  {metric:<25}: {value}", flush=True)
            
        print("\n--- FAILURE ANALYSIS ---", flush=True)
        print(report["failure_analysis"], flush=True)
        
        print("\n--- DETAILED ANSWERS & REASONING COMPARISON ---", flush=True)
        for idx, item in enumerate(report["predictions_vs_actual"], 1):
            print(f"\n[{idx}] Query: {item['question']}", flush=True)
            print(f"    Category: {item['category']}", flush=True)
            print(f"    Expected: {item['expected']}", flush=True)
            print(f"    Predicted: {item['predicted']}", flush=True)
            print("-" * 50, flush=True)
            
    except Exception as e:
        print("\n=== FATAL EXCEPTION OCCURRED ===", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

import sqlite3
import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../benchmarks.db"))

class BenchmarkStore:
    @staticmethod
    def get_connection():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    def initialize_db(cls):
        conn = cls.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    case_id TEXT,
                    run_id INTEGER,
                    result TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (case_id, run_id)
                )
            """)
            conn.commit()
        except Exception as e:
            logger.error(f"Error initializing benchmarks database: {e}")
        finally:
            conn.close()

    @classmethod
    def get_latest_result(cls, case_id: str) -> Optional[Dict[str, Any]]:
        cls.initialize_db()
        conn = cls.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT run_id, result FROM benchmark_runs WHERE case_id = ? ORDER BY run_id DESC LIMIT 1",
                (case_id,)
            )
            row = cursor.fetchone()
            if row:
                result_data = json.loads(row["result"])
                result_data["run_id"] = row["run_id"]
                result_data["case_id"] = case_id
                return result_data
            return None
        except Exception as e:
            logger.error(f"Error fetching latest benchmark result: {e}")
            return None
        finally:
            conn.close()

    @classmethod
    def save_result(cls, case_id: str, result: Dict[str, Any]) -> int:
        cls.initialize_db()
        conn = cls.get_connection()
        try:
            cursor = conn.cursor()
            # Find max run_id
            cursor.execute(
                "SELECT COALESCE(MAX(run_id), 0) as max_run FROM benchmark_runs WHERE case_id = ?",
                (case_id,)
            )
            row = cursor.fetchone()
            next_run_id = (row["max_run"] if row else 0) + 1

            # Make a copy and ensure run_id/case_id are not in it or keep it clean
            result_copy = dict(result)
            result_copy["run_id"] = next_run_id
            result_copy["case_id"] = case_id

            cursor.execute(
                "INSERT INTO benchmark_runs (case_id, run_id, result) VALUES (?, ?, ?)",
                (case_id, next_run_id, json.dumps(result_copy))
            )
            conn.commit()
            return next_run_id
        except Exception as e:
            logger.error(f"Error saving benchmark result: {e}")
            raise e
        finally:
            conn.close()

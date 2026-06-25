import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)
FEEDBACK_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../feedbacks.json"))

class FeedbackStore:
    """
    Simple file-based JSON storage to track expert legal feedback,
    overrides, and corrections for each case.
    """

    @staticmethod
    def _read_all() -> Dict[str, List[Dict[str, Any]]]:
        if not os.path.exists(FEEDBACK_FILE):
            return {}
        try:
            with open(FEEDBACK_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read feedback file: {e}")
            return {}

    @staticmethod
    def _write_all(data: Dict[str, List[Dict[str, Any]]]):
        try:
            os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
            with open(FEEDBACK_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write feedback file: {e}")

    @classmethod
    def get_feedbacks(cls, case_id: str) -> List[Dict[str, Any]]:
        """Retrieve all expert feedbacks for a case."""
        data = cls._read_all()
        return data.get(case_id, [])

    @classmethod
    def save_feedbacks(cls, case_id: str, feedbacks: List[Dict[str, Any]]):
        """Overwrite all expert feedbacks for a case."""
        data = cls._read_all()
        data[case_id] = feedbacks
        cls._write_all(data)

    @classmethod
    def add_feedback(cls, case_id: str, feedback: Dict[str, Any]):
        """Append a new feedback item for a case."""
        data = cls._read_all()
        if case_id not in data:
            data[case_id] = []
        data[case_id].append(feedback)
        cls._write_all(data)

    @classmethod
    def clear_feedbacks(cls, case_id: str):
        """Remove all feedback overrides for a case."""
        data = cls._read_all()
        if case_id in data:
            del data[case_id]
            cls._write_all(data)

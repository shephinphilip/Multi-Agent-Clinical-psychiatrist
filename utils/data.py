"""
utils/data.py
------------------------------------
Enhanced version for Zenark adaptive question system.
Adds category mapping and reload support.
"""

import importlib.util
import os
import json
import logging
from types import ModuleType
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MentalHealthDataset:
    """
    MentalHealthDataset loads question sets from `dataset/question_pool.py`
    or from JSON files for flexibility.
    Now supports:
      ✅ questions_by_category mapping
      ✅ reload() to refresh dataset mid-session
      ✅ mixed formats (dict-of-lists, list-of-dicts)
    """

    dataset_dir = "dataset"

    def __init__(self, dataname: str = "question_pool"):
        self.dataname = dataname
        self.data_path_py = os.path.join(self.dataset_dir, f"{dataname}.py")
        self.data_path_json = os.path.join(self.dataset_dir, f"{dataname}.json")

        logger.info(f"📘 Initializing dataset loader for {dataname}")
        self.questions: List[Dict[str, Any]] = []
        self.questions_by_category: Dict[str, List[str]] = {}

        self.load()
        self._build_category_map()

    # ------------------------------------------------------------
    # LOADERS
    # ------------------------------------------------------------
    def load(self) -> None:
        """Load from .py or .json depending on which exists."""
        if os.path.exists(self.data_path_py):
            logger.info(f"📘 Loading question pool from {self.data_path_py}")
            self._load_from_py()
        elif os.path.exists(self.data_path_json):
            logger.info(f"📗 Loading question pool from {self.data_path_json}")
            self._load_from_json()
        else:
            raise FileNotFoundError(f"❌ No dataset found in {self.dataset_dir}")

    def _load_from_py(self) -> None:
        """Dynamically import the question_pool.py file."""
        spec = importlib.util.spec_from_file_location("question_pool", self.data_path_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"❌ Could not load module spec from {self.data_path_py}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore

        if hasattr(module, "question_pool"):
            self.questions = getattr(module, "question_pool")
        elif hasattr(module, "QUESTIONS"):
            self.questions = getattr(module, "QUESTIONS")
        else:
            raise ValueError("❌ question_pool.py must define `question_pool` or `QUESTIONS`.")

    def _load_from_json(self) -> None:
        """Load from JSON structure (supports dict-of-lists or list-of-dicts)."""
        try:
            with open(self.data_path_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                # e.g. {"family": [...], "friends": [...]}
                logger.info(f"📂 Detected dict-based dataset with {len(data.keys())} categories.")
                self.questions = []
                for cat, qs in data.items():
                    if not isinstance(qs, list):
                        continue
                    for q in qs:
                        self.questions.append({"category": cat, "text": q})
            elif isinstance(data, list):
                # e.g. [{"intent": "family", "question": "..."}]
                logger.info(f"📋 Detected list-based dataset with {len(data)} entries.")
                self.questions = data
            else:
                raise ValueError("❌ Invalid JSON format: expected list or dict.")

        except Exception as e:
            logger.exception(f"❌ Error loading dataset: {e}")
            raise

    # ------------------------------------------------------------
    # CATEGORY MAPPING
    # ------------------------------------------------------------
    def _build_category_map(self) -> None:
        """Build a category→question_text mapping for easy retrieval."""
        category_map: Dict[str, List[str]] = {}

        for q in self.questions:
            cat = (
                q.get("category")
                or q.get("intent")
                or q.get("domain")
                or "general"
            ).lower()
            text = (
                q.get("text")
                or q.get("question")
                or q.get("value")
            )
            if not text:
                continue
            category_map.setdefault(cat, []).append(text)

        self.questions_by_category = category_map
        logger.info(f"✅ Built category map: {len(category_map)} categories loaded.")

    # ------------------------------------------------------------
    # ACCESSORS
    # ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        q = self.questions[idx]
        return {
            "id": q.get("id"),
            "category": q.get("category", "general_stress"),
            "text": q.get("text") or q.get("question"),
            "difficulty": q.get("difficulty", "medium"),
            "intent": q.get("intent", "diagnostic"),
            "options": q.get("options", None),
        }

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Retrieve all questions in a specific category."""
        return [q for q in self.questions if q.get("category", "").lower() == category.lower()]

    def get_random_question(self, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return one random question, optionally filtered by category."""
        import random
        if not self.questions:
            return None
        if category:
            candidates = self.get_by_category(category)
            return random.choice(candidates) if candidates else None
        return random.choice(self.questions)

    # ------------------------------------------------------------
    # RELOAD
    # ------------------------------------------------------------
    def reload(self) -> None:
        """Reload dataset from disk and rebuild mapping."""
        logger.info("🔄 Reloading MentalHealthDataset...")
        self.questions.clear()
        self.questions_by_category.clear()
        self.load()
        self._build_category_map()
        logger.info("✅ Dataset reloaded successfully.")

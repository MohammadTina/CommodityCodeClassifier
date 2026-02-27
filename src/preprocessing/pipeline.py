"""
Preprocessing pipeline and step base class.
Each step is independently configurable and toggleable via config/preprocessing.yaml.
New steps can be added without changing any existing code.
"""

import logging
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ─── Base Step ───────────────────────────────────────────────────────────────

class BaseStep(ABC):
    """Abstract base class for all preprocessing steps."""

    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get("enabled", True)

    @abstractmethod
    def apply(self, query: str) -> str:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(enabled={self.enabled})"


# ─── Concrete Steps ──────────────────────────────────────────────────────────

class StripIrrelevantPhrasesStep(BaseStep):
    """
    Removes phrases that add no classification value.
    Configurable list of phrases to strip.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.phrases = [p.lower() for p in config.get("phrases", [])]

    def apply(self, query: str) -> str:
        result = query
        for phrase in self.phrases:
            result = re.sub(
                re.escape(phrase),
                "",
                result,
                flags=re.IGNORECASE
            )
        return result.strip()


class AbbreviationExpansionStep(BaseStep):
    """
    Expands known abbreviations to full terms.
    Configurable mappings — add new ones via config without code changes.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.mappings = config.get("mappings", {})

    def apply(self, query: str) -> str:
        result = query
        for abbr, expansion in self.mappings.items():
            # whole word match only — avoid replacing partial words
            result = re.sub(
                rf"\b{re.escape(abbr)}\b",
                expansion,
                result,
                flags=re.IGNORECASE
            )
        return result


class LowercaseNormalizeStep(BaseStep):
    """Normalize query to lowercase."""

    def apply(self, query: str) -> str:
        return query.lower()


class QueryRewritingStep(BaseStep):
    """
    Uses LLM to rewrite informal query into commodity description style.
    Disabled by default — enable once baseline accuracy is established.
    """
    def __init__(self, config: dict, llm_client=None):
        super().__init__(config)
        self.llm_client = llm_client

    def apply(self, query: str) -> str:
        if not self.llm_client:
            logger.warning("QueryRewritingStep: no LLM client configured, skipping")
            return query

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a commodity classification expert. "
                    "Rewrite the user's product description into formal "
                    "commodity classification language. "
                    "Return only the rewritten description, nothing else."
                )
            },
            {"role": "user", "content": query}
        ]
        try:
            return self.llm_client.complete(prompt).strip()
        except Exception as e:
            logger.warning(f"QueryRewritingStep failed: {e}, using original query")
            return query


class HyDEStep(BaseStep):
    """
    Hypothetical Document Embedding.
    LLM generates a hypothetical commodity description matching the query,
    then we embed that instead of the raw query — improves retrieval accuracy.
    Disabled by default — enable after baseline accuracy is established.
    """
    def __init__(self, config: dict, llm_client=None):
        super().__init__(config)
        self.llm_client = llm_client

    def apply(self, query: str) -> str:
        if not self.llm_client:
            logger.warning("HyDEStep: no LLM client configured, skipping")
            return query

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a commodity classification expert. "
                    "Given a product description, write a single sentence "
                    "that resembles how this product would be described "
                    "in an official customs tariff schedule. "
                    "Return only the tariff-style description, nothing else."
                )
            },
            {"role": "user", "content": query}
        ]
        try:
            hypothetical = self.llm_client.complete(prompt).strip()
            logger.debug(f"HyDE generated: {hypothetical}")
            return hypothetical
        except Exception as e:
            logger.warning(f"HyDEStep failed: {e}, using original query")
            return query


# ─── Step Factory ────────────────────────────────────────────────────────────

class StepFactory:
    """
    Creates preprocessing steps from config.
    Register new step types here — no other code changes needed.
    """
    _registry = {
        "strip_irrelevant_phrases": StripIrrelevantPhrasesStep,
        "abbreviation_expansion": AbbreviationExpansionStep,
        "lowercase_normalize": LowercaseNormalizeStep,
        "query_rewriting": QueryRewritingStep,
        "hyde": HyDEStep,
    }

    @classmethod
    def create(cls, step_config: dict, llm_client=None) -> BaseStep:
        name = step_config.get("name")
        if name not in cls._registry:
            raise ValueError(
                f"Unknown preprocessing step: '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        step_class = cls._registry[name]

        # LLM-dependent steps get the client injected
        if name in ("query_rewriting", "hyde"):
            return step_class(step_config, llm_client=llm_client)
        return step_class(step_config)

    @classmethod
    def register(cls, name: str, step_class: type):
        """Register a new step type — extensible without core changes."""
        cls._registry[name] = step_class


# ─── Pipeline ────────────────────────────────────────────────────────────────

class PreprocessingPipeline:
    """
    Ordered pipeline of preprocessing steps.
    Steps execute in config order, each enabled/disabled independently.
    """

    def __init__(self, config: dict, llm_client=None):
        self.enabled = config.get("enabled", True)
        self.steps: list[BaseStep] = []

        if self.enabled:
            for step_config in config.get("steps", []):
                if step_config.get("enabled", True):
                    step = StepFactory.create(step_config, llm_client)
                    self.steps.append(step)
                    logger.debug(f"Preprocessing step loaded: {step}")

        logger.info(
            f"Preprocessing pipeline: {len(self.steps)} active steps"
        )

    def process(self, query: str) -> str:
        """
        Apply all enabled steps in order.

        Args:
            query: raw user input

        Returns:
            Cleaned, normalized query
        """
        if not self.enabled or not query:
            return query

        result = query
        for step in self.steps:
            before = result
            result = step.apply(result)
            if result != before:
                logger.debug(
                    f"{step.__class__.__name__}: "
                    f"'{before}' → '{result}'"
                )

        return result.strip()

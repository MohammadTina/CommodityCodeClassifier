"""
LLM response parser.
Parses the structured text output from the LLM into a clean dict.
Designed to be robust — handles variations in LLM output formatting.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Valid confidence values
VALID_CONFIDENCE = {"HIGH", "LOW", "NO MATCH"}


class ResponseParser:
    """
    Parses the structured classification output from the LLM.
    Falls back gracefully if the LLM doesn't follow format exactly.
    """

    def parse(
        self,
        raw_response: str,
        original_query: str,
        candidates: list[dict]
    ) -> dict:
        """
        Parse raw LLM response into structured classification result.

        Args:
            raw_response: raw text from LLM
            original_query: original user query (for context)
            candidates: retrieved candidates (for validation)

        Returns:
            Structured classification dict
        """
        try:
            result = self._extract_fields(raw_response)
            result = self._validate_and_correct(result, candidates)
            return result
        except Exception as e:
            logger.error(f"Response parsing failed: {e}")
            return self._fallback_result(raw_response)

    def _extract_fields(self, text: str) -> dict:
        """Extract structured fields from LLM output text."""

        confidence = self._extract_field(
            text,
            r"Confidence\s*:\s*(.+?)(?:\n|$)"
        )
        national_code = self._extract_field(
            text,
            r"National Code\s*:\s*(.+?)(?:\n|$)"
        )
        description = self._extract_field(
            text,
            r"Description\s*:\s*(.+?)(?:\n|$)"
        )
        reasoning = self._extract_section(text, "REASONING")
        hierarchy_raw = self._extract_section(text, "HIERARCHY PATH")
        alternatives_raw = self._extract_section(text, "ALTERNATIVES CONSIDERED")
        exclusions = self._extract_section(text, "EXCLUSIONS NOTED")

        # normalize confidence value
        confidence = self._normalize_confidence(confidence)

        # clean national code
        if national_code and national_code.upper() in ("N/A", "NONE", ""):
            national_code = None

        return {
            "confidence": confidence,
            "national_code": national_code,
            "description": description,
            "reasoning": reasoning or raw_response,
            "hierarchy_path": self._parse_hierarchy(hierarchy_raw),
            "alternatives": self._parse_alternatives(alternatives_raw),
            "exclusions_noted": exclusions
        }

    def _extract_field(self, text: str, pattern: str) -> Optional[str]:
        """Extract a single field using regex."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            return value if value else None
        return None

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a multi-line section from the response."""
        pattern = rf"{re.escape(section_name)}:\s*\n(.*?)(?=\n[A-Z][A-Z ]+:|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            return content if content else None
        return None

    def _normalize_confidence(self, raw: Optional[str]) -> str:
        """Normalize confidence to one of: HIGH, LOW, NO MATCH."""
        if not raw:
            return "LOW"
        upper = raw.upper().strip()
        if "HIGH" in upper:
            return "HIGH"
        if "NO MATCH" in upper or "NO_MATCH" in upper:
            return "NO MATCH"
        return "LOW"

    def _parse_hierarchy(self, raw: Optional[str]) -> Optional[dict]:
        """Parse hierarchy path section into structured dict."""
        if not raw:
            return None

        hierarchy = {}
        patterns = {
            "section": r"Section\s*[-→>]+\s*(.+?)(?:\n|$)",
            "chapter": r"Chapter\s*[-→>]+\s*(.+?)(?:\n|$)",
            "heading": r"Heading\s*[-→>]+\s*(.+?)(?:\n|$)",
            "subheading": r"Subheading\s*[-→>]+\s*(.+?)(?:\n|$)",
            "national": r"National\s*[-→>]+\s*(.+?)(?:\n|$)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                hierarchy[key] = match.group(1).strip()

        return hierarchy if hierarchy else None

    def _parse_alternatives(self, raw: Optional[str]) -> Optional[list[dict]]:
        """Parse alternatives section into list of dicts."""
        if not raw or raw.strip() in ("", "N/A", "None"):
            return None

        alternatives = []
        # look for code patterns like 12345678 in the alternatives text
        code_pattern = r"\b(\d{6,10})\b"
        codes = re.findall(code_pattern, raw)

        for code in codes:
            # extract surrounding text as reasoning
            idx = raw.find(code)
            snippet = raw[max(0, idx - 20):idx + 200].strip()
            alternatives.append({
                "national_code": code,
                "reasoning": snippet
            })

        return alternatives if alternatives else None

    def _validate_and_correct(
        self,
        result: dict,
        candidates: list[dict]
    ) -> dict:
        """
        Validate that the returned national code exists in our candidates.
        If the LLM hallucinated a code not in our list, flag it.
        """
        national_code = result.get("national_code")
        if not national_code:
            return result

        candidate_codes = {c["national_code"] for c in candidates}

        # Handle multiple codes separated by comma or semicolon
        separators = [",", ";", " or ", "/"]
        for sep in separators:
            if sep in national_code:
                # Split and clean
                codes = [c.strip() for c in national_code.replace(sep, ",").split(",")]
                codes = [c for c in codes if c]  # remove empty strings

                # Find first valid code
                for code in codes:
                    if code in candidate_codes:
                        logger.info(
                            f"LLM returned multiple codes '{national_code}', "
                            f"using first valid: {code}"
                        )
                        result["national_code"] = code
                        return result

                # None of the codes are valid - fall through to warning below
                break

        # Single code validation
        if national_code not in candidate_codes:
            logger.warning(
                f"LLM returned code '{national_code}' not in candidates. "
                f"Candidates: {candidate_codes}"
            )
            result["confidence"] = "LOW"
            result["reasoning"] = (
                f"[WARNING: LLM suggested code {national_code} which was "
                f"not in the retrieved candidates. Classification downgraded to LOW.] "
                f"\n\n{result.get('reasoning', '')}"
            )
            result["national_code"] = None

        return result

    def _fallback_result(self, raw_response: str) -> dict:
        """Return a safe fallback if parsing completely fails."""
        return {
            "confidence": "LOW",
            "national_code": None,
            "description": None,
            "reasoning": f"Response parsing failed. Raw LLM output: {raw_response[:500]}",
            "hierarchy_path": None,
            "alternatives": None,
            "exclusions_noted": None
        }

"""
Mathematical Expression Verifier using SymPy.
Validates reasoning paths by comparing final answers.
"""

import re
import sympy
from sympy import simplify, sympify, Eq, solve, N
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from typing import Tuple, Optional, Any, Dict
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARSE_ERROR = "parse_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class VerificationResult:
    status: VerificationStatus
    expected: Optional[str] = None
    predicted: Optional[str] = None
    confidence: float = 1.0
    error_message: Optional[str] = None
    intermediate_steps_valid: bool = True


class MathVerifier:
    """
    Verifies mathematical expressions and reasoning paths.
    Supports:
    - Numeric answer comparison
    - Symbolic expression equivalence
    - Unit handling
    - Fraction/decimal equivalence
    """
    
    def __init__(self, tolerance: float = 1e-6, timeout: int = 10):
        self.tolerance = tolerance
        self.timeout = timeout
        self.transformations = standard_transformations + (implicit_multiplication_application,)
        self._numeric_pattern = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")

    def _strip_generation_artifacts(self, text: str) -> str:
        """Remove common chat-template and generation artifacts."""
        if text is None:
            return ""

        cleaned = str(text)
        cleaned = re.sub(r"<\|[^>]+?\|>", " ", cleaned)
        cleaned = re.sub(r"\[/?INST\]", " ", cleaned)
        cleaned = cleaned.replace("</s>", " ")
        cleaned = re.sub(r"`{3,}", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _extract_last_numeric_candidate(self, text: str) -> Optional[str]:
        """Extract the last numeric-looking answer from free-form text."""
        if not text:
            return None

        sanitized = self._strip_generation_artifacts(text)
        sanitized = re.sub(r"\\text\{[^}]*\}", " ", sanitized)
        matches = self._numeric_pattern.findall(sanitized)
        if not matches:
            return None
        return matches[-1].replace(",", "")

    def _clean_candidate_answer(self, answer: str) -> Optional[str]:
        """Normalize raw regex captures into usable answer strings."""
        if answer is None:
            return None

        cleaned = self._strip_generation_artifacts(answer)
        cleaned = cleaned.strip(" \t\n\r`*$:;,.!?\"'()[]{}")
        cleaned = re.sub(
            r"^(?:the\s+)?(?:final\s+)?answer(?:\s+is)?[:\s-]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"^#+\s*", "", cleaned).strip()

        if not cleaned or re.fullmatch(r"[|#:\-]+", cleaned):
            return None

        # Preserve structured assignments like x=2, y=-3.
        if "=" in cleaned and re.search(r"[a-zA-Z_]", cleaned):
            return cleaned

        numeric_candidate = self._extract_last_numeric_candidate(cleaned)
        if numeric_candidate is not None:
            return numeric_candidate

        if len(cleaned.split()) > 8:
            return None
        return cleaned

    def extract_final_answer(self, text: str) -> Optional[str]:
        """
        Extract the final answer from a reasoning path.
        Looks for patterns like:
        - "The answer is X"
        - "#### X"
        - "= X" at the end
        - Boxed answers: \\boxed{X}
        """
        patterns = [
            r'\\boxed\{([^}]+)\}',
            r'####\s*(.+?)(?:\n|$)',
            r'[Tt]he (?:final )?answer is[:\s]*([^\n.]+)',
            r'[Tt]herefore[,:]?\s*(?:the answer is\s*)?([^\n.]+)',
            r'[Ss]o[,:]?\s*(?:the answer is\s*)?([^\n.]+?)(?:\.|$)',
            r'=\s*([^\n=]+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                answer = matches[-1].strip()
                # Clean up common artifacts
                answer = re.sub(r'^\$|\$$', '', answer)
                answer = re.sub(r'^\\text\{|\}$', '', answer)
                return answer
        
        return None

    def _coerce_to_answer(self, text: str) -> str:
        """
        Use answer extraction for reasoning traces, but keep direct answer strings intact.
        """
        extracted = self.extract_final_answer(text)
        if extracted is None:
            return text

        lowered = text.lower()
        reasoning_markers = ("####", "\\boxed", "answer is", "therefore", "\n")
        if any(marker in lowered for marker in reasoning_markers):
            return extracted

        # If extraction would discard significant structure from a short direct answer,
        # preserve the original text.
        if len(text) <= len(extracted) + 8:
            return text
        return extracted
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize an answer string for comparison."""
        if answer is None:
            return ""
        
        answer = self._strip_generation_artifacts(str(answer)).strip()
        
        # Remove currency symbols and units
        answer = re.sub(r'[\$£€]', '', answer)
        answer = re.sub(r'\s*(dollars?|cents?|percent|%|miles?|hours?|minutes?|seconds?|kg|km|m|cm|mm)\s*$', '', answer, flags=re.IGNORECASE)
        
        # Handle fractions written as "X/Y"
        answer = answer.replace('\\frac', '')
        
        # Remove commas in numbers
        answer = re.sub(r'(\d),(\d)', r'\1\2', answer)
        
        # Remove LaTeX formatting
        answer = re.sub(r'\\[a-zA-Z]+', '', answer)
        answer = answer.replace('{', '').replace('}', '')
        answer = answer.strip(" \t\n\r`*$:;,.!?\"'")
        
        return answer.strip()
    
    def parse_numeric(self, value: str) -> Optional[float]:
        """Try to parse a string as a numeric value."""
        try:
            normalized = self.normalize_answer(value)
            
            # Handle fractions
            if '/' in normalized:
                parts = normalized.split('/')
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
            
            # Handle mixed numbers like "3 1/2"
            mixed_match = re.match(r'(-?\d+)\s+(\d+)/(\d+)', normalized)
            if mixed_match:
                whole, num, denom = mixed_match.groups()
                return float(whole) + float(num) / float(denom)
            
            # Direct numeric parse
            return float(normalized)
        except (ValueError, ZeroDivisionError):
            return None
    
    def parse_symbolic(self, expr: str) -> Optional[Any]:
        """Parse a string as a symbolic expression."""
        try:
            normalized = self.normalize_answer(expr)
            return parse_expr(normalized, transformations=self.transformations)
        except Exception:
            return None
    
    def compare_numeric(self, pred: str, expected: str) -> Tuple[bool, float]:
        """Compare two values numerically."""
        pred_num = self.parse_numeric(pred)
        exp_num = self.parse_numeric(expected)
        
        if pred_num is None or exp_num is None:
            return False, 0.0
        
        if exp_num == 0:
            is_equal = abs(pred_num) < self.tolerance
        else:
            relative_error = abs(pred_num - exp_num) / abs(exp_num)
            is_equal = relative_error < self.tolerance
        
        confidence = 1.0 if is_equal else max(0, 1 - abs(pred_num - exp_num) / max(abs(exp_num), 1))
        return is_equal, confidence
    
    def compare_symbolic(self, pred: str, expected: str) -> Tuple[bool, float]:
        """Compare two expressions symbolically."""
        pred_sym = self.parse_symbolic(pred)
        exp_sym = self.parse_symbolic(expected)
        
        if pred_sym is None or exp_sym is None:
            return False, 0.0
        
        try:
            # Try symbolic simplification
            diff = simplify(pred_sym - exp_sym)
            is_equal = diff == 0
            
            # If not symbolically equal, try numeric evaluation
            if not is_equal:
                pred_val = complex(N(pred_sym))
                exp_val = complex(N(exp_sym))
                if abs(exp_val) > 0:
                    is_equal = abs(pred_val - exp_val) / abs(exp_val) < self.tolerance
                else:
                    is_equal = abs(pred_val) < self.tolerance
            
            return is_equal, 1.0 if is_equal else 0.0
        except Exception:
            return False, 0.0

    def parse_variable_assignments(self, value: str) -> Optional[Dict[str, Any]]:
        """Parse answers like 'x=2, y=-3' into a variable->value mapping."""
        normalized = self.normalize_answer(value).strip()
        normalized = normalized.strip("()[]{}")
        if "=" not in normalized:
            return None

        assignments: Dict[str, Any] = {}
        parts = [part.strip() for part in re.split(r"[,\n;]+", normalized) if part.strip()]
        if not parts:
            return None

        for part in parts:
            if "=" not in part:
                return None
            lhs, rhs = part.split("=", 1)
            var_name = lhs.strip()
            value_str = rhs.strip()

            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", var_name):
                return None

            numeric_value = self.parse_numeric(value_str)
            if numeric_value is not None:
                assignments[var_name] = numeric_value
                continue

            symbolic_value = self.parse_symbolic(value_str)
            if symbolic_value is None:
                return None
            assignments[var_name] = symbolic_value

        return assignments or None

    def compare_variable_assignments(self, pred: str, expected: str) -> Tuple[bool, float]:
        """Compare multi-variable assignment answers."""
        pred_assignments = self.parse_variable_assignments(pred)
        exp_assignments = self.parse_variable_assignments(expected)

        if pred_assignments is None or exp_assignments is None:
            return False, 0.0

        if set(pred_assignments.keys()) != set(exp_assignments.keys()):
            return False, 0.0

        for var_name, exp_value in exp_assignments.items():
            pred_value = pred_assignments[var_name]

            if isinstance(pred_value, (int, float)) and isinstance(exp_value, (int, float)):
                if exp_value == 0:
                    if abs(pred_value) >= self.tolerance:
                        return False, 0.0
                else:
                    rel_err = abs(pred_value - exp_value) / abs(exp_value)
                    if rel_err >= self.tolerance:
                        return False, 0.0
                continue

            try:
                diff = simplify(pred_value - exp_value)
                if diff != 0:
                    return False, 0.0
            except Exception:
                return False, 0.0

        return True, 1.0
    
    def verify(self, predicted: str, expected: str) -> VerificationResult:
        """
        Verify if predicted answer matches expected answer.
        Tries multiple comparison strategies.
        """
        # Extract final answers if full reasoning paths provided
        pred_answer = self._coerce_to_answer(predicted)
        exp_answer = self._coerce_to_answer(expected)
        
        # Normalize both answers
        pred_norm = self.normalize_answer(pred_answer)
        exp_norm = self.normalize_answer(exp_answer)
        
        # Exact string match
        if pred_norm.lower() == exp_norm.lower():
            return VerificationResult(
                status=VerificationStatus.CORRECT,
                expected=exp_norm,
                predicted=pred_norm,
                confidence=1.0
            )

        # Multi-variable assignment comparison
        is_equal, confidence = self.compare_variable_assignments(pred_norm, exp_norm)
        if is_equal:
            return VerificationResult(
                status=VerificationStatus.CORRECT,
                expected=exp_norm,
                predicted=pred_norm,
                confidence=confidence,
            )
        
        # Numeric comparison
        is_equal, confidence = self.compare_numeric(pred_norm, exp_norm)
        if is_equal:
            return VerificationResult(
                status=VerificationStatus.CORRECT,
                expected=exp_norm,
                predicted=pred_norm,
                confidence=confidence
            )
        
        # Symbolic comparison
        is_equal, confidence = self.compare_symbolic(pred_norm, exp_norm)
        if is_equal:
            return VerificationResult(
                status=VerificationStatus.CORRECT,
                expected=exp_norm,
                predicted=pred_norm,
                confidence=confidence
            )
        
        return VerificationResult(
            status=VerificationStatus.INCORRECT,
            expected=exp_norm,
            predicted=pred_norm,
            confidence=0.0
        )
    
    def verify_reasoning_path(self, reasoning: str, expected_answer: str) -> VerificationResult:
        """
        Verify a complete reasoning path.
        Extracts the final answer and compares it.
        """
        final_answer = self.extract_final_answer(reasoning)
        
        if final_answer is None:
            return VerificationResult(
                status=VerificationStatus.PARSE_ERROR,
                expected=expected_answer,
                predicted=None,
                error_message="Could not extract final answer from reasoning"
            )
        
        return self.verify(final_answer, expected_answer)
    
    def verify_intermediate_steps(self, steps: list[str]) -> list[VerificationResult]:
        """
        Verify intermediate calculation steps.
        Each step should be in format "expression = result"
        """
        results = []
        
        for step in steps:
            if '=' not in step:
                results.append(VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    error_message="Step does not contain equation"
                ))
                continue
            
            parts = step.split('=')
            if len(parts) != 2:
                results.append(VerificationResult(
                    status=VerificationStatus.PARSE_ERROR,
                    error_message="Invalid equation format"
                ))
                continue
            
            lhs, rhs = parts[0].strip(), parts[1].strip()
            
            try:
                lhs_val = parse_expr(lhs, transformations=self.transformations)
                rhs_val = parse_expr(rhs, transformations=self.transformations)
                
                if simplify(lhs_val - rhs_val) == 0:
                    results.append(VerificationResult(
                        status=VerificationStatus.CORRECT,
                        expected=rhs,
                        predicted=str(simplify(lhs_val))
                    ))
                else:
                    results.append(VerificationResult(
                        status=VerificationStatus.INCORRECT,
                        expected=rhs,
                        predicted=str(simplify(lhs_val))
                    ))
            except Exception as e:
                results.append(VerificationResult(
                    status=VerificationStatus.PARSE_ERROR,
                    error_message=str(e)
                ))
        
        return results


# GSM8K specific verifier
class GSM8KVerifier(MathVerifier):
    """
    Specialized verifier for GSM8K dataset.
    GSM8K answers are always numeric and use #### delimiter.
    """
    
    def extract_final_answer(self, text: str) -> Optional[str]:
        """Extract answer using GSM8K format."""
        # GSM8K uses #### to mark the final answer
        match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(',', '')
        
        # Fallback to parent implementation
        return super().extract_final_answer(text)
    
    def verify(self, predicted: str, expected: str) -> VerificationResult:
        """Verify GSM8K style numeric answers."""
        pred_answer = self.extract_final_answer(predicted) or predicted
        exp_answer = self.extract_final_answer(expected) or expected
        
        try:
            pred_num = float(pred_answer.replace(',', ''))
            exp_num = float(exp_answer.replace(',', ''))
            
            # GSM8K requires exact numeric match (integers)
            if abs(pred_num - exp_num) < 0.001:
                return VerificationResult(
                    status=VerificationStatus.CORRECT,
                    expected=str(exp_num),
                    predicted=str(pred_num),
                    confidence=1.0
                )
            else:
                return VerificationResult(
                    status=VerificationStatus.INCORRECT,
                    expected=str(exp_num),
                    predicted=str(pred_num),
                    confidence=0.0
                )
        except ValueError as e:
            return VerificationResult(
                status=VerificationStatus.PARSE_ERROR,
                expected=exp_answer,
                predicted=pred_answer,
                error_message=str(e)
            )

"""Legacy recovery proposal/metadata helpers, not the DSR recursive code engine.

CandidateEvaluator defaults are synthetic and must not establish research evidence.
"""
from .candidate_generator import CandidateGenerator
from .candidate_evaluator import CandidateEvaluator
from .evidence_gate import EvidenceGate
from .version_manager import VersionManager

__all__ = ["CandidateGenerator", "CandidateEvaluator", "EvidenceGate", "VersionManager"]

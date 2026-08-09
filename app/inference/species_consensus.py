"""
Multi-ROI Species Consensus Module.
Aggregates RAGLO bark candidate inference outputs across multiple ROIs of the same tree trunk.
Only candidates that individually PASS open-set verification cast known-species votes.
"""
from typing import Dict, Any, List, Tuple, Optional


def evaluate_multi_roi_consensus(
    candidates_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Apply research multi-ROI consensus rules across candidate inference results for a single trunk.

    Consensus Rules:
    - Only candidates that individually PASS open-set (status == 'known') cast a known vote.
    - Rejected candidates never count as known votes.
    - CASE A: Multiple candidates pass open-set and agree on species -> Final: Agreed Species (e.g., 2/3 votes)
    - CASE B: All candidates pass open-set and agree -> Final: Agreed Species (e.g., 3/3 votes)
    - CASE C: Only 1 candidate exists -> Final: Use its open-set decision directly (no fake consensus).
    - CASE D: Multiple known candidates disagree -> Final: Uncertain (Reason: Inconsistent bark-region predictions)
    - CASE E: All candidates rejected by open-set -> Final: Unknown / Unrecognized species
    - CASE F: 1 known vote out of 3 candidates (1 known, 2 rejected) -> Final: Uncertain (Insufficient candidate consensus)
    """
    total_candidates = len(candidates_results)
    if total_candidates == 0:
        return {
            "status": "no_candidates",
            "final_species": "Not evaluated",
            "known_votes": 0,
            "candidate_count": 0,
            "agreement_ratio": 0.0,
            "agreed_class": None,
            "reason": "No bark candidates evaluated."
        }

    # Extract candidates that passed open-set (status == 'known')
    known_candidates = [c for c in candidates_results if c.get("status") == "known" and not c.get("open_set", {}).get("open_set_rejected", True)]
    rejected_candidates = [c for c in candidates_results if c.get("status") != "known" or c.get("open_set", {}).get("open_set_rejected", True)]

    known_votes_count = len(known_candidates)

    # CASE C: Single Candidate
    if total_candidates == 1:
        single_cand = candidates_results[0]
        if single_cand.get("status") == "known" and not single_cand.get("open_set", {}).get("open_set_rejected", True):
            species = single_cand.get("final_species")
            return {
                "status": "known",
                "final_species": species,
                "known_votes": 1,
                "candidate_count": 1,
                "agreement_ratio": 1.0,
                "agreed_class": species,
                "reason": "Single ROI candidate passed open-set verification."
            }
        else:
            return {
                "status": "open_set_rejected",
                "final_species": "Unknown / Unrecognized species",
                "known_votes": 0,
                "candidate_count": 1,
                "agreement_ratio": 0.0,
                "agreed_class": None,
                "reason": "Single ROI candidate failed open-set verification."
            }

    # CASE E: All candidates rejected
    if known_votes_count == 0:
        return {
            "status": "open_set_rejected",
            "final_species": "Unknown / Unrecognized species",
            "known_votes": 0,
            "candidate_count": total_candidates,
            "agreement_ratio": 0.0,
            "agreed_class": None,
            "reason": "All bark candidates rejected by open-set verification."
        }

    # Count votes for each species among known candidates
    species_vote_counts: Dict[str, int] = {}
    for cand in known_candidates:
        sp = cand.get("final_species")
        if sp:
            species_vote_counts[sp] = species_vote_counts.get(sp, 0) + 1

    # CASE D: Multiple known candidates disagree on species
    if len(species_vote_counts) > 1:
        return {
            "status": "uncertain",
            "final_species": "Uncertain",
            "known_votes": known_votes_count,
            "candidate_count": total_candidates,
            "agreement_ratio": round(float(known_votes_count) / total_candidates, 4),
            "agreed_class": None,
            "reason": "Inconsistent bark-region predictions"
        }

    # Single species voted by known candidates
    top_species, vote_count = list(species_vote_counts.items())[0]

    # CASE F: 1 known, multiple rejected (e.g. 1 known, 2 rejected out of 3)
    if known_votes_count == 1 and total_candidates > 1:
        return {
            "status": "uncertain",
            "final_species": "Uncertain",
            "known_votes": 1,
            "candidate_count": total_candidates,
            "agreement_ratio": round(1.0 / total_candidates, 4),
            "agreed_class": top_species,
            "reason": f"Insufficient consensus (only 1 of {total_candidates} candidates passed open-set)."
        }

    # CASE A & B: Majority / unanimous agreement among passed candidates (e.g. 2/3 or 3/3 known)
    return {
        "status": "known",
        "final_species": top_species,
        "known_votes": known_votes_count,
        "candidate_count": total_candidates,
        "agreement_ratio": round(float(known_votes_count) / total_candidates, 4),
        "agreed_class": top_species,
        "reason": f"{known_votes_count} of {total_candidates} candidates agreed on {top_species}."
    }

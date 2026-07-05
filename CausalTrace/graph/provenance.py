"""
Provenance computation for tracking observation→action attribution.

This module implements methods to:
1. Chunk observations into smaller units
2. Compute which observation chunks influenced which actions
3. Detect injected content in observations
4. Build provenance chains for backward slicing
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from causaltrace.models.trajectory import (
    Action,
    ObservationChunk,
    ProvenanceData,
    Trajectory,
)


# Untrusted domains (for marking injections)
UNTRUSTED_DOMAINS = {"forum", "cdn", "social", "public"}

# Injection keywords (for detection)
INJECTION_KEYWORDS = [
    "ATTENTION",
    "NEW PRIORITY INSTRUCTION",
    "SYSTEM UPDATE",
    "DEBUG MODE",
    "IMPORTANT:",
    "IGNORE PREVIOUS",
    "SECURITY VERIFICATION",
    "attacker.com",
    "evil.com",
    "ATTACKER-",
]


class ProvenanceComputer:
    """
    Computes provenance for actions in a trajectory.

    Provenance tracks which observations influenced which actions,
    enabling backward slicing from suspicious actions to injection sources.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        attribution_method: str = "heuristic",
        untrusted_domains: Optional[Set[str]] = None,
    ):
        """
        Initialize provenance computer.

        Args:
            chunk_size: Maximum size for observation chunks (in characters)
            attribution_method: Method for computing attribution ("heuristic", "attention", "llm_cite")
            untrusted_domains: Set of domains considered untrusted
        """
        self.chunk_size = chunk_size
        self.attribution_method = attribution_method
        self.untrusted_domains = untrusted_domains or UNTRUSTED_DOMAINS

    def chunk_observation(
        self,
        content: str,
        domain: Optional[str],
        action_id: int,
        source: str = "webpage",
    ) -> List[ObservationChunk]:
        """
        Chunk an observation into smaller units.

        For long observations (e.g., webpages), we split into chunks to enable
        fine-grained provenance tracking.

        Args:
            content: The observation content
            domain: Domain the observation came from
            action_id: ID of the action that produced this observation
            source: Source type (e.g., "webpage", "api_response")

        Returns:
            List of ObservationChunk objects
        """
        chunks = []

        # Simple chunking: split by paragraphs or fixed size
        paragraphs = content.split("\n\n")

        chunk_counter = 0
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_id = f"obs_{action_id}_chunk_{chunk_counter}"
                    chunks.append(
                        ObservationChunk(
                            chunk_id=chunk_id,
                            content=current_chunk.strip(),
                            source=source,
                            domain=domain,
                            metadata={"action_id": action_id, "chunk_index": chunk_counter},
                        )
                    )
                    chunk_counter += 1
                    current_chunk = ""

            current_chunk += para + "\n\n"

        # Add final chunk
        if current_chunk.strip():
            chunk_id = f"obs_{action_id}_chunk_{chunk_counter}"
            chunks.append(
                ObservationChunk(
                    chunk_id=chunk_id,
                    content=current_chunk.strip(),
                    source=source,
                    domain=domain,
                    metadata={"action_id": action_id, "chunk_index": chunk_counter},
                )
            )

        return chunks

    def detect_injection(self, chunk: ObservationChunk) -> bool:
        """
        Detect if a chunk contains injection content.

        Args:
            chunk: Observation chunk to check

        Returns:
            True if injection detected
        """
        content_upper = chunk.content.upper()

        # Check for injection keywords
        for keyword in INJECTION_KEYWORDS:
            if keyword.upper() in content_upper:
                return True

        return False

    def compute_attribution_heuristic(
        self,
        action: Action,
        observation_chunks: List[ObservationChunk],
    ) -> ProvenanceData:
        """
        Compute provenance using heuristics.

        Heuristic: An action was influenced by chunks if:
        1. The action's input/target contains text from the chunk
        2. The chunk comes from a recently visited domain
        3. The chunk contains data keys consumed by the action

        Args:
            action: The action to compute provenance for
            observation_chunks: All observation chunks seen before this action

        Returns:
            ProvenanceData tracking which chunks influenced this action
        """
        influenced_chunks = []
        confidence_scores = {}
        untrusted_domains_found = set()
        injection_detected = False

        # Get action's textual content
        action_text = f"{action.target} {action.result or ''}".lower()

        # Check each observation chunk
        for chunk in observation_chunks:
            # Skip chunks from after this action
            if chunk.metadata.get("action_id", 0) >= action.action_id:
                continue

            confidence = 0.0

            # Check if action text contains chunk content
            chunk_text = chunk.content[:200].lower()  # Use first 200 chars
            if chunk_text in action_text or any(word in action_text for word in chunk_text.split()[:10]):
                confidence += 0.5

            # Check if chunk domain matches action domain
            if chunk.domain and action.domain and chunk.domain in action.domain:
                confidence += 0.3

            # Check if action consumes data produced by chunk's action
            # (This requires data_consumed to reference observation chunks)
            if action.data_consumed:
                # For now, simple heuristic: recent chunks are more likely
                if chunk.metadata.get("action_id", 0) >= action.action_id - 2:
                    confidence += 0.2

            # Also check if chunk is recent (within last 3 actions)
            if chunk.metadata.get("action_id", 0) >= action.action_id - 3:
                confidence += 0.1

            # If confidence above threshold, record this chunk
            if confidence > 0.2:  # Lower threshold to be more inclusive
                influenced_chunks.append(chunk.chunk_id)
                confidence_scores[chunk.chunk_id] = min(confidence, 1.0)

                # Check if chunk is from untrusted domain
                if chunk.domain:
                    for untrusted in self.untrusted_domains:
                        if untrusted in chunk.domain:
                            untrusted_domains_found.add(chunk.domain)
                            break

                # Check if chunk contains injection
                if self.detect_injection(chunk):
                    injection_detected = True

        return ProvenanceData(
            observation_chunks=influenced_chunks,
            confidence_scores=confidence_scores,
            attribution_method="heuristic",
            is_untrusted=len(untrusted_domains_found) > 0,
            untrusted_domains=untrusted_domains_found,
            injection_detected=injection_detected,
        )

    def compute_provenance(
        self,
        trajectory: Trajectory,
        chunk_observations: bool = True,
    ) -> Trajectory:
        """
        Compute provenance for all actions in a trajectory.

        Args:
            trajectory: Input trajectory
            chunk_observations: Whether to chunk observations (recommended)

        Returns:
            Trajectory with provenance data added to each action
        """
        observation_chunks = []

        # Process each action
        for i, action in enumerate(trajectory.actions):
            # Chunk this action's observation (result)
            if chunk_observations and action.result:
                chunks = self.chunk_observation(
                    content=action.result,
                    domain=action.domain,
                    action_id=action.action_id,
                    source="action_result",
                )
                observation_chunks.extend(chunks)

            # Compute provenance for this action
            if i > 0:  # Skip first action (no prior observations)
                provenance = self.compute_attribution_heuristic(
                    action=action,
                    observation_chunks=observation_chunks,
                )
                action.provenance = provenance

        # Store chunks in trajectory
        trajectory.observation_chunks = observation_chunks

        return trajectory


def compute_provenance_for_trajectory(
    trajectory: Trajectory,
    chunk_size: int = 500,
    attribution_method: str = "heuristic",
) -> Trajectory:
    """
    Convenience function to compute provenance for a trajectory.

    Args:
        trajectory: Input trajectory
        chunk_size: Size of observation chunks
        attribution_method: Attribution method to use

    Returns:
        Trajectory with provenance data
    """
    computer = ProvenanceComputer(
        chunk_size=chunk_size,
        attribution_method=attribution_method,
    )
    return computer.compute_provenance(trajectory)


def trace_action_to_injection(
    action: Action,
    trajectory: Trajectory,
) -> Optional[List[ObservationChunk]]:
    """
    Trace an action back to injection source.

    Given a suspicious action, find the observation chunks that led to it,
    filtering for chunks that contain injection content.

    Args:
        action: The suspicious action
        trajectory: The full trajectory

    Returns:
        List of observation chunks in the provenance chain that contain injections,
        or None if no injection found
    """
    if not action.provenance:
        return None

    # Get chunks that influenced this action
    influenced_chunk_ids = action.provenance.observation_chunks

    # Find actual chunk objects
    chunks_with_injection = []
    for chunk in trajectory.observation_chunks:
        if chunk.chunk_id in influenced_chunk_ids:
            # Check if chunk contains injection
            computer = ProvenanceComputer()
            if computer.detect_injection(chunk):
                chunks_with_injection.append(chunk)

    return chunks_with_injection if chunks_with_injection else None


def get_provenance_chain(
    action: Action,
    trajectory: Trajectory,
) -> List[Tuple[ObservationChunk, float]]:
    """
    Get full provenance chain for an action.

    Returns all observation chunks that influenced this action,
    along with confidence scores.

    Args:
        action: The action
        trajectory: The full trajectory

    Returns:
        List of (chunk, confidence) tuples
    """
    if not action.provenance:
        return []

    chain = []
    for chunk in trajectory.observation_chunks:
        if chunk.chunk_id in action.provenance.observation_chunks:
            confidence = action.provenance.confidence_scores.get(chunk.chunk_id, 0.0)
            chain.append((chunk, confidence))

    # Sort by confidence (highest first)
    chain.sort(key=lambda x: x[1], reverse=True)

    return chain

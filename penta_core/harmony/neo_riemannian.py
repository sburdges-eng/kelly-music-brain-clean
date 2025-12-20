"""
Neo-Riemannian Transformations - Advanced harmonic transformations.

Implements the Neo-Riemannian theory operations:
- P (Parallel): C major <-> C minor
- R (Relative): C major <-> A minor
- L (Leading-tone): C major <-> E minor
- And compound transformations
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class NeoRiemannianTransform(Enum):
    """Basic Neo-Riemannian transformations."""
    P = "P"  # Parallel: major <-> minor (same root)
    R = "R"  # Relative: major <-> relative minor
    L = "L"  # Leading-tone exchange
    N = "N"  # Nebenverwandt (compound: RLP)
    S = "S"  # Slide (compound: LPR)
    H = "H"  # Hexatonic pole (compound: LPL)


# Note names for reference
NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


@dataclass
class Triad:
    """A musical triad (3-note chord)."""
    root: str
    quality: str  # "major" or "minor"

    @property
    def pitch_classes(self) -> List[int]:
        """Get pitch classes (0-11) for the triad."""
        root_pc = NOTES.index(self.root)
        if self.quality == "major":
            return [root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12]
        else:  # minor
            return [root_pc, (root_pc + 3) % 12, (root_pc + 7) % 12]

    def __str__(self) -> str:
        suffix = "" if self.quality == "major" else "m"
        return f"{self.root}{suffix}"

    @classmethod
    def from_pitch_classes(cls, pcs: List[int]) -> Optional["Triad"]:
        """Create triad from pitch classes."""
        if len(pcs) != 3:
            return None

        pcs = sorted(pcs)

        # Check for major triad (0, 4, 7)
        for root_pc in range(12):
            major_pcs = sorted([(root_pc + i) % 12 for i in [0, 4, 7]])
            if pcs == major_pcs:
                return cls(NOTES[root_pc], "major")

            minor_pcs = sorted([(root_pc + i) % 12 for i in [0, 3, 7]])
            if pcs == minor_pcs:
                return cls(NOTES[root_pc], "minor")

        return None


def parallel_transform(triad: Triad) -> Triad:
    """
    P (Parallel) transformation.

    Flips quality while keeping the root.
    C major <-> C minor
    """
    new_quality = "minor" if triad.quality == "major" else "major"
    return Triad(triad.root, new_quality)


def relative_transform(triad: Triad) -> Triad:
    """
    R (Relative) transformation.

    Moves to the relative major/minor.
    C major <-> A minor (down 3 semitones for relative minor)
    A minor <-> C major (up 3 semitones for relative major)
    """
    root_idx = NOTES.index(triad.root)

    if triad.quality == "major":
        # C major -> A minor (down 3 semitones)
        new_root = NOTES[(root_idx - 3) % 12]
        return Triad(new_root, "minor")
    else:
        # A minor -> C major (up 3 semitones)
        new_root = NOTES[(root_idx + 3) % 12]
        return Triad(new_root, "major")


def leading_tone_transform(triad: Triad) -> Triad:
    """
    L (Leading-tone) transformation.

    Moves the root by semitone while keeping 2 common tones.
    C major <-> E minor
    """
    root_idx = NOTES.index(triad.root)

    if triad.quality == "major":
        # C major -> E minor (up 4 semitones)
        new_root = NOTES[(root_idx + 4) % 12]
        return Triad(new_root, "minor")
    else:
        # E minor -> C major (down 4 semitones)
        new_root = NOTES[(root_idx - 4) % 12]
        return Triad(new_root, "major")


def apply_transform(
    triad: Triad,
    transform: NeoRiemannianTransform,
) -> Triad:
    """
    Apply a Neo-Riemannian transformation.

    Args:
        triad: Input triad
        transform: Transformation to apply

    Returns:
        Transformed triad
    """
    if transform == NeoRiemannianTransform.P:
        return parallel_transform(triad)

    elif transform == NeoRiemannianTransform.R:
        return relative_transform(triad)

    elif transform == NeoRiemannianTransform.L:
        return leading_tone_transform(triad)

    elif transform == NeoRiemannianTransform.N:
        # Nebenverwandt = RLP
        result = relative_transform(triad)
        result = leading_tone_transform(result)
        return parallel_transform(result)

    elif transform == NeoRiemannianTransform.S:
        # Slide = LPR
        result = leading_tone_transform(triad)
        result = parallel_transform(result)
        return relative_transform(result)

    elif transform == NeoRiemannianTransform.H:
        # Hexatonic pole = LPL
        result = leading_tone_transform(triad)
        result = parallel_transform(result)
        return leading_tone_transform(result)

    return triad


def apply_transform_sequence(
    triad: Triad,
    transforms: List[NeoRiemannianTransform],
) -> Triad:
    """Apply a sequence of transformations."""
    result = triad
    for transform in transforms:
        result = apply_transform(result, transform)
    return result


def get_transform_path(
    start: Triad,
    end: Triad,
    max_length: int = 4,
) -> Optional[List[NeoRiemannianTransform]]:
    """
    Find a transformation path between two triads.

    Uses BFS to find the shortest path.

    Args:
        start: Starting triad
        end: Target triad
        max_length: Maximum path length to search

    Returns:
        List of transformations, or None if not found
    """
    from collections import deque

    if str(start) == str(end):
        return []

    # BFS
    queue = deque([(start, [])])
    visited = {str(start)}

    while queue:
        current, path = queue.popleft()

        if len(path) >= max_length:
            continue

        for transform in [NeoRiemannianTransform.P, NeoRiemannianTransform.R, NeoRiemannianTransform.L]:
            next_triad = apply_transform(current, transform)
            next_str = str(next_triad)

            if next_str == str(end):
                return path + [transform]

            if next_str not in visited:
                visited.add(next_str)
                queue.append((next_triad, path + [transform]))

    return None


def get_tonnetz_neighbors(triad: Triad) -> List[Tuple[NeoRiemannianTransform, Triad]]:
    """
    Get all Tonnetz neighbors of a triad.

    Returns triads reachable by a single P, R, or L transformation.
    """
    neighbors = []
    for transform in [NeoRiemannianTransform.P, NeoRiemannianTransform.R, NeoRiemannianTransform.L]:
        neighbor = apply_transform(triad, transform)
        neighbors.append((transform, neighbor))
    return neighbors


def generate_hexatonic_system(starting_triad: Triad) -> List[Triad]:
    """
    Generate the hexatonic system containing the starting triad.

    A hexatonic system contains 6 triads connected by P and L.
    """
    system = [starting_triad]
    current = starting_triad

    for i in range(5):
        if i % 2 == 0:
            current = parallel_transform(current)
        else:
            current = leading_tone_transform(current)
        system.append(current)

    return system


def get_hexatonic_systems() -> List[List[str]]:
    """Get all four hexatonic systems."""
    return [
        ["C", "Cm", "Ab", "Abm", "E", "Em"],  # Northern
        ["G", "Gm", "Eb", "Ebm", "B", "Bm"],  # Eastern
        ["D", "Dm", "Bb", "Bbm", "Gb", "Gbm"],  # Southern
        ["A", "Am", "F", "Fm", "Db", "Dbm"],  # Western
    ]


def analyze_progression_transforms(
    chords: List[Tuple[str, str]],
) -> List[Optional[NeoRiemannianTransform]]:
    """
    Analyze the transforms between chords in a progression.

    Args:
        chords: List of (root, quality) tuples

    Returns:
        List of transforms (None if no single transform works)
    """
    if len(chords) < 2:
        return []

    transforms = []

    for i in range(len(chords) - 1):
        current = Triad(chords[i][0], "major" if "maj" in chords[i][1] or chords[i][1] == "" else "minor")
        next_chord = Triad(chords[i + 1][0], "major" if "maj" in chords[i + 1][1] or chords[i + 1][1] == "" else "minor")

        # Find single transform
        found = None
        for transform in NeoRiemannianTransform:
            if str(apply_transform(current, transform)) == str(next_chord):
                found = transform
                break

        transforms.append(found)

    return transforms

"""
Thesaurus Loader - Loads the 6×6×6 DAiW emotion thesaurus for ML training.

Provides utilities to:
- Load all 216 emotion nodes from JSON files
- Generate node IDs and hierarchical labels
- Map nodes to musical attributes
- Create training labels for multi-head classification
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class IntensityTier:
    """A single intensity tier within a sub-sub-emotion."""
    level: int  # 1-6
    label: str  # "subtle", "mild", etc.
    synonyms: List[str] = field(default_factory=list)


@dataclass
class SubSubEmotion:
    """A sub-sub-emotion (leaf node in 6×6×6 hierarchy)."""
    id: str  # e.g., "Iia"
    name: str  # e.g., "satisfied"
    description: str
    node_id: int  # 0-215 computed ID
    intensity_tiers: Dict[str, IntensityTier] = field(default_factory=dict)
    
    # Parent references
    sub_emotion_name: str = ""
    base_emotion_name: str = ""
    
    # Indices for hierarchy
    base_idx: int = 0
    sub_idx: int = 0
    subsub_idx: int = 0


@dataclass
class SubEmotion:
    """A sub-emotion (middle level in 6×6×6 hierarchy)."""
    id: str  # e.g., "Ii"
    name: str  # e.g., "CONTENTMENT"
    description: str
    arousal_bias: float = 0.5
    sub_sub_emotions: Dict[str, SubSubEmotion] = field(default_factory=dict)
    
    # Parent reference
    base_emotion_name: str = ""
    base_idx: int = 0
    sub_idx: int = 0


@dataclass 
class BaseEmotion:
    """A base emotion (top level in 6×6×6 hierarchy)."""
    id: str  # e.g., "I"
    name: str  # e.g., "HAPPY"
    description: str
    valence: str  # "positive", "negative", "mixed"
    arousal_range: Tuple[float, float] = (0.0, 1.0)
    sub_emotions: Dict[str, SubEmotion] = field(default_factory=dict)
    base_idx: int = 0


@dataclass
class EmotionNode:
    """Flattened representation of a single emotion node for ML training."""
    node_id: int  # 0-215
    full_id: str  # e.g., "Iia"
    name: str  # e.g., "satisfied"
    
    # Hierarchy
    base_name: str  # e.g., "HAPPY"
    sub_name: str  # e.g., "CONTENTMENT"
    subsub_name: str  # e.g., "satisfied"
    
    # Indices (for multi-head labels)
    base_idx: int  # 0-5
    sub_idx: int  # 0-5
    subsub_idx: int  # 0-5
    sub_global_idx: int  # 0-35 (for 36-class sub-emotion head)
    
    # Attributes
    description: str = ""
    valence: str = "neutral"
    arousal_bias: float = 0.5
    
    # Intensity synonyms (all tiers combined)
    all_synonyms: List[str] = field(default_factory=list)


@dataclass
class ThesaurusLabels:
    """Multi-head training labels for a sample."""
    node_id: int  # 216-class
    base_idx: int  # 6-class
    sub_global_idx: int  # 36-class
    intensity_tier: int  # 6-class (0-5)
    key_idx: int  # 24-class


# =============================================================================
# THESAURUS LOADER
# =============================================================================


class ThesaurusLoader:
    """
    Loads and manages the 6×6×6 DAiW emotion thesaurus.
    
    Usage:
        loader = ThesaurusLoader(Path("data/emotion_thesaurus"))
        loader.load()
        
        # Get a specific node
        node = loader.get_node(42)
        
        # Get training labels
        labels = loader.get_labels(node_id=42, intensity=3, key_idx=0)
    """
    
    # Canonical order of base emotions
    BASE_EMOTION_ORDER = ["HAPPY", "SAD", "ANGRY", "FEAR", "SURPRISE", "DISGUST"]
    
    # Intensity tier labels
    INTENSITY_LABELS = ["subtle", "mild", "moderate", "strong", "intense", "overwhelming"]
    
    # Key labels (12 major + 12 minor)
    KEY_LABELS = [
        "C_major", "C_minor", "Db_major", "Db_minor",
        "D_major", "D_minor", "Eb_major", "Eb_minor",
        "E_major", "E_minor", "F_major", "F_minor",
        "Gb_major", "Gb_minor", "G_major", "G_minor",
        "Ab_major", "Ab_minor", "A_major", "A_minor",
        "Bb_major", "Bb_minor", "B_major", "B_minor",
    ]
    
    def __init__(self, thesaurus_dir: Path):
        """Initialize with path to thesaurus directory."""
        self.thesaurus_dir = Path(thesaurus_dir)
        self.base_emotions: Dict[str, BaseEmotion] = {}
        self.nodes: Dict[int, EmotionNode] = {}  # node_id → EmotionNode
        self.nodes_by_name: Dict[str, EmotionNode] = {}  # name → EmotionNode
        self.loaded = False
    
    def load(self) -> None:
        """Load all thesaurus JSON files."""
        if self.loaded:
            return
        
        # Load metadata
        metadata_path = self.thesaurus_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Load each base emotion file
        for base_idx, base_name in enumerate(self.BASE_EMOTION_ORDER):
            filename = f"{base_name.lower()}.json"
            filepath = self.thesaurus_dir / filename
            
            if not filepath.exists():
                logger.warning(f"Missing thesaurus file: {filepath}")
                continue
            
            with open(filepath) as f:
                data = json.load(f)
            
            base_emotion = self._parse_base_emotion(data, base_idx)
            self.base_emotions[base_name] = base_emotion
        
        # Build flat node index
        self._build_node_index()
        self.loaded = True
        
        logger.info(f"Loaded {len(self.nodes)} emotion nodes from thesaurus")
    
    def _parse_base_emotion(self, data: Dict, base_idx: int) -> BaseEmotion:
        """Parse a base emotion JSON file."""
        base = BaseEmotion(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            valence=data.get("valence", "neutral"),
            arousal_range=tuple(data.get("arousal_range", [0.0, 1.0])),
            base_idx=base_idx,
        )
        
        # Parse sub-emotions
        sub_emotions_data = data.get("sub_emotions", {})
        for sub_idx, (sub_name, sub_data) in enumerate(sub_emotions_data.items()):
            if sub_idx >= 6:
                logger.warning(f"More than 6 sub-emotions for {base.name}, truncating")
                break
                
            sub = SubEmotion(
                id=sub_data.get("id", ""),
                name=sub_name,
                description=sub_data.get("description", ""),
                arousal_bias=sub_data.get("arousal_bias", 0.5),
                base_emotion_name=base.name,
                base_idx=base_idx,
                sub_idx=sub_idx,
            )
            
            # Parse sub-sub-emotions
            subsub_data = sub_data.get("sub_sub_emotions", {})
            for subsub_idx, (subsub_name, subsub_info) in enumerate(subsub_data.items()):
                if subsub_idx >= 6:
                    logger.warning(f"More than 6 sub-sub-emotions for {sub_name}, truncating")
                    break
                
                # Calculate node ID
                node_id = base_idx * 36 + sub_idx * 6 + subsub_idx
                
                subsub = SubSubEmotion(
                    id=subsub_info.get("id", ""),
                    name=subsub_name,
                    description=subsub_info.get("description", ""),
                    node_id=node_id,
                    sub_emotion_name=sub_name,
                    base_emotion_name=base.name,
                    base_idx=base_idx,
                    sub_idx=sub_idx,
                    subsub_idx=subsub_idx,
                )
                
                # Parse intensity tiers
                intensity_data = subsub_info.get("intensity_tiers", {})
                for tier_key, synonyms in intensity_data.items():
                    # tier_key is like "1_subtle", "2_mild", etc.
                    try:
                        tier_level = int(tier_key.split("_")[0])
                        tier_label = tier_key.split("_")[1] if "_" in tier_key else tier_key
                    except (ValueError, IndexError):
                        tier_level = 1
                        tier_label = tier_key
                    
                    tier = IntensityTier(
                        level=tier_level,
                        label=tier_label,
                        synonyms=synonyms if isinstance(synonyms, list) else [synonyms],
                    )
                    subsub.intensity_tiers[tier_key] = tier
                
                sub.sub_sub_emotions[subsub_name] = subsub
            
            base.sub_emotions[sub_name] = sub
        
        return base
    
    def _build_node_index(self) -> None:
        """Build flat index of all 216 nodes."""
        for base_name, base in self.base_emotions.items():
            for sub_name, sub in base.sub_emotions.items():
                for subsub_name, subsub in sub.sub_sub_emotions.items():
                    # Collect all synonyms from all intensity tiers
                    all_synonyms = []
                    for tier in subsub.intensity_tiers.values():
                        all_synonyms.extend(tier.synonyms)
                    
                    # Calculate global sub-emotion index (0-35)
                    sub_global_idx = subsub.base_idx * 6 + subsub.sub_idx
                    
                    node = EmotionNode(
                        node_id=subsub.node_id,
                        full_id=subsub.id,
                        name=subsub.name,
                        base_name=base.name,
                        sub_name=sub.name,
                        subsub_name=subsub.name,
                        base_idx=subsub.base_idx,
                        sub_idx=subsub.sub_idx,
                        subsub_idx=subsub.subsub_idx,
                        sub_global_idx=sub_global_idx,
                        description=subsub.description,
                        valence=base.valence,
                        arousal_bias=sub.arousal_bias,
                        all_synonyms=all_synonyms,
                    )
                    
                    self.nodes[subsub.node_id] = node
                    self.nodes_by_name[subsub.name] = node
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_node(self, node_id: int) -> Optional[EmotionNode]:
        """Get a node by ID (0-215)."""
        if not self.loaded:
            self.load()
        return self.nodes.get(node_id)
    
    def get_node_by_name(self, name: str) -> Optional[EmotionNode]:
        """Get a node by its sub-sub-emotion name."""
        if not self.loaded:
            self.load()
        return self.nodes_by_name.get(name.lower())
    
    def get_all_nodes(self) -> List[EmotionNode]:
        """Get all 216 nodes as a list."""
        if not self.loaded:
            self.load()
        return [self.nodes[i] for i in range(216) if i in self.nodes]
    
    def get_labels(
        self,
        node_id: int,
        intensity_tier: int = 3,  # 0-5
        key_idx: int = 0,  # 0-23
    ) -> ThesaurusLabels:
        """
        Get multi-head training labels for a sample.
        
        Args:
            node_id: The emotion node ID (0-215)
            intensity_tier: The intensity tier (0-5)
            key_idx: The musical key index (0-23)
        
        Returns:
            ThesaurusLabels with all classification targets
        """
        if not self.loaded:
            self.load()
        
        node = self.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Unknown node_id: {node_id}")
        
        return ThesaurusLabels(
            node_id=node_id,
            base_idx=node.base_idx,
            sub_global_idx=node.sub_global_idx,
            intensity_tier=intensity_tier,
            key_idx=key_idx,
        )
    
    def node_id_to_hierarchy(self, node_id: int) -> Tuple[int, int, int]:
        """Convert node ID to (base_idx, sub_idx, subsub_idx)."""
        base_idx = node_id // 36
        sub_idx = (node_id % 36) // 6
        subsub_idx = node_id % 6
        return base_idx, sub_idx, subsub_idx
    
    def hierarchy_to_node_id(
        self, base_idx: int, sub_idx: int, subsub_idx: int
    ) -> int:
        """Convert (base_idx, sub_idx, subsub_idx) to node ID."""
        return base_idx * 36 + sub_idx * 6 + subsub_idx
    
    def get_base_emotion_names(self) -> List[str]:
        """Get list of base emotion names in order."""
        return self.BASE_EMOTION_ORDER.copy()
    
    def get_sub_emotion_names(self, base_name: str) -> List[str]:
        """Get sub-emotion names for a base emotion."""
        if not self.loaded:
            self.load()
        base = self.base_emotions.get(base_name.upper())
        if base is None:
            return []
        return list(base.sub_emotions.keys())
    
    def get_key_name(self, key_idx: int) -> str:
        """Get key name from index."""
        if 0 <= key_idx < len(self.KEY_LABELS):
            return self.KEY_LABELS[key_idx]
        return "unknown"
    
    def get_key_idx(self, key_name: str) -> int:
        """Get key index from name."""
        try:
            return self.KEY_LABELS.index(key_name)
        except ValueError:
            return 0
    
    def get_intensity_label(self, tier: int) -> str:
        """Get intensity label from tier index."""
        if 0 <= tier < len(self.INTENSITY_LABELS):
            return self.INTENSITY_LABELS[tier]
        return "unknown"
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of loaded thesaurus."""
        if not self.loaded:
            self.load()
        
        return {
            "total_nodes": len(self.nodes),
            "base_emotions": len(self.base_emotions),
            "sub_emotions": sum(len(b.sub_emotions) for b in self.base_emotions.values()),
            "intensity_tiers": 6,
            "keys": 24,
            "expected_nodes": 216,
            "nodes_loaded": len(self.nodes) == 216,
            "base_emotion_names": list(self.base_emotions.keys()),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def load_thesaurus(
    thesaurus_dir: Optional[Path] = None,
) -> ThesaurusLoader:
    """
    Load the emotion thesaurus from default or specified location.
    
    Args:
        thesaurus_dir: Path to thesaurus directory. If None, uses default.
    
    Returns:
        Loaded ThesaurusLoader instance
    """
    if thesaurus_dir is None:
        # Try default locations
        possible_paths = [
            Path("data/emotion_thesaurus"),
            Path(__file__).parent.parent.parent.parent.parent / "data" / "emotion_thesaurus",
        ]
        for path in possible_paths:
            if path.exists():
                thesaurus_dir = path
                break
        else:
            raise FileNotFoundError("Could not find emotion thesaurus directory")
    
    loader = ThesaurusLoader(thesaurus_dir)
    loader.load()
    return loader


def get_node_label_tensor(
    node_id: int,
    intensity_tier: int = 3,
    key_idx: int = 0,
) -> Dict[str, int]:
    """
    Get label dictionary for a training sample.
    
    Returns dict suitable for PyTorch training with multi-head model.
    """
    loader = load_thesaurus()
    labels = loader.get_labels(node_id, intensity_tier, key_idx)
    
    return {
        "emotion_node": labels.node_id,
        "base_emotion": labels.base_idx,
        "sub_emotion": labels.sub_global_idx,
        "intensity_tier": labels.intensity_tier,
        "key_detection": labels.key_idx,
    }


def validate_thesaurus_completeness(thesaurus_dir: Path) -> Dict[str, Any]:
    """
    Validate that the thesaurus has all 216 expected nodes.
    
    Returns validation report with any missing nodes.
    """
    loader = ThesaurusLoader(thesaurus_dir)
    loader.load()
    
    missing_nodes = []
    for node_id in range(216):
        if node_id not in loader.nodes:
            base_idx, sub_idx, subsub_idx = loader.node_id_to_hierarchy(node_id)
            missing_nodes.append({
                "node_id": node_id,
                "expected_location": f"base[{base_idx}]/sub[{sub_idx}]/subsub[{subsub_idx}]",
            })
    
    return {
        "total_expected": 216,
        "total_loaded": len(loader.nodes),
        "complete": len(missing_nodes) == 0,
        "missing_count": len(missing_nodes),
        "missing_nodes": missing_nodes,
    }


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path("data/emotion_thesaurus")
    
    if path.exists():
        loader = load_thesaurus(path)
        print(f"Summary: {loader.summary()}")
        
        # Show a few example nodes
        print("\nExample nodes:")
        for node_id in [0, 6, 36, 108, 215]:
            node = loader.get_node(node_id)
            if node:
                print(f"  [{node_id:3d}] {node.base_name}/{node.sub_name}/{node.name}")
    else:
        print(f"Path not found: {path}")


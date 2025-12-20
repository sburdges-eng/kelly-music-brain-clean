#!/usr/bin/env python3
"""
Generate consolidated 216-node emotion thesaurus JSON from hierarchical files.

This script reads the hierarchical emotion thesaurus JSON files (joy.json, sad.json, etc.)
and converts them into a flat 216-node structure that NodeMLMapper can load.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

# Base VAD profiles for each emotion category
BASE_VAD_PROFILES = {
    "joy": {"valence": 0.8, "arousal": 0.6, "dominance": 0.6},
    "sad": {"valence": -0.6, "arousal": 0.3, "dominance": 0.3},
    "anger": {"valence": -0.5, "arousal": 0.8, "dominance": 0.8},
    "fear": {"valence": -0.7, "arousal": 0.7, "dominance": 0.2},
    "surprise": {"valence": 0.3, "arousal": 0.8, "dominance": 0.5},
    "disgust": {"valence": -0.6, "arousal": 0.5, "dominance": 0.6},
}

# Mode mapping
MODE_MAPPING = {
    "joy": "major",
    "sad": "minor",
    "anger": "minor",
    "fear": "minor",
    "surprise": "major",
    "disgust": "minor",
}

def parse_hierarchical_emotion(file_path: Path) -> Dict[str, Any]:
    """Parse a hierarchical emotion JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_nodes_from_hierarchy(emotion_data: Dict, category: str, base_id: int) -> List[Dict]:
    """Generate 36 nodes per category (6 sub-emotions × 6 intensity levels)."""
    nodes = []
    node_id = base_id
    
    # Get base VAD profile
    base_vad = BASE_VAD_PROFILES.get(category.lower(), {"valence": 0.0, "arousal": 0.5, "dominance": 0.5})
    base_mode = MODE_MAPPING.get(category.lower(), "major")
    
    # Get sub-emotions
    sub_emotions = emotion_data.get("sub_emotions", {})
    
    # 6 sub-emotions per base emotion
    sub_emotion_list = list(sub_emotions.items())[:6]  # Limit to 6
    
    for sub_idx, (sub_name, sub_data) in enumerate(sub_emotion_list):
        sub_sub_emotions = sub_data.get("sub_sub_emotions", {})
        
        # Get first sub-sub-emotion as representative for this sub-emotion
        # We'll use its intensity tiers
        sub_sub_list = list(sub_sub_emotions.items())
        if not sub_sub_list:
            # Fallback: create synthetic intensity tiers
            sub_sub_data = {}
        else:
            _, sub_sub_data = sub_sub_list[0]
        
        intensity_tiers = sub_sub_data.get("intensity_tiers", {})
        
        # 6 intensity levels per sub-emotion
        for intensity_level in range(1, 7):
            tier_key = f"{intensity_level}_"
            tier_names = []
            
            # Find matching tier
            for key, names in intensity_tiers.items():
                if key.startswith(tier_key):
                    tier_names = names
                    break
            
            if not tier_names:
                # Fallback: use sub_name with intensity
                tier_names = [f"{sub_name.lower()}_{intensity_level}"]
            
            # Calculate VAD coordinates
            intensity_scale = intensity_level / 6.0  # 0.167 to 1.0
            sub_variation = (sub_idx - 2.5) / 5.0 * 0.3  # -0.15 to 0.15
            
            valence = max(-1.0, min(1.0, base_vad["valence"] + sub_variation))
            arousal = max(0.0, min(1.0, base_vad["arousal"] * (0.5 + intensity_scale * 0.5)))
            dominance = max(0.0, min(1.0, base_vad["dominance"] + sub_variation * 0.5))
            intensity = intensity_scale
            
            # Create node
            node = {
                "id": node_id,
                "name": tier_names[0] if tier_names else f"{category}_{sub_name}_{intensity_level}",
                "category": category.lower(),
                "subcategory": sub_name.lower(),
                "vad": {
                    "valence": round(valence, 3),
                    "arousal": round(arousal, 3),
                    "dominance": round(dominance, 3),
                    "intensity": round(intensity, 3)
                },
                "relatedEmotions": [],
                "mode": base_mode,
                "tempoMultiplier": round(1.0 + (arousal - 0.5) * 0.4, 2),
                "dynamicsScale": round(0.5 + intensity * 0.5, 2)
            }
            
            nodes.append(node)
            node_id += 1
    
    # Add related emotions (nearby nodes in same category)
    for i, node in enumerate(nodes):
        related = []
        # Add adjacent nodes in same category
        for offset in [-3, -2, -1, 1, 2, 3]:
            related_id = node["id"] + offset
            if 0 <= related_id < len(nodes) and nodes[related_id]["category"] == node["category"]:
                related.append(related_id)
        node["relatedEmotions"] = related
    
    return nodes

def main():
    """Main conversion function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    emotion_dir = project_root / "emotion_thesaurus"
    output_file = emotion_dir / "emotion_nodes.json"
    
    if not emotion_dir.exists():
        print(f"Error: Emotion thesaurus directory not found: {emotion_dir}")
        return 1
    
    all_nodes = []
    emotion_files = {
        "joy": emotion_dir / "joy.json",
        "sad": emotion_dir / "sad.json",
        "anger": emotion_dir / "anger.json",
        "fear": emotion_dir / "fear.json",
        "surprise": emotion_dir / "surprise.json",
        "disgust": emotion_dir / "disgust.json",
    }
    
    base_id = 0
    for category, file_path in emotion_files.items():
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping {category}")
            continue
        
        print(f"Processing {category}...")
        emotion_data = parse_hierarchical_emotion(file_path)
        nodes = generate_nodes_from_hierarchy(emotion_data, category, base_id)
        all_nodes.extend(nodes)
        base_id += len(nodes)
        print(f"  Generated {len(nodes)} nodes for {category}")
    
    # Ensure we have exactly 216 nodes
    if len(all_nodes) > 216:
        all_nodes = all_nodes[:216]
        print(f"Warning: Truncated to 216 nodes")
    elif len(all_nodes) < 216:
        print(f"Warning: Only {len(all_nodes)} nodes generated (expected 216)")
    
    # Create output structure
    output_data = {
        "version": "1.0",
        "total_nodes": len(all_nodes),
        "nodes": all_nodes
    }
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccessfully generated {len(all_nodes)} emotion nodes")
    print(f"Output file: {output_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())

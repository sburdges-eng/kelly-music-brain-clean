"""
Automates K2 research prompts via Cursor CLI (when available)
For now, use as a template generator
"""

RESEARCH_DOMAINS = {
    "audio_analysis": {
        "keywords": ["librosa", "audio feature extraction", "music analysis"],
        "tech_stack": ["Python", "NumPy", "SciPy"],
        "focus": ["tempo detection", "key detection", "emotion classification"]
    },
    "daw_integration": {
        "keywords": ["OSC", "MIDI", "DAW control", "Logic Pro"],
        "tech_stack": ["Python", "python-osc", "mido"],
        "focus": ["transport control", "parameter automation", "project sync"]
    },
    "juce_audio": {
        "keywords": ["JUCE", "VST3", "audio plugin", "DSP"],
        "tech_stack": ["C++17", "CMake", "JUCE 7"],
        "focus": ["plugin architecture", "real-time audio", "parameter handling"]
    },
    "music_theory": {
        "keywords": ["chord detection", "scale analysis", "harmony", "music theory"],
        "tech_stack": ["Python", "music21", "mingus"],
        "focus": ["chord progression", "voice leading", "scale detection"]
    },
    "emotion_mapping": {
        "keywords": ["emotion detection", "mood classification", "audio emotion"],
        "tech_stack": ["Python", "TensorFlow", "PyTorch"],
        "focus": ["emotion classification", "mood detection", "affective computing"]
    },
    "python_cpp_bridge": {
        "keywords": ["pybind11", "Python C++", "audio bridge", "real-time"],
        "tech_stack": ["C++", "Python", "pybind11"],
        "focus": ["real-time audio", "low latency", "data passing"]
    },
    "sample_management": {
        "keywords": ["sample library", "audio catalog", "tagging", "search"],
        "tech_stack": ["Python", "SQLite", "audio analysis"],
        "focus": ["cataloging", "tagging", "search algorithms"]
    }
}

def generate_research_prompt(domain, specific_question, context_files=None):
    """Generate Cursor K2 research prompt"""
    if domain not in RESEARCH_DOMAINS:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(RESEARCH_DOMAINS.keys())}")
    
    config = RESEARCH_DOMAINS[domain]
    
    context_refs = ""
    if context_files:
        context_refs = "\n".join([f"@{file}" for file in context_files])
    
    prompt = f"""
RESEARCH DOMAIN: {domain}

KEYWORDS: {', '.join(config['keywords'])}
TECH STACK: {', '.join(config['tech_stack'])}
FOCUS AREAS: {', '.join(config['focus'])}

SPECIFIC QUESTION: 
{specific_question}

CONTEXT FILES:
{context_refs if context_refs else '[Add relevant files from your project]'}

TASK: 
1. Search GitHub for repos matching keywords
2. Filter by: stars > 100, updated in 2024, active community
3. Analyze top 5-10 repos for patterns
4. Extract code examples
5. Provide recommendations for my kelly-music-brain-clean project

OUTPUT:
- Repository list with relevance scores
- Key patterns found (with code)
- Actionable recommendations
- Integration roadmap
"""
    
    return prompt

def save_prompt(domain, question, context_files=None, output_dir="research/prompts"):
    """Save generated prompt to file"""
    import os
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    prompt = generate_research_prompt(domain, question, context_files)
    
    # Create safe filename
    safe_domain = domain.replace("_", "-")
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{output_dir}/{safe_domain}-{timestamp}.md"
    
    with open(filename, "w") as f:
        f.write(prompt)
    
    return filename

# Example usage
if __name__ == "__main__":
    domains = [
        ("audio_analysis", "How do professional tools handle real-time audio feature extraction with minimal latency?", ["music_brain/analyzer.py"]),
        ("daw_integration", "What's the best approach for bidirectional communication with Logic Pro X?", ["music_brain/logic_pro.py"]),
        ("juce_audio", "How should I structure a multi-timbral VST3 plugin with complex routing?", ["iDAW_Core/PluginProcessor.cpp"]),
        ("emotion_mapping", "What are the state-of-the-art methods for detecting emotion from audio features?", ["music_brain/emotional_mapping.py"]),
        ("music_theory", "How do professional tools detect chord progressions and analyze harmonic movement?", ["music_brain/harmony_tools.py"]),
    ]
    
    print("Generating research prompts...\n")
    for domain, question, context in domains:
        filename = save_prompt(domain, question, context_files=context, output_dir="prompts")
        print(f"✅ Generated: {filename}")
    
    print("\n📋 Next steps:")
    print("1. Review prompts in research/prompts/")
    print("2. Copy a prompt to Cursor Composer (Cmd+I)")
    print("3. Let K2 research and provide recommendations")
    print("4. Document findings in research/findings/")

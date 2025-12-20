#include "engine/ColorFrequencyMapper.h"
#include <algorithm>
#include <cmath>

namespace kelly {

ColorFrequencyMapper::ColorFrequencyMapper() {
    initializeEmotionColors();
}

void ColorFrequencyMapper::initializeEmotionColors() {
    // Initialize emotion-to-color mappings
    emotionColorMap_["Joy"] = {580.0f, 517.0f, 2.14f, {1.0f, 0.9f, 0.0f}, "Yellow"};
    emotionColorMap_["Sadness"] = {470.0f, 638.0f, 2.64f, {0.2f, 0.4f, 0.8f}, "Blue"};
    emotionColorMap_["Anger"] = {620.0f, 484.0f, 2.00f, {0.9f, 0.1f, 0.1f}, "Red"};
    emotionColorMap_["Fear"] = {400.0f, 749.0f, 3.10f, {0.6f, 0.2f, 0.8f}, "Violet"};
    emotionColorMap_["Love"] = {540.0f, 556.0f, 2.30f, {0.2f, 0.8f, 0.3f}, "Green"};
    emotionColorMap_["Trust"] = {500.0f, 600.0f, 2.48f, {0.3f, 0.7f, 0.9f}, "Cyan"};
    emotionColorMap_["Surprise"] = {550.0f, 545.0f, 2.25f, {1.0f, 0.6f, 0.0f}, "Orange"};
    emotionColorMap_["Disgust"] = {520.0f, 577.0f, 2.38f, {0.4f, 0.6f, 0.2f}, "Olive"};
}

ColorMapping ColorFrequencyMapper::emotionToColor(const VADState& vad) const {
    // Map VAD to color frequency
    float freq = valenceToFrequency(vad.valence);
    float wavelength = frequencyToWavelength(freq);
    
    ColorMapping mapping;
    mapping.wavelength = wavelength;
    mapping.frequency = freq;
    mapping.energy = 1240.0f / wavelength;  // E(eV) = 1240 / λ(nm)
    
    wavelengthToRGB(wavelength, mapping.rgb);
    
    // Name based on valence
    if (vad.valence > 0.5f) {
        mapping.name = "Warm";
    } else if (vad.valence < -0.5f) {
        mapping.name = "Cool";
    } else {
        mapping.name = "Neutral";
    }
    
    return mapping;
}

ColorMapping ColorFrequencyMapper::emotionNameToColor(const std::string& emotionName) const {
    auto it = emotionColorMap_.find(emotionName);
    if (it != emotionColorMap_.end()) {
        return it->second;
    }
    
    // Default: neutral gray
    ColorMapping defaultMapping;
    defaultMapping.wavelength = 550.0f;
    defaultMapping.frequency = 545.0f;
    defaultMapping.energy = 2.25f;
    defaultMapping.rgb[0] = defaultMapping.rgb[1] = defaultMapping.rgb[2] = 0.5f;
    defaultMapping.name = "Neutral";
    return defaultMapping;
}

float ColorFrequencyMapper::valenceToFrequency(float valence, float fMin, float fMax) const {
    // f_color = f_min + (V+1)(f_max - f_min)/2
    float normalizedValence = (valence + 1.0f) / 2.0f;  // Map [-1,1] to [0,1]
    return fMin + normalizedValence * (fMax - fMin);
}

void ColorFrequencyMapper::wavelengthToRGB(float wavelength, float rgb[3]) const {
    // Simplified wavelength to RGB conversion
    // Based on standard color matching functions
    
    float r = 0.0f, g = 0.0f, b = 0.0f;
    
    if (wavelength >= 380.0f && wavelength < 440.0f) {
        r = -(wavelength - 440.0f) / (440.0f - 380.0f);
        g = 0.0f;
        b = 1.0f;
    } else if (wavelength >= 440.0f && wavelength < 490.0f) {
        r = 0.0f;
        g = (wavelength - 440.0f) / (490.0f - 440.0f);
        b = 1.0f;
    } else if (wavelength >= 490.0f && wavelength < 510.0f) {
        r = 0.0f;
        g = 1.0f;
        b = -(wavelength - 510.0f) / (510.0f - 490.0f);
    } else if (wavelength >= 510.0f && wavelength < 580.0f) {
        r = (wavelength - 510.0f) / (580.0f - 510.0f);
        g = 1.0f;
        b = 0.0f;
    } else if (wavelength >= 580.0f && wavelength < 645.0f) {
        r = 1.0f;
        g = -(wavelength - 645.0f) / (645.0f - 580.0f);
        b = 0.0f;
    } else if (wavelength >= 645.0f && wavelength <= 780.0f) {
        r = 1.0f;
        g = 0.0f;
        b = 0.0f;
    }
    
    // Adjust brightness
    float brightness = 1.0f;
    if (wavelength >= 380.0f && wavelength < 420.0f) {
        brightness = 0.3f + 0.7f * (wavelength - 380.0f) / (420.0f - 380.0f);
    } else if (wavelength >= 420.0f && wavelength < 700.0f) {
        brightness = 1.0f;
    } else if (wavelength >= 700.0f && wavelength <= 780.0f) {
        brightness = 0.3f + 0.7f * (780.0f - wavelength) / (780.0f - 700.0f);
    }
    
    rgb[0] = std::clamp(r * brightness, 0.0f, 1.0f);
    rgb[1] = std::clamp(g * brightness, 0.0f, 1.0f);
    rgb[2] = std::clamp(b * brightness, 0.0f, 1.0f);
}

float ColorFrequencyMapper::frequencyToWavelength(float frequencyTHz) const {
    // λ = c / f, where c = 3×10⁸ m/s = 3×10¹⁴ nm/s
    // f in THz, so: λ(nm) = 3×10⁵ / f(THz)
    if (frequencyTHz <= 0.0f) return 0.0f;
    return 300000.0f / frequencyTHz;
}

float ColorFrequencyMapper::wavelengthToFrequency(float wavelengthNm) const {
    // f = c / λ
    if (wavelengthNm <= 0.0f) return 0.0f;
    return 300000.0f / wavelengthNm;
}

} // namespace kelly

#pragma once

#include "engine/VADCalculator.h"
#include <string>
#include <map>

namespace kelly {

/**
 * Color and Frequency Mapping
 * 
 * Maps emotions to:
 * - Wavelength (nm)
 * - Frequency (THz)
 * - Energy (eV)
 * - RGB values
 */

struct ColorMapping {
    float wavelength;      // nm
    float frequency;       // THz
    float energy;          // eV
    float rgb[3];          // RGB values (0-1)
    std::string name;      // Color name
};

/**
 * Color Frequency Mapper
 */
class ColorFrequencyMapper {
public:
    ColorFrequencyMapper();
    
    /**
     * Map emotion to color based on VAD
     */
    ColorMapping emotionToColor(const VADState& vad) const;
    
    /**
     * Map emotion name to color
     */
    ColorMapping emotionNameToColor(const std::string& emotionName) const;
    
    /**
     * Calculate color frequency from valence
     * f_color = f_min + (V+1)(f_max - f_min)/2
     */
    float valenceToFrequency(float valence, float fMin = 400.0f, float fMax = 750.0f) const;
    
    /**
     * Convert wavelength to RGB
     */
    void wavelengthToRGB(float wavelength, float rgb[3]) const;
    
    /**
     * Convert frequency to wavelength
     */
    float frequencyToWavelength(float frequencyTHz) const;
    
    /**
     * Convert wavelength to frequency
     */
    float wavelengthToFrequency(float wavelengthNm) const;
    
private:
    std::map<std::string, ColorMapping> emotionColorMap_;
    
    void initializeEmotionColors();
};

} // namespace kelly

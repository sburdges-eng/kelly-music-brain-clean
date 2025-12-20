#pragma once

#include <juce_core/juce_core.h>
#include <vector>

namespace kelly {

/**
 * PathResolver - Centralized path resolution for data files
 * 
 * Provides a consistent way to find data files across the plugin,
 * with multiple fallback paths for different deployment scenarios.
 */
class PathResolver {
public:
    /**
     * Get all search paths in priority order
     */
    static std::vector<juce::File> getSearchPaths();
    
    /**
     * Find a data file by name
     * @param filename The filename to search for (e.g., "eq_presets.json")
     * @return The first existing file found, or empty File if not found
     */
    static juce::File findDataFile(const juce::String& filename);
    
    /**
     * Find a data directory
     * @return The first existing data directory found, or empty File if not found
     */
    static juce::File findDataDirectory();
    
    /**
     * Get the user data directory (creates if needed)
     */
    static juce::File getUserDataDirectory();
};

} // namespace kelly

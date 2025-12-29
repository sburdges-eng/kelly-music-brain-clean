/**
 * @file PluginInfo.h
 * @brief Information about an external audio plugin.
 */

#pragma once

#include <string>
#include <vector>

namespace daiw {
namespace plugins {

/**
 * Supported plugin formats.
 */
enum class PluginFormat {
    VST3,
    AU,
    CLAP,
    LV2,
    Internal
};

/**
 * Metadata for an external audio plugin.
 */
struct PluginInfo {
    std::string name;
    std::string manufacturer;
    std::string version;
    std::string identifier;   // Unique identifier (e.g., VST3 GUID or AU FourCC)
    std::string path;         // Filesystem path to the plugin
    PluginFormat format;
    bool isInstrument = false;
    int numInputs = 0;
    int numOutputs = 0;
    size_t latencySamples = 0;
    bool isShell = false;     // True if this is a shell plugin (like Waves)
};

} // namespace plugins
} // namespace daiw


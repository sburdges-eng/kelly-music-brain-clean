#include "common/PathResolver.h"
#include <juce_core/juce_core.h>

namespace kelly {

std::vector<juce::File> PathResolver::getSearchPaths() {
    std::vector<juce::File> searchPaths;

    auto appFile = juce::File::getSpecialLocation(juce::File::currentApplicationFile);
    auto execFile = juce::File::getSpecialLocation(juce::File::currentExecutableFile);

#if JUCE_MAC || JUCE_IOS
    // 1. macOS App bundle Resources folder (for standalone app)
    // Structure: App.app/Contents/Resources/data/
    if (appFile.exists() && appFile.getFileExtension() == ".app") {
        auto resourcesPath = appFile.getChildFile("Contents").getChildFile("Resources").getChildFile("data");
        searchPaths.push_back(resourcesPath);
    }

    // 2. Plugin bundle Resources folder (macOS AU/VST3)
    // Structure: Plugin.component/Contents/MacOS/Plugin -> Plugin.component/Contents/Resources/data/
    // Structure: Plugin.vst3/Contents/MacOS/Plugin -> Plugin.vst3/Contents/Resources/data/
    if (execFile.exists()) {
        auto parentDir = execFile.getParentDirectory();
        auto contentsDir = parentDir.getParentDirectory();
        // Check if we're in a bundle structure (Contents/MacOS/Plugin or Contents/x86_64-mac/Plugin)
        // Verify we have a Contents directory and it has a Resources subdirectory
        if (parentDir.getFileName().equalsIgnoreCase("MacOS") ||
            parentDir.getFileName().contains("-mac")) {
            if (contentsDir.getFileName() == "Contents" && contentsDir.getChildFile("Resources").exists()) {
                auto resourcesPath = contentsDir.getChildFile("Resources").getChildFile("data");
                searchPaths.push_back(resourcesPath);
            }
        }
    }
#elif JUCE_WINDOWS
    // 1. Windows VST3 plugin Resources folder
    // Structure: Plugin.vst3/Contents/Resources/data/ (VST3 on Windows also uses Contents/Resources)
    if (execFile.exists()) {
        auto parentDir = execFile.getParentDirectory();
        // Check if we're in a VST3 bundle (Contents/x86_64-win/Plugin.exe or Contents/x86-win/Plugin.exe)
        if (parentDir.getParentDirectory().getChildFile("Resources").exists()) {
            auto resourcesPath = parentDir.getParentDirectory().getChildFile("Resources").getChildFile("data");
            searchPaths.push_back(resourcesPath);
        }
    }

    // 2. Windows VST2 plugin location
    // Structure: Plugin.dll -> same directory/data/
    if (execFile.exists() && execFile.getFileExtension() == ".dll") {
        auto pluginDir = execFile.getParentDirectory();
        // Check if this might be a VST2 plugin directory
        auto dataPath = pluginDir.getChildFile("data");
        searchPaths.push_back(dataPath);
    }
#elif JUCE_LINUX
    // Linux plugin locations
    if (execFile.exists()) {
        auto parentDir = execFile.getParentDirectory();
        // VST3 on Linux: Plugin.vst3/Contents/x86_64-linux/Plugin.so -> Contents/Resources/data/
        if (parentDir.getParentDirectory().getChildFile("Resources").exists()) {
            auto resourcesPath = parentDir.getParentDirectory().getChildFile("Resources").getChildFile("data");
            searchPaths.push_back(resourcesPath);
        }
    }
#endif

    // 3. Same directory as executable (development)
    // Structure: ./Plugin (executable) -> ./data/
    if (execFile.exists()) {
        searchPaths.push_back(execFile.getParentDirectory().getChildFile("data"));
    }

    // 4. User's Application Support folder
    // Structure: ~/Library/Application Support/Kelly MIDI Companion/data/
    auto appSupport = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory);
    searchPaths.push_back(appSupport.getChildFile("Kelly MIDI Companion").getChildFile("data"));

    // 5. Common data locations (Windows)
    // Structure: C:/ProgramData/Kelly MIDI Companion/data/
    auto commonAppData = juce::File::getSpecialLocation(juce::File::commonApplicationDataDirectory);
    searchPaths.push_back(commonAppData.getChildFile("Kelly MIDI Companion").getChildFile("data"));

    // 6. Development fallback - relative to working directory
    // Structure: ./data/ (relative to current working directory)
    searchPaths.push_back(juce::File::getCurrentWorkingDirectory().getChildFile("data"));

    // 7. Check for emotions subdirectory in each path (for backward compatibility)
    // Some deployments may have data/emotions/ structure
    std::vector<juce::File> additionalPaths;
    for (const auto& path : searchPaths) {
        if (path.isDirectory()) {
            auto emotionsPath = path.getChildFile("emotions");
            if (emotionsPath.isDirectory()) {
                additionalPaths.push_back(emotionsPath);
            }
        }
    }
    searchPaths.insert(searchPaths.end(), additionalPaths.begin(), additionalPaths.end());

    return searchPaths;
}

juce::File PathResolver::findDataFile(const juce::String& filename) {
    auto searchPaths = getSearchPaths();

    for (const auto& dir : searchPaths) {
        if (dir.isDirectory()) {
            juce::File file = dir.getChildFile(filename);
            if (file.existsAsFile()) {
                juce::Logger::writeToLog("PathResolver: Found data file '" + filename + "' at " + file.getFullPathName());
                return file;
            }
        }
    }

    juce::Logger::writeToLog("PathResolver: Data file '" + filename + "' not found in any search path");
    return juce::File();
}

juce::File PathResolver::findDataDirectory() {
    auto searchPaths = getSearchPaths();

    // Find first existing directory
    for (size_t i = 0; i < searchPaths.size(); ++i) {
        const auto& dir = searchPaths[i];
        if (dir.isDirectory()) {
            // Check for any data file to confirm it's a data directory
            if (dir.getChildFile("sad.json").existsAsFile() ||
                dir.getChildFile("happy.json").existsAsFile() ||
                dir.getChildFile("eq_presets.json").existsAsFile()) {
                juce::Logger::writeToLog("PathResolver: Found data directory at " + dir.getFullPathName() + " (path #" + juce::String(i + 1) + ")");
                return dir;
            } else {
                juce::Logger::writeToLog("PathResolver: Checked path #" + juce::String(i + 1) + ": " + dir.getFullPathName() + " (directory exists but no data files found)");
            }
        } else {
            juce::Logger::writeToLog("PathResolver: Checked path #" + juce::String(i + 1) + ": " + dir.getFullPathName() + " (does not exist)");
        }
    }

    juce::Logger::writeToLog("PathResolver: No data directory found in any search path");
    return juce::File();
}

juce::File PathResolver::getUserDataDirectory() {
    auto appSupport = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory);
    auto userDataDir = appSupport.getChildFile("Kelly MIDI Companion").getChildFile("data");

    // Create directory if it doesn't exist
    if (!userDataDir.exists()) {
        userDataDir.createDirectory();
    }

    return userDataDir;
}

} // namespace kelly

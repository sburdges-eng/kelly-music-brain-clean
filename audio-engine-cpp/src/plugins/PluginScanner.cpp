/**
 * @file PluginScanner.cpp
 * @brief Implementation of the PluginScanner class.
 */

#include "PluginScanner.h"
#include "PluginSandbox.h"
#include <chrono>
#include <iostream>
#include <filesystem>

namespace daiw {
namespace plugins {

PluginScanner::PluginScanner(PluginQuarantine& quarantine) : quarantine_(quarantine) {}

PluginScanner::~PluginScanner() {
    cancel();
    if (scanThread_.joinable()) {
        scanThread_.join();
    }
}

void PluginScanner::scanPaths(const std::vector<std::string>& paths, 
                             ProgressCallback onProgress, 
                             ResultCallback onComplete) {
    if (isScanning_) return;
    
    isScanning_ = true;
    shouldCancel_ = false;
    foundPlugins_.clear();
    
    // Clean up old thread if it exists
    if (scanThread_.joinable()) {
        scanThread_.join();
    }
    
    scanThread_ = std::thread(&PluginScanner::runScan, this, paths, onProgress, onComplete);
}

void PluginScanner::cancel() {
    shouldCancel_ = true;
}

void PluginScanner::runScan(std::vector<std::string> paths, 
                           ProgressCallback onProgress, 
                           ResultCallback onComplete) {
    std::vector<std::string> allFiles;
    try {
        for (const auto& path : paths) {
            if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
                for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
                    if (shouldCancel_) break;
                    
                    // Basic filter for plugin extensions
                    auto ext = entry.path().extension().string();
                    if (ext == ".vst3" || ext == ".component" || ext == ".vst" || ext == ".clap") {
                        allFiles.push_back(entry.path().string());
                    }
                }
            }
            if (shouldCancel_) break;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during directory iteration: " << e.what() << std::endl;
    }
    
    size_t total = allFiles.size();
    for (size_t i = 0; i < total; ++i) {
        if (shouldCancel_) break;
        
        const auto& file = allFiles[i];
        if (onProgress) {
            onProgress(static_cast<float>(i) / static_cast<float>(total), file);
        }
        
        // Skip if quarantined
        if (quarantine_.isQuarantined(file)) {
            continue;
        }
        
        // Perform scan with isolation (P1-008)
        auto sandbox = PluginSandbox::create();
        if (sandbox->launch(file, 9000 + static_cast<int>(i))) {
            auto startTime = std::chrono::steady_clock::now();
            bool timedOut = true;
            
            while (std::chrono::steady_clock::now() - startTime < std::chrono::seconds(timeoutSeconds_)) {
                if (!sandbox->isAlive()) {
                    timedOut = false;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            if (timedOut) {
                sandbox->terminate();
                quarantine_.quarantine(file, QuarantineReason::Timeout);
            } else {
                std::string output = sandbox->getOutput();
                PluginInfo info;
                info.path = file;
                
                // Parse basic output from host process
                if (output.find("RESULT:NAME=") != std::string::npos) {
                    size_t namePos = output.find("RESULT:NAME=") + 12;
                    size_t nameEnd = output.find("\n", namePos);
                    info.name = output.substr(namePos, nameEnd - namePos);
                }
                
                if (output.find("RESULT:FORMAT=") != std::string::npos) {
                    size_t formatPos = output.find("RESULT:FORMAT=") + 14;
                    size_t formatEnd = output.find("\n", formatPos);
                    std::string formatStr = output.substr(formatPos, formatEnd - formatPos);
                    if (formatStr.find("VST3") != std::string::npos) info.format = PluginFormat::VST3;
                    else if (formatStr.find("AU") != std::string::npos) info.format = PluginFormat::AU;
                }

                if (!info.name.empty()) {
                    foundPlugins_.push_back(info);
                } else if (scanSinglePlugin(file, info)) { 
                    foundPlugins_.push_back(info);
                }
            }
        } else {
            quarantine_.quarantine(file, QuarantineReason::Crash);
        }
    }
    
    isScanning_ = false;
    if (onComplete) {
        onComplete(foundPlugins_);
    }
}

bool PluginScanner::scanSinglePlugin(const std::string& path, PluginInfo& info) {
    // Simulation of plugin loading logic.
    // In a real implementation, this would use JUCE's AudioPluginFormatManager
    // or call out to a separate process.
    
    // For now, we simulate a small delay to represent loading time.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Basic heuristics to fill info
    std::filesystem::path p(path);
    info.path = path;
    info.name = p.stem().string();
    
    auto ext = p.extension().string();
    if (ext == ".vst3") info.format = PluginFormat::VST3;
    else if (ext == ".component") info.format = PluginFormat::AU;
    else if (ext == ".clap") info.format = PluginFormat::CLAP;
    else info.format = PluginFormat::Internal;
    
    return true;
}

} // namespace plugins
} // namespace daiw

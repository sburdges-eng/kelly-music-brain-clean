#pragma once

#include "PluginSandbox.h"
#include <map>
#include <string>
#include <memory>

namespace kelly {
namespace plugins {

/**
 * PluginManager - Manages all active plugins, hosting them in sandboxes.
 * 
 * Part of Phase 1: Audio Engine Core (P1-008)
 */
class PluginManager {
public:
    PluginManager();
    ~PluginManager() = default;

    /**
     * Load a plugin at the specified path into a new sandbox.
     */
    bool loadPlugin(const std::string& name, const std::string& path);

    /**
     * Unload a plugin.
     */
    void unloadPlugin(const std::string& name);

    /**
     * Send a command to a specific plugin.
     */
    bool pluginCommand(const std::string& name, const std::string& command, const std::string& args);

    /**
     * Get the state of a plugin sandbox.
     */
    PluginSandbox::State getPluginState(const std::string& name) const;

private:
    std::map<std::string, std::unique_ptr<PluginSandbox>> activePlugins_;
};

} // namespace plugins
} // namespace kelly


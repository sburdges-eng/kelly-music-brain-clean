#include "PluginManager.h"

namespace kelly {
namespace plugins {

PluginManager::PluginManager() = default;

bool PluginManager::loadPlugin(const std::string& name, const std::string& path) {
    if (activePlugins_.find(name) != activePlugins_.end()) {
        return false; // Already loaded
    }

    auto sandbox = std::make_unique<PluginSandbox>(path);
    if (sandbox->start()) {
        activePlugins_[name] = std::move(sandbox);
        return true;
    }
    return false;
}

void PluginManager::unloadPlugin(const std::string& name) {
    activePlugins_.erase(name);
}

bool PluginManager::pluginCommand(const std::string& name, const std::string& command, const std::string& args) {
    auto it = activePlugins_.find(name);
    if (it != activePlugins_.end()) {
        return it->second->sendCommand(command, args);
    }
    return false;
}

PluginSandbox::State PluginManager::getPluginState(const std::string& name) const {
    auto it = activePlugins_.find(name);
    if (it != activePlugins_.end()) {
        return it->second->getState();
    }
    return PluginSandbox::State::Stopped;
}

} // namespace plugins
} // namespace kelly


/**
 * @file PluginQuarantine.h
 * @brief Manages a list of plugins that are blocked due to crashes or timeouts.
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include "PluginInfo.h"

namespace daiw {
namespace plugins {

/**
 * Reasons why a plugin might be quarantined.
 */
enum class QuarantineReason {
    Crash,
    Timeout,
    InvalidMetadata,
    UserBlocked
};

/**
 * Entry in the quarantine list.
 */
struct QuarantineEntry {
    std::string pluginPath;
    QuarantineReason reason;
    std::string timestamp;
    int retryCount = 0;
};

/**
 * PluginQuarantine manages persistent storage of blocked plugins.
 */
class PluginQuarantine {
public:
    explicit PluginQuarantine(const std::string& dbPath);
    ~PluginQuarantine();

    /**
     * Add a plugin to the quarantine list.
     */
    void quarantine(const std::string& path, QuarantineReason reason);

    /**
     * Remove a plugin from the quarantine list.
     */
    void unquarantine(const std::string& path);

    /**
     * Check if a plugin is currently quarantined.
     */
    bool isQuarantined(const std::string& path) const;

    /**
     * Get all currently quarantined entries.
     */
    std::vector<QuarantineEntry> getAllEntries() const;

    /**
     * Clear all quarantine entries.
     */
    void clear();

private:
    std::string dbPath_;
    mutable std::mutex mutex_;
    
    void initDb();
    std::string reasonToString(QuarantineReason reason) const;
    QuarantineReason stringToReason(const std::string& str) const;
};

} // namespace plugins
} // namespace daiw

/**
 * @file PluginScanner.h
 * @brief Background scanner for audio plugins with timeout support.
 */

#pragma once

#include "PluginInfo.h"
#include "PluginQuarantine.h"
#include <atomic>
#include <functional>
#include <future>
#include <string>
#include <thread>
#include <vector>

namespace daiw {
namespace plugins {

/**
 * PluginScanner manages background scanning of plugin directories.
 * Implements timeout and quarantine logic to ensure stability.
 */
class PluginScanner {
public:
  using ProgressCallback =
      std::function<void(float progress, const std::string &currentPlugin)>;
  using ResultCallback =
      std::function<void(const std::vector<PluginInfo> &foundPlugins)>;

  explicit PluginScanner(PluginQuarantine &quarantine);
  ~PluginScanner();

  /**
   * Start a background scan of the provided paths.
   */
  void scanPaths(const std::vector<std::string> &paths,
                 ProgressCallback onProgress, ResultCallback onComplete);

  /**
   * Cancel the current scan.
   */
  void cancel();

  /**
   * Check if a scan is currently in progress.
   */
  bool isScanning() const { return isScanning_; }

  /**
   * Set the timeout for scanning an individual plugin in seconds.
   */
  void setTimeout(int seconds) { timeoutSeconds_ = seconds; }

private:
  PluginQuarantine &quarantine_;
  std::vector<PluginInfo> foundPlugins_;
  std::atomic<bool> isScanning_{false};
  std::atomic<bool> shouldCancel_{false};
  int timeoutSeconds_ = 5;

  std::thread scanThread_;

  void runScan(std::vector<std::string> paths, ProgressCallback onProgress,
               ResultCallback onComplete);

  /**
   * Low-level plugin scan logic.
   * In P1-008, this will be moved to a separate process.
   */
  bool scanSinglePlugin(const std::string &path, PluginInfo &info);
};

} // namespace plugins
} // namespace daiw

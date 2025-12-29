/**
 * @file PluginQuarantine.cpp
 * @brief Implementation of the PluginQuarantine class.
 */

#include "PluginQuarantine.h"
#include <sqlite3.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>

namespace daiw {
namespace plugins {

PluginQuarantine::PluginQuarantine(const std::string& dbPath) : dbPath_(dbPath) {
    initDb();
}

PluginQuarantine::~PluginQuarantine() = default;

void PluginQuarantine::initDb() {
    sqlite3* db;
    if (sqlite3_open(dbPath_.c_str(), &db) == SQLITE_OK) {
        const char* sql = "CREATE TABLE IF NOT EXISTS quarantine ("
                          "path TEXT PRIMARY KEY, "
                          "reason TEXT, "
                          "timestamp TEXT, "
                          "retry_count INTEGER DEFAULT 0);";
        char* errMsg = nullptr;
        if (sqlite3_exec(db, sql, nullptr, nullptr, &errMsg) != SQLITE_OK) {
            std::cerr << "SQLite error: " << errMsg << std::endl;
            sqlite3_free(errMsg);
        }
        sqlite3_close(db);
    }
}

void PluginQuarantine::quarantine(const std::string& path, QuarantineReason reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    sqlite3* db;
    if (sqlite3_open(dbPath_.c_str(), &db) == SQLITE_OK) {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
        
        const char* sql = "INSERT OR REPLACE INTO quarantine (path, reason, timestamp, retry_count) "
                          "VALUES (?, ?, ?, COALESCE((SELECT retry_count FROM quarantine WHERE path = ?) + 1, 0));";
        
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, path.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, reasonToString(reason).c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 3, ss.str().c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 4, path.c_str(), -1, SQLITE_TRANSIENT);
            
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
        sqlite3_close(db);
    }
}

void PluginQuarantine::unquarantine(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    sqlite3* db;
    if (sqlite3_open(dbPath_.c_str(), &db) == SQLITE_OK) {
        const char* sql = "DELETE FROM quarantine WHERE path = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, path.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
        sqlite3_close(db);
    }
}

bool PluginQuarantine::isQuarantined(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    bool result = false;
    sqlite3* db;
    if (sqlite3_open(dbPath_.c_str(), &db) == SQLITE_OK) {
        const char* sql = "SELECT COUNT(*) FROM quarantine WHERE path = ?;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, path.c_str(), -1, SQLITE_TRANSIENT);
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                result = sqlite3_column_int(stmt, 0) > 0;
            }
            sqlite3_finalize(stmt);
        }
        sqlite3_close(db);
    }
    return result;
}

std::vector<QuarantineEntry> PluginQuarantine::getAllEntries() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<QuarantineEntry> entries;
    sqlite3* db;
    if (sqlite3_open(dbPath_.c_str(), &db) == SQLITE_OK) {
        const char* sql = "SELECT path, reason, timestamp, retry_count FROM quarantine;";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                QuarantineEntry entry;
                entry.pluginPath = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
                entry.reason = stringToReason(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
                entry.timestamp = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
                entry.retryCount = sqlite3_column_int(stmt, 3);
                entries.push_back(entry);
            }
            sqlite3_finalize(stmt);
        }
        sqlite3_close(db);
    }
    return entries;
}

void PluginQuarantine::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    sqlite3* db;
    if (sqlite3_open(dbPath_.c_str(), &db) == SQLITE_OK) {
        const char* sql = "DELETE FROM quarantine;";
        sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
        sqlite3_close(db);
    }
}

std::string PluginQuarantine::reasonToString(QuarantineReason reason) const {
    switch (reason) {
        case QuarantineReason::Crash: return "crash";
        case QuarantineReason::Timeout: return "timeout";
        case QuarantineReason::InvalidMetadata: return "invalid_metadata";
        case QuarantineReason::UserBlocked: return "user_blocked";
        default: return "unknown";
    }
}

QuarantineReason PluginQuarantine::stringToReason(const std::string& str) const {
    if (str == "crash") return QuarantineReason::Crash;
    if (str == "timeout") return QuarantineReason::Timeout;
    if (str == "invalid_metadata") return QuarantineReason::InvalidMetadata;
    if (str == "user_blocked") return QuarantineReason::UserBlocked;
    return QuarantineReason::Crash; // Default
}

} // namespace plugins
} // namespace daiw

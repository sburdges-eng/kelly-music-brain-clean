/**
 * @file PluginSandbox.cpp
 * @brief Platform-specific implementation of PluginSandbox.
 */

#include "PluginSandbox.h"
#include <iostream>
#include <thread>
#include <chrono>

#if defined(__APPLE__)
#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <vector>

extern char **environ;

namespace daiw {
namespace plugins {

/**
 * @class MacOSPluginSandbox
 * @brief macOS-specific implementation using posix_spawn.
 */
class MacOSPluginSandbox : public PluginSandbox {
public:
    MacOSPluginSandbox() : pid_(-1) {
        pipeFds_[0] = -1;
        pipeFds_[1] = -1;
    }
    ~MacOSPluginSandbox() override { 
        terminate(); 
        closePipes();
    }

    bool launch(const std::string& pluginPath, int ipcPort) override {
        if (pid_ > 0) terminate();
        closePipes();

        if (pipe(pipeFds_) != 0) {
            return false;
        }

        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, pipeFds_[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&actions, pipeFds_[0]);

        std::string portStr = std::to_string(ipcPort);
        std::string hostExecutable = "./PluginHostProcess"; 
        
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(hostExecutable.c_str()));
        argv.push_back(const_cast<char*>(pluginPath.c_str()));
        argv.push_back(const_cast<char*>(portStr.c_str()));
        argv.push_back(nullptr);

        int status = posix_spawn(&pid_, hostExecutable.c_str(), &actions, nullptr, argv.data(), environ);
        
        posix_spawn_file_actions_destroy(&actions);
        close(pipeFds_[1]); // Close write end in parent
        pipeFds_[1] = -1;

        if (status != 0) {
            std::cerr << "Failed to spawn PluginHostProcess: " << status << std::endl;
            close(pipeFds_[0]);
            pipeFds_[0] = -1;
            pid_ = -1;
            return false;
        }

        return true;
    }

    std::string getOutput() override {
        if (pipeFds_[0] == -1) return "";
        
        char buffer[4096];
        std::string output;
        
        // Non-blocking read
        int flags = fcntl(pipeFds_[0], F_GETFL, 0);
        fcntl(pipeFds_[0], F_SETFL, flags | O_NONBLOCK);
        
        ssize_t bytesRead;
        while ((bytesRead = read(pipeFds_[0], buffer, sizeof(buffer))) > 0) {
            output.append(buffer, bytesRead);
        }
        
        return output;
    }

    void terminate() override {
        if (pid_ > 0) {
            kill(pid_, SIGTERM);
            
            // Wait for the process to exit to avoid zombies
            int status;
            for (int i = 0; i < 10; ++i) { // Try for 1 second
                if (waitpid(pid_, &status, WNOHANG) == pid_) {
                    pid_ = -1;
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            // Force kill if it didn't respond to SIGTERM
            kill(pid_, SIGKILL);
            waitpid(pid_, &status, 0);
            pid_ = -1;
        }
    }

    bool isAlive() const override {
        if (pid_ <= 0) return false;
        
        int status;
        pid_t result = waitpid(pid_, &status, WNOHANG);
        
        if (result == 0) {
            return true; // Still running
        } else if (result == pid_) {
            return false; // Finished
        }
        
        return false;
    }

    int getProcessId() const override { return static_cast<int>(pid_); }

private:
    pid_t pid_;
    int pipeFds_[2];

    void closePipes() {
        if (pipeFds_[0] != -1) { close(pipeFds_[0]); pipeFds_[0] = -1; }
        if (pipeFds_[1] != -1) { close(pipeFds_[1]); pipeFds_[1] = -1; }
    }
};

std::unique_ptr<PluginSandbox> PluginSandbox::create() {
    return std::make_unique<MacOSPluginSandbox>();
}

} // namespace plugins
} // namespace daiw

#elif defined(_WIN32)
// Windows implementation would go here using CreateProcess and Job Objects
namespace daiw {
namespace plugins {
std::unique_ptr<PluginSandbox> PluginSandbox::create() { return nullptr; }
}
}
#else
// Linux implementation would go here using fork/exec and seccomp
namespace daiw {
namespace plugins {
std::unique_ptr<PluginSandbox> PluginSandbox::create() { return nullptr; }
}
}
#endif

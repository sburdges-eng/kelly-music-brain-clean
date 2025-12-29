# CMake generated Testfile for 
# Source directory: /Volumes/Extreme SSD/kelly-project/plugin-juce
# Build directory: /Volumes/Extreme SSD/kelly-project/audio-engine-cpp/build-debug/plugin-juce
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[PluginTests]=] "/Volumes/Extreme SSD/kelly-project/audio-engine-cpp/build-debug/plugin-juce/PentaCorePlugin_Tests")
set_tests_properties([=[PluginTests]=] PROPERTIES  _BACKTRACE_TRIPLES "/Volumes/Extreme SSD/kelly-project/plugin-juce/CMakeLists.txt;111;add_test;/Volumes/Extreme SSD/kelly-project/plugin-juce/CMakeLists.txt;0;")
subdirs("../_deps/catch2-build")

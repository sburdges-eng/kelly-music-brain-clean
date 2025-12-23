/*
  ==============================================================================

   This file is part of the JUCE library.
   Copyright (c) 2022 - Raw Material Software Limited

   JUCE is an open source library subject to commercial or open-source
   licensing.

   The code included in this file is provided under the terms of the ISC license
   http://www.isc.org/downloads/software-support-policy/isc-license. Permission
   To use, copy, modify, and/or distribute this software for any purpose with or
   without fee is hereby granted provided that the above copyright notice and
   this permission notice appear in all copies.

   JUCE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY, AND ALL WARRANTIES, WHETHER
   EXPRESSED OR IMPLIED, INCLUDING MERCHANTABILITY AND FITNESS FOR PURPOSE, ARE
   DISCLAIMED.

  ==============================================================================
*/

#pragma once

//==============================================================================
// macOS SDK 26.2+ compatibility fix for __CLOCK_AVAILABILITY and __API_AVAILABLE
// The SDK headers use these macros in function declarations and enum definitions.
// For compatibility with older toolchains and to avoid complex availability checks,
// we define these macros to be empty, which allows the code to compile on all versions.
#if (defined(__APPLE__) || defined(__APPLE_CPP__) || defined(__APPLE_CC__))
 #ifndef TARGET_OS_MAC
  #include <TargetConditionals.h>
 #endif
 // Set deployment target macros if not already set (required for system headers)
 // MIN_REQUIRED sets the minimum macOS version we support (10.12 = Sierra)
 #ifndef MAC_OS_X_VERSION_MIN_REQUIRED
  #define MAC_OS_X_VERSION_MIN_REQUIRED MAC_OS_X_VERSION_10_12
 #endif
 // Define SDK compatibility macros to empty to avoid compilation issues
 // This allows modern SDK headers to compile with older deployment targets
 #ifndef __CLOCK_AVAILABILITY
  #define __CLOCK_AVAILABILITY
 #endif
 #ifndef __API_AVAILABLE
  #define __API_AVAILABLE(...)
 #endif
 #ifndef __API_DEPRECATED
  #define __API_DEPRECATED(...)
 #endif
 #ifndef __API_UNAVAILABLE
  #define __API_UNAVAILABLE(...)
 #endif
 #ifndef __OSX_AVAILABLE
  #define __OSX_AVAILABLE(...)
 #endif
 #ifndef __OSX_AVAILABLE_STARTING
  #define __OSX_AVAILABLE_STARTING(...)
 #endif
 #ifndef __OSX_AVAILABLE_BUT_DEPRECATED
  #define __OSX_AVAILABLE_BUT_DEPRECATED(...)
 #endif
 #ifndef __OSX_AVAILABLE_BUT_DEPRECATED_MSG
  #define __OSX_AVAILABLE_BUT_DEPRECATED_MSG(...)
 #endif
 #ifndef __WATCHOS_PROHIBITED
  #define __WATCHOS_PROHIBITED
 #endif
 #ifndef __TVOS_PROHIBITED
  #define __TVOS_PROHIBITED
 #endif
 #ifndef __DARWIN_1050
  #define __DARWIN_1050(...)
 #endif
 #ifndef __IOS_AVAILABLE
  #define __IOS_AVAILABLE(...)
 #endif
 #ifndef __TVOS_AVAILABLE
  #define __TVOS_AVAILABLE(...)
 #endif
 #ifndef __WATCHOS_AVAILABLE
  #define __WATCHOS_AVAILABLE(...)
 #endif
#endif

//==============================================================================
/** Current JUCE version number.

    See also SystemStats::getJUCEVersion() for a string version.
*/
#define JUCE_MAJOR_VERSION      7
#define JUCE_MINOR_VERSION      0
#define JUCE_BUILDNUMBER        12

/** Current JUCE version number.

    Bits 16 to 32 = major version.
    Bits 8 to 16 = minor version.
    Bits 0 to 8 = point release.

    See also SystemStats::getJUCEVersion() for a string version.
*/
#define JUCE_VERSION   ((JUCE_MAJOR_VERSION << 16) + (JUCE_MINOR_VERSION << 8) + JUCE_BUILDNUMBER)

#if ! DOXYGEN
#define JUCE_VERSION_ID \
    [[maybe_unused]] volatile auto juceVersionId = "juce_version_" JUCE_STRINGIFY(JUCE_MAJOR_VERSION) "_" JUCE_STRINGIFY(JUCE_MINOR_VERSION) "_" JUCE_STRINGIFY(JUCE_BUILDNUMBER);
#endif

//==============================================================================
#include "juce_CompilerSupport.h"
#include "juce_CompilerWarnings.h"
#include "juce_PlatformDefs.h"

//==============================================================================
// Essential C++ Standard Library headers required by JUCE core
#include <type_traits>
#include <utility>
#include <memory>
#include <cstddef>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <list>
#include <typeindex>
#include <functional>
#include <array>
#include <unordered_map>
#include <optional>
#include <atomic>
#include <numeric>
#include <future>

//==============================================================================
// Now we'll include some common OS headers..
JUCE_BEGIN_IGNORE_WARNINGS_MSVC (4514 4245 4100)

#if JUCE_MSVC
 #include <intrin.h>
#endif


#if JUCE_MAC || JUCE_IOS
 #include <libkern/OSAtomic.h>
 #include <libkern/OSByteOrder.h>
 #include <xlocale.h>
 #include <signal.h>
#endif

#if JUCE_LINUX || JUCE_BSD
 #include <cstring>
 #include <signal.h>

 #if __INTEL_COMPILER
  #if __ia64__
   #include <ia64intrin.h>
  #else
   #include <ia32intrin.h>
  #endif
 #endif
#endif

#if JUCE_MSVC && JUCE_DEBUG
 #include <crtdbg.h>
#endif

JUCE_END_IGNORE_WARNINGS_MSVC

#if JUCE_MINGW
 #include <cstring>
 #include <sys/types.h>
#endif

#if JUCE_ANDROID
 #include <cstring>
 #include <byteswap.h>
#endif

// undef symbols that are sometimes set by misguided 3rd-party headers..
#undef TYPE_BOOL
#undef max
#undef min
#undef major
#undef minor
#undef KeyPress

//==============================================================================
// DLL building settings on Windows
#if JUCE_MSVC
 #ifdef JUCE_DLL_BUILD
  #define JUCE_API __declspec (dllexport)
  #pragma warning (disable: 4251)
 #elif defined (JUCE_DLL)
  #define JUCE_API __declspec (dllimport)
  #pragma warning (disable: 4251)
 #endif
 #ifdef __INTEL_COMPILER
  #pragma warning (disable: 1125) // (virtual override warning)
 #endif
#elif defined (JUCE_DLL) || defined (JUCE_DLL_BUILD)
 #define JUCE_API __attribute__ ((visibility ("default")))
#endif

//==============================================================================
#ifndef JUCE_API
 #define JUCE_API   /**< This macro is added to all JUCE public class declarations. */
#endif

#if JUCE_MSVC && JUCE_DLL_BUILD
 #define JUCE_PUBLIC_IN_DLL_BUILD(declaration)  public: declaration; private:
#else
 #define JUCE_PUBLIC_IN_DLL_BUILD(declaration)  declaration;
#endif

/** This macro is added to all JUCE public function declarations. */
#define JUCE_PUBLIC_FUNCTION        JUCE_API JUCE_CALLTYPE

#ifndef DOXYGEN
 #define JUCE_NAMESPACE juce  // This old macro is deprecated: you should just use the juce namespace directly.
#endif

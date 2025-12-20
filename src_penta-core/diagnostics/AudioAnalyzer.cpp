#include "penta/diagnostics/AudioAnalyzer.h"
#include <cmath>
#include <algorithm>

// SIMD includes for optimized RMS calculation
#ifdef PENTA_ENABLE_SIMD
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <x86intrin.h>
    #endif
#endif

namespace penta::diagnostics {

AudioAnalyzer::AudioAnalyzer()
    : rmsLevel_(0.0f)
    , peakLevel_(0.0f)
    , clipping_(false)
    , minLevel_(1.0f)
    , maxLevel_(0.0f)
    , clippingThreshold_(0.99f)
    , decayRate_(0.995f)  // Peak hold decay per sample
{
}

void AudioAnalyzer::analyze(const float* buffer, size_t frames, size_t channels) noexcept {
    if (!buffer || frames == 0 || channels == 0) {
        return;
    }
    
    const size_t totalSamples = frames * channels;
    
    // Calculate RMS and peak levels
    float sumSquares = 0.0f;
    float localPeak = 0.0f;
    bool localClipping = false;
    
#ifdef PENTA_ENABLE_SIMD
    // SIMD-optimized version using AVX2
    // Process 8 floats at a time
    const size_t simdWidth = 8;
    const size_t simdSamples = (totalSamples / simdWidth) * simdWidth;
    
    __m256 vSumSquares = _mm256_setzero_ps();
    __m256 vPeak = _mm256_setzero_ps();
    __m256 vThreshold = _mm256_set1_ps(clippingThreshold_);
    
    for (size_t i = 0; i < simdSamples; i += simdWidth) {
        __m256 vSamples = _mm256_loadu_ps(&buffer[i]);
        
        // Absolute values for peak detection
        __m256 vAbs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vSamples);
        
        // Accumulate squares for RMS
        vSumSquares = _mm256_fmadd_ps(vSamples, vSamples, vSumSquares);
        
        // Track maximum for peak
        vPeak = _mm256_max_ps(vPeak, vAbs);
        
        // Check for clipping
        __m256 vClipMask = _mm256_cmp_ps(vAbs, vThreshold, _CMP_GE_OQ);
        if (_mm256_movemask_ps(vClipMask) != 0) {
            localClipping = true;
        }
    }
    
    // Horizontal reduction for sum of squares
    __m128 vLow = _mm256_castps256_ps128(vSumSquares);
    __m128 vHigh = _mm256_extractf128_ps(vSumSquares, 1);
    __m128 vSum128 = _mm_add_ps(vLow, vHigh);
    vSum128 = _mm_hadd_ps(vSum128, vSum128);
    vSum128 = _mm_hadd_ps(vSum128, vSum128);
    sumSquares = _mm_cvtss_f32(vSum128);
    
    // Horizontal reduction for peak
    vLow = _mm256_castps256_ps128(vPeak);
    vHigh = _mm256_extractf128_ps(vPeak, 1);
    __m128 vMax128 = _mm_max_ps(vLow, vHigh);
    vMax128 = _mm_max_ps(vMax128, _mm_shuffle_ps(vMax128, vMax128, _MM_SHUFFLE(2, 3, 0, 1)));
    vMax128 = _mm_max_ps(vMax128, _mm_shuffle_ps(vMax128, vMax128, _MM_SHUFFLE(1, 0, 3, 2)));
    localPeak = _mm_cvtss_f32(vMax128);
    
    // Process remaining samples with scalar code
    for (size_t i = simdSamples; i < totalSamples; ++i) {
        float sample = buffer[i];
        float absSample = std::abs(sample);
        
        sumSquares += sample * sample;
        localPeak = std::max(localPeak, absSample);
        
        if (absSample >= clippingThreshold_) {
            localClipping = true;
        }
    }
#else
    // Scalar fallback version
    for (size_t i = 0; i < totalSamples; ++i) {
        float sample = buffer[i];
        float absSample = std::abs(sample);
        
        sumSquares += sample * sample;
        localPeak = std::max(localPeak, absSample);
        
        if (absSample >= clippingThreshold_) {
            localClipping = true;
        }
    }
#endif
    
    // Calculate RMS
    float rms = std::sqrt(sumSquares / static_cast<float>(totalSamples));
    
    // Apply peak decay
    float currentPeak = peakLevel_.load(std::memory_order_relaxed);
    float newPeak = std::max(localPeak, currentPeak * decayRate_);
    
    // Update atomic values (RT-safe)
    rmsLevel_.store(rms, std::memory_order_relaxed);
    peakLevel_.store(newPeak, std::memory_order_relaxed);
    
    if (localClipping) {
        clipping_.store(true, std::memory_order_relaxed);
    }
    
    // Update min/max for dynamic range calculation
    float currentMin = minLevel_.load(std::memory_order_relaxed);
    float currentMax = maxLevel_.load(std::memory_order_relaxed);
    
    if (rms > kEpsilon && rms < currentMin) {
        minLevel_.store(rms, std::memory_order_relaxed);
    }
    if (rms > currentMax) {
        maxLevel_.store(rms, std::memory_order_relaxed);
    }
}

float AudioAnalyzer::getRmsLevel() const noexcept {
    return rmsLevel_.load(std::memory_order_relaxed);
}

float AudioAnalyzer::getPeakLevel() const noexcept {
    return peakLevel_.load(std::memory_order_relaxed);
}

float AudioAnalyzer::getDynamicRange() const noexcept {
    float min = minLevel_.load(std::memory_order_relaxed);
    float max = maxLevel_.load(std::memory_order_relaxed);
    
    if (min <= kEpsilon || max <= kEpsilon) {
        return 0.0f;
    }
    
    // Convert to dB: 20 * log10(max / min)
    return 20.0f * std::log10(max / min);
}

void AudioAnalyzer::setClippingThreshold(float threshold) noexcept {
    clippingThreshold_ = std::clamp(threshold, 0.0f, 1.0f);
}

void AudioAnalyzer::setDecayRate(float rate) noexcept {
    decayRate_ = std::clamp(rate, 0.0f, 1.0f);
}

void AudioAnalyzer::reset() {
    rmsLevel_.store(0.0f, std::memory_order_relaxed);
    peakLevel_.store(0.0f, std::memory_order_relaxed);
    clipping_.store(false, std::memory_order_relaxed);
    minLevel_.store(1.0f, std::memory_order_relaxed);
    maxLevel_.store(0.0f, std::memory_order_relaxed);
}

} // namespace penta::diagnostics

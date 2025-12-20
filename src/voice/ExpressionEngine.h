#pragma once

#include "voice/LyricTypes.h"
#include "common/Types.h"
#include <vector>

namespace kelly {

/**
 * ExpressionEngine - Applies vocal expression and dynamics to synthesis.
 *
 * This class handles:
 * - Expression curve application to vocal notes
 * - Emotion-based expression mapping
 * - Real-time expression modulation
 */
class ExpressionEngine {
public:
    ExpressionEngine();
    ~ExpressionEngine() = default;

    /**
     * Apply expression to vocal characteristics based on emotion.
     * @param baseCharacteristics Base vocal characteristics
     * @param emotion Emotion node
     * @return Modified vocal characteristics with expression applied
     */
    VocalExpression applyEmotionExpression(
        const VocalExpression& baseExpression,
        const EmotionNode& emotion
    );

    /**
     * Generate expression curve over time.
     * @param duration Duration in beats
     * @param expression Expression parameters
     * @param numPoints Number of points in curve
     * @return Vector of expression values over time (0.0 to 1.0)
     */
    std::vector<float> generateExpressionCurve(
        double duration,
        const VocalExpression& expression,
        int numPoints = 100
    ) const;

    /**
     * Generate dynamics curve (crescendo/diminuendo).
     * @param duration Duration in beats
     * @param crescendoAmount Crescendo amount (0.0 to 1.0)
     * @param diminuendoAmount Diminuendo amount (0.0 to 1.0)
     * @param numPoints Number of points
     * @return Vector of dynamics values
     */
    std::vector<float> generateDynamicsCurve(
        double duration,
        float crescendoAmount,
        float diminuendoAmount,
        int numPoints = 100
    ) const;

    /**
     * Generate vibrato depth curve.
     * @param duration Duration in beats
     * @param baseDepth Base vibrato depth
     * @param variationAmount Amount of variation
     * @param numPoints Number of points
     * @return Vector of vibrato depth values
     */
    std::vector<float> generateVibratoCurve(
        double duration,
        float baseDepth,
        float variationAmount,
        int numPoints = 100
    ) const;

    /**
     * Apply expression to a single sample in real-time.
     * @param expression Current expression
     * @param position Position in note (0.0 = start, 1.0 = end)
     * @return Expression values at this position
     */
    VocalExpression getExpressionAtPosition(
        const VocalExpression& expression,
        float position
    ) const;

private:
    /**
     * Map emotion VAD values to expression parameters.
     */
    void mapEmotionToExpression(const EmotionNode& emotion, VocalExpression& expression) const;

    /**
     * Apply crescendo/diminuendo to dynamics.
     */
    float applyDynamicsCurve(float position, float crescendo, float diminuendo) const;
};

} // namespace kelly

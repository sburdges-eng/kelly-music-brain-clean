#include "voice/ExpressionEngine.h"
#include "common/Types.h"
#include <algorithm>
#include <cmath>

namespace kelly {

ExpressionEngine::ExpressionEngine() {}

VocalExpression
ExpressionEngine::applyEmotionExpression(const VocalExpression &baseExpression,
                                         const EmotionNode &emotion) {
  VocalExpression expression = baseExpression;

  // Map emotion to expression
  mapEmotionToExpression(emotion, expression);

  // Valence affects brightness
  // Positive emotions = brighter, negative = darker
  expression.brightness = baseExpression.brightness + (emotion.valence * 0.3f);
  expression.brightness = std::clamp(expression.brightness, 0.0f, 1.0f);

  // Arousal affects vibrato rate
  // High arousal = faster vibrato
  expression.vibratoRate =
      baseExpression.vibratoRate + (emotion.arousal * 2.0f);
  expression.vibratoRate = std::clamp(expression.vibratoRate, 2.0f, 8.0f);

  // Intensity affects vibrato depth
  expression.vibratoDepth =
      baseExpression.vibratoDepth + (emotion.intensity * 0.2f);
  expression.vibratoDepth = std::clamp(expression.vibratoDepth, 0.0f, 1.0f);

  // Dominance affects dynamics
  // High dominance = louder dynamics
  expression.dynamics = baseExpression.dynamics + (emotion.dominance * 0.3f);
  expression.dynamics = std::clamp(expression.dynamics, 0.0f, 1.0f);

  // Arousal affects articulation
  // High arousal = more staccato, low arousal = more legato
  expression.articulation =
      baseExpression.articulation + ((1.0f - emotion.arousal) * 0.4f);
  expression.articulation = std::clamp(expression.articulation, 0.0f, 1.0f);

  return expression;
}

void ExpressionEngine::mapEmotionToExpression(
    const EmotionNode &emotion, VocalExpression &expression) const {
  // Map emotion category to specific expression characteristics
  switch (emotion.categoryEnum) {
  case EmotionCategory::Joy:
    expression.brightness = 0.8f;
    expression.dynamics = 0.7f;
    expression.vibratoRate = 5.5f;
    expression.articulation = 0.3f; // More legato
    break;

  case EmotionCategory::Sadness:
    expression.brightness = 0.3f;
    expression.dynamics = 0.4f;
    expression.vibratoRate = 4.0f;
    expression.breathiness = 0.4f;  // More breathy
    expression.articulation = 0.2f; // Very legato
    break;

  case EmotionCategory::Anger:
    expression.brightness = 0.6f;
    expression.dynamics = 0.9f;
    expression.vibratoRate = 6.0f;
    expression.vibratoDepth = 0.5f;
    expression.articulation = 0.7f; // More staccato
    break;

  case EmotionCategory::Fear:
    expression.brightness = 0.4f;
    expression.dynamics = 0.5f;
    expression.vibratoRate = 5.0f;
    expression.breathiness = 0.5f;
    expression.articulation = 0.6f; // Somewhat staccato
    break;

  case EmotionCategory::Surprise:
    expression.brightness = 0.7f;
    expression.dynamics = 0.8f;
    expression.vibratoRate = 5.5f;
    expression.articulation = 0.8f; // Very staccato
    break;

  case EmotionCategory::Trust:
    expression.brightness = 0.6f;
    expression.dynamics = 0.6f;
    expression.vibratoRate = 4.5f;
    expression.articulation = 0.3f; // Legato
    break;

  case EmotionCategory::Anticipation:
    expression.brightness = 0.65f;
    expression.dynamics = 0.7f;
    expression.vibratoRate = 5.0f;
    expression.articulation = 0.4f;
    break;

  default:
    // Use default values
    break;
  }
}

std::vector<float> ExpressionEngine::generateExpressionCurve(
    double duration, const VocalExpression &expression, int numPoints) const {
  std::vector<float> curve;
  curve.reserve(numPoints);

  for (int i = 0; i < numPoints; ++i) {
    float position = static_cast<float>(i) / static_cast<float>(numPoints - 1);
    VocalExpression posExpression =
        getExpressionAtPosition(expression, position);

    // Combine expression parameters into a single value (simplified)
    float value = (posExpression.dynamics + posExpression.brightness) * 0.5f;
    curve.push_back(value);
  }

  return curve;
}

std::vector<float>
ExpressionEngine::generateDynamicsCurve(double duration, float crescendoAmount,
                                        float diminuendoAmount,
                                        int numPoints) const {
  std::vector<float> curve;
  curve.reserve(numPoints);

  for (int i = 0; i < numPoints; ++i) {
    float position = static_cast<float>(i) / static_cast<float>(numPoints - 1);
    float value =
        applyDynamicsCurve(position, crescendoAmount, diminuendoAmount);
    curve.push_back(value);
  }

  return curve;
}

std::vector<float> ExpressionEngine::generateVibratoCurve(double duration,
                                                          float baseDepth,
                                                          float variationAmount,
                                                          int numPoints) const {
  std::vector<float> curve;
  curve.reserve(numPoints);

  for (int i = 0; i < numPoints; ++i) {
    float position = static_cast<float>(i) / static_cast<float>(numPoints - 1);

    // Add sinusoidal variation
    float variation =
        std::sin(position * 2.0f * static_cast<float>(M_PI) * 3.0f) *
        variationAmount;
    float value = baseDepth + variation;
    value = std::clamp(value, 0.0f, 1.0f);

    curve.push_back(value);
  }

  return curve;
}

VocalExpression
ExpressionEngine::getExpressionAtPosition(const VocalExpression &expression,
                                          float position) const {
  VocalExpression posExpression = expression;

  // Apply dynamics curve (crescendo/diminuendo)
  float dynamicsMultiplier =
      applyDynamicsCurve(position, expression.crescendo, expression.diminuendo);
  posExpression.dynamics *= dynamicsMultiplier;
  posExpression.dynamics = std::clamp(posExpression.dynamics, 0.0f, 1.0f);

  // Apply vibrato depth variation (slight variation over time)
  float vibratoVariation =
      std::sin(position * 2.0f * static_cast<float>(M_PI) * 2.0f) * 0.1f;
  posExpression.vibratoDepth += vibratoVariation;
  posExpression.vibratoDepth =
      std::clamp(posExpression.vibratoDepth, 0.0f, 1.0f);

  return posExpression;
}

float ExpressionEngine::applyDynamicsCurve(float position, float crescendo,
                                           float diminuendo) const {
  float multiplier = 1.0f;

  // Apply crescendo (increase at start)
  if (crescendo > 0.0f && position < 0.5f) {
    float crescendoCurve = position / 0.5f; // 0 to 1 over first half
    multiplier += crescendo * crescendoCurve;
  }

  // Apply diminuendo (decrease at end)
  if (diminuendo > 0.0f && position > 0.5f) {
    float diminuendoCurve = (position - 0.5f) / 0.5f; // 0 to 1 over second half
    multiplier -= diminuendo * diminuendoCurve;
  }

  return std::clamp(multiplier, 0.0f, 2.0f);
}

} // namespace kelly

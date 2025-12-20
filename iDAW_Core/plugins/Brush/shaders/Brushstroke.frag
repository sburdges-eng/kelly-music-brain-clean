/**
 * Brushstroke.frag - Paint brush visualization
 * Creates a paint stroke effect that responds to filter modulation
 */

#version 330 core

in vec2 fragTexCoord;
out vec4 fragColor;

uniform float strokePosition;    // Position on spectrum (0-1)
uniform float strokeWidth;       // Filter bandwidth
uniform float strokeIntensity;   // Resonance glow
uniform float strokeAngle;       // LFO phase (radians)
uniform float wetness;           // Mix/blend amount
uniform float bristleSpread;     // Resonance spread
uniform vec2 trailPos[2];        // Trailing positions
uniform float time;              // Animation time

// Noise for texture
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Brush bristle pattern
float bristlePattern(vec2 uv, float spread) {
    float pattern = 0.0;

    // Multiple bristle lines
    for (int i = 0; i < 8; i++) {
        float offset = (float(i) - 3.5) * 0.02 * (1.0 + spread);
        float bristle = smoothstep(0.02, 0.0, abs(uv.y + offset));
        bristle *= noise(vec2(uv.x * 50.0 + float(i) * 10.0, float(i)));
        pattern += bristle;
    }

    return pattern * 0.3;
}

// Paint drip effect
float paintDrip(vec2 uv, float wetness) {
    float drip = 0.0;

    for (int i = 0; i < 5; i++) {
        float fi = float(i);
        vec2 dripStart = vec2(hash(vec2(fi, 0.0)), 0.5);
        float dripLength = wetness * hash(vec2(fi, 1.0)) * 0.3;
        float dripWidth = 0.02 * (1.0 - uv.y * 0.5);

        vec2 dripPos = dripStart + vec2(0.0, -dripLength * (uv.y - 0.5));

        float d = length(vec2(uv.x - dripPos.x, (uv.y - dripStart.y) * 3.0));
        drip += smoothstep(dripWidth, 0.0, d) * step(dripStart.y - dripLength, uv.y) * step(uv.y, dripStart.y);
    }

    return drip * wetness;
}

// Main brush stroke shape
float brushShape(vec2 uv, float position, float width, float angle) {
    // Rotate UV by angle
    float c = cos(angle * 0.1);
    float s = sin(angle * 0.1);
    vec2 ruv = vec2(
        (uv.x - 0.5) * c - (uv.y - 0.5) * s + 0.5,
        (uv.x - 0.5) * s + (uv.y - 0.5) * c + 0.5
    );

    // Horizontal stroke
    float yPos = position;
    float dist = abs(ruv.y - yPos);

    // Feathered edges
    float stroke = smoothstep(width * 0.5, width * 0.2, dist);

    // Add texture variation
    stroke *= 0.8 + 0.2 * noise(ruv * 30.0);

    return stroke;
}

void main() {
    vec2 uv = fragTexCoord;

    // Canvas background (off-white paper)
    vec3 canvasColor = vec3(0.95, 0.93, 0.9);

    // Add paper texture
    float paperTex = noise(uv * 100.0) * 0.03;
    canvasColor -= paperTex;

    // Main brush stroke
    float mainStroke = brushShape(uv, strokePosition, strokeWidth, strokeAngle);

    // Trail strokes (history)
    float trail1 = brushShape(uv, trailPos[0].x, strokeWidth * 0.7, strokeAngle - 0.2) * 0.5;
    float trail2 = brushShape(uv, trailPos[1].x, strokeWidth * 0.5, strokeAngle - 0.4) * 0.25;

    // Combine strokes
    float totalStroke = mainStroke + trail1 + trail2;
    totalStroke = clamp(totalStroke, 0.0, 1.0);

    // Paint color (vibrant blue with resonance-based hue shift)
    vec3 paintColor = vec3(0.2, 0.4, 0.8);

    // Shift hue based on intensity (resonance)
    paintColor.r += strokeIntensity * 0.3;
    paintColor.g -= strokeIntensity * 0.1;

    // Add bristle texture
    float bristles = bristlePattern(uv - vec2(0.0, strokePosition), bristleSpread);
    paintColor += vec3(0.1, 0.05, 0.0) * bristles;

    // Add drips for high wetness
    float drips = paintDrip(uv, wetness);
    totalStroke += drips;

    // Resonance glow
    float glow = mainStroke * strokeIntensity * 0.5;
    vec3 glowColor = vec3(0.8, 0.6, 1.0) * glow;

    // Blend paint onto canvas
    vec3 finalColor = mix(canvasColor, paintColor, totalStroke);
    finalColor += glowColor;

    // Edge darkening (paint pooling)
    float edge = smoothstep(strokeWidth * 0.3, strokeWidth * 0.4, abs(uv.y - strokePosition));
    edge = 1.0 - edge * mainStroke * 0.3;
    finalColor *= edge;

    // Vignette
    float vignette = 1.0 - length(uv - 0.5) * 0.4;
    finalColor *= vignette;

    fragColor = vec4(finalColor, 1.0);
}

/**
 * Cutout.frag - Stencil pattern visualization
 * Creates a paper cutout effect that responds to ducking
 */

#version 330 core

in vec2 fragTexCoord;
out vec4 fragColor;

uniform float cutoutDepth;      // 0-1, how deep the cutout is
uniform float cutoutProgress;   // Animation progress
uniform float patternRotation;  // Pattern rotation angle
uniform float edgeSharpness;    // Sharp vs feathered edges
uniform float inputLevel;       // Sidechain input level
uniform float outputLevel;      // Main output level
uniform float time;             // Animation time

// Paper texture noise
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

// Create stencil pattern shape
float stencilPattern(vec2 uv, float rotation) {
    // Rotate UV
    float c = cos(rotation);
    float s = sin(rotation);
    vec2 ruv = vec2(
        uv.x * c - uv.y * s,
        uv.x * s + uv.y * c
    );

    // Create a geometric cutout pattern (star shape)
    float angle = atan(ruv.y, ruv.x);
    float radius = length(ruv);

    // Star pattern
    float star = 0.3 + 0.15 * sin(angle * 5.0);

    // Sharp or soft edges
    float edge = smoothstep(star - 0.1 * (1.0 - edgeSharpness),
                            star + 0.1 * (1.0 - edgeSharpness),
                            radius);

    return edge;
}

void main() {
    vec2 uv = fragTexCoord * 2.0 - 1.0;

    // Paper base color (cream/off-white)
    vec3 paperColor = vec3(0.95, 0.92, 0.88);

    // Add paper texture
    float paperTexture = noise(fragTexCoord * 50.0) * 0.05;
    paperColor -= paperTexture;

    // Calculate stencil cutout
    float pattern = stencilPattern(uv, patternRotation + time * 0.1);

    // Cutout shadow (3D depth effect)
    vec2 shadowOffset = vec2(0.02, -0.02) * cutoutDepth;
    float shadowPattern = stencilPattern(uv + shadowOffset, patternRotation + time * 0.1);

    // Cutout color (dark opening)
    vec3 cutoutColor = vec3(0.1, 0.1, 0.15);

    // Blend based on depth
    float depth = pattern * cutoutDepth;

    // Shadow layer
    float shadow = (1.0 - shadowPattern) * cutoutDepth * 0.5;
    paperColor -= shadow;

    // Main cutout
    vec3 finalColor = mix(paperColor, cutoutColor, depth);

    // Add edge highlight
    float edgeGlow = abs(pattern - 0.5) * 2.0;
    edgeGlow = pow(edgeGlow, 4.0) * inputLevel;
    finalColor += edgeGlow * vec3(0.3, 0.2, 0.1);

    // Output level indicator (subtle glow)
    float centerDist = length(uv);
    finalColor += vec3(0.1, 0.2, 0.3) * outputLevel * (1.0 - centerDist) * 0.2;

    fragColor = vec4(finalColor, 1.0);
}

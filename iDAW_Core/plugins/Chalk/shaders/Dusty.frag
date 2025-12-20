/**
 * Dusty.frag - Chalkboard texture visualization
 * Creates a dusty chalkboard effect that responds to lo-fi degradation
 */

#version 330 core

in vec2 fragTexCoord;
out vec4 fragColor;

uniform float chalkDensity;     // Amount of chalk marks (bit reduction)
uniform float smearAmount;      // Smearing level (SR reduction)
uniform float particleCount;    // Dust particles floating
uniform float eraserProgress;   // Erasing animation
uniform float noiseFloor;       // Visible noise
uniform float time;             // Animation time

// Hash function for noise
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float hash3(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}

// Value noise
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

// Chalk stroke pattern
float chalkStroke(vec2 uv, float density) {
    float stroke = 0.0;

    // Multiple octaves of noise for texture
    stroke += noise(uv * 20.0) * 0.5;
    stroke += noise(uv * 40.0) * 0.25;
    stroke += noise(uv * 80.0) * 0.125;

    // Threshold based on density
    stroke = smoothstep(0.5 - density * 0.4, 0.5 + density * 0.1, stroke);

    return stroke;
}

// Floating dust particles
float dustParticles(vec2 uv, float t, float count) {
    float dust = 0.0;

    for (int i = 0; i < 20; i++) {
        float fi = float(i);
        vec2 pos = vec2(
            hash(vec2(fi, 0.0)) + sin(t * 0.3 + fi) * 0.1,
            fract(hash(vec2(fi, 1.0)) + t * 0.05 * (0.5 + hash(vec2(fi, 2.0))))
        );

        float d = length(uv - pos);
        float size = 0.003 * (0.5 + hash(vec2(fi, 3.0)));
        dust += smoothstep(size, 0.0, d) * count * 0.02;
    }

    return dust;
}

// Smear/blur effect
vec3 smearEffect(vec2 uv, float amount) {
    vec2 offset = vec2(noise(uv * 5.0 + time * 0.1) - 0.5,
                       noise(uv * 5.0 + 100.0) - 0.5) * amount * 0.05;
    return vec3(offset, 0.0);
}

void main() {
    vec2 uv = fragTexCoord;

    // Chalkboard base color (dark green-gray)
    vec3 boardColor = vec3(0.15, 0.2, 0.18);

    // Add subtle board texture
    float boardTexture = noise(uv * 100.0) * 0.02;
    boardColor += boardTexture;

    // Chalk marks
    float chalk = chalkStroke(uv, chalkDensity);
    vec3 chalkColor = vec3(0.9, 0.9, 0.85);  // Off-white chalk

    // Apply smear effect to chalk
    vec2 smearOffset = smearEffect(uv, smearAmount).xy;
    float smearedChalk = chalkStroke(uv + smearOffset, chalkDensity);
    chalk = mix(chalk, smearedChalk, smearAmount);

    // Blend chalk onto board
    vec3 finalColor = mix(boardColor, chalkColor, chalk * 0.6);

    // Add eraser marks (darker patches where chalk was removed)
    float erased = noise(uv * 8.0 + vec2(eraserProgress * 2.0, 0.0));
    erased = smoothstep(0.4, 0.6, erased) * eraserProgress;
    finalColor = mix(finalColor, boardColor * 0.9, erased * 0.3);

    // Add floating dust particles
    float dust = dustParticles(uv, time, particleCount);
    finalColor += vec3(0.8, 0.8, 0.75) * dust;

    // Add noise floor visualization (static-like effect)
    float staticNoise = hash(uv * 1000.0 + time * 100.0) * noiseFloor * 0.2;
    finalColor += staticNoise;

    // Vignette
    float vignette = 1.0 - length(uv - 0.5) * 0.8;
    finalColor *= vignette;

    fragColor = vec4(finalColor, 1.0);
}

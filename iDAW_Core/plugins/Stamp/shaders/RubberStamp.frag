/**
 * RubberStamp.frag - Rubber stamp visualization
 * Creates a stamping effect that responds to stutter/repeat
 */

#version 330 core

in vec2 fragTexCoord;
out vec4 fragColor;

uniform float stampProgress;     // Current stamp animation (0-1)
uniform float inkIntensity;      // Ink amount (decay)
uniform float patternRotation;   // Pattern rotation
uniform int stampCount;          // Number of stamps
uniform float pressureLevel;     // Stamp pressure
uniform bool isStamping;         // Currently active
uniform float time;              // Animation time

// Hash for procedural patterns
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(269.5, 183.3))) * 43758.5453);
}

// Noise for texture
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

// Stamp pattern shape (circular with design)
float stampShape(vec2 uv, float rotation) {
    // Rotate
    float c = cos(rotation);
    float s = sin(rotation);
    vec2 ruv = vec2(
        uv.x * c - uv.y * s,
        uv.x * s + uv.y * c
    );

    // Outer circle
    float radius = length(ruv);
    float circle = smoothstep(0.4, 0.38, radius);

    // Inner circle (border)
    float innerCircle = smoothstep(0.35, 0.33, radius) - smoothstep(0.33, 0.31, radius);
    circle -= innerCircle * 0.5;

    // Add a star/logo pattern in center
    float angle = atan(ruv.y, ruv.x);
    float star = 0.15 + 0.05 * sin(angle * 6.0);
    float starShape = smoothstep(star + 0.02, star, radius);

    circle = max(circle, starShape);

    // Add text-like pattern
    float textRing = step(0.22, radius) * step(radius, 0.28);
    float textPattern = step(0.5, sin(angle * 20.0 + noise(ruv * 10.0) * 2.0));
    circle = max(circle, textRing * textPattern * 0.7);

    return circle;
}

// Ink texture
float inkTexture(vec2 uv, float intensity) {
    float tex = noise(uv * 100.0);
    tex += noise(uv * 200.0) * 0.5;
    tex = tex * 0.5 + 0.5;

    // More variation at low intensity (fading ink)
    tex *= 0.8 + 0.2 * intensity;

    // Random gaps (running out of ink)
    if (intensity < 0.5) {
        float gaps = noise(uv * 50.0);
        if (gaps > intensity + 0.3) {
            tex *= 0.3;
        }
    }

    return tex;
}

// Single stamp impression
vec4 drawStamp(vec2 uv, vec2 center, float rotation, float intensity, float size) {
    vec2 stampUV = (uv - center) / size;
    float shape = stampShape(stampUV, rotation);

    if (shape < 0.01)
        return vec4(0.0);

    // Ink color (deep red/burgundy stamp ink)
    vec3 inkColor = vec3(0.6, 0.1, 0.15);

    // Apply ink texture
    float tex = inkTexture(stampUV, intensity);
    inkColor *= tex;

    // Apply intensity (fading)
    float alpha = shape * intensity * tex;

    return vec4(inkColor, alpha);
}

void main() {
    vec2 uv = fragTexCoord;

    // Paper background
    vec3 paperColor = vec3(0.96, 0.94, 0.9);
    float paperTex = noise(uv * 80.0) * 0.02;
    paperColor -= paperTex;

    // Add paper fibers
    float fibers = noise(uv * vec2(200.0, 20.0)) * 0.01;
    paperColor -= fibers;

    vec4 finalColor = vec4(paperColor, 1.0);

    // Draw multiple stamp impressions based on count
    int maxStamps = min(stampCount + 1, 8);

    for (int i = 0; i < maxStamps; i++) {
        float fi = float(i);

        // Calculate stamp position (slightly offset each time)
        vec2 center = vec2(0.5, 0.5);
        center.x += (hash(vec2(fi, 0.0)) - 0.5) * 0.1;
        center.y += (hash(vec2(fi, 1.0)) - 0.5) * 0.1;

        // Rotation varies slightly
        float rotation = patternRotation + (hash(vec2(fi, 2.0)) - 0.5) * 0.2;

        // Intensity decreases with each stamp
        float intensity = inkIntensity * pow(0.9, fi);

        // Size slightly varies
        float size = 0.8 + hash(vec2(fi, 3.0)) * 0.1;

        // Current stamp (last one) animates
        if (i == maxStamps - 1 && isStamping) {
            // Stamp press animation
            float press = stampProgress;
            size *= 0.9 + press * 0.1;  // Slight size change on press
            intensity *= press;  // Fade in
        }

        vec4 stamp = drawStamp(uv, center, rotation, intensity, size);

        // Blend stamp onto paper
        finalColor.rgb = mix(finalColor.rgb, stamp.rgb, stamp.a * 0.9);
    }

    // Add slight shadow under stamps
    float shadow = 0.0;
    for (int i = 0; i < min(stampCount + 1, 4); i++) {
        float fi = float(i);
        vec2 center = vec2(0.5, 0.5);
        center.x += (hash(vec2(fi, 0.0)) - 0.5) * 0.1;
        center.y += (hash(vec2(fi, 1.0)) - 0.5) * 0.1;

        vec2 shadowOffset = vec2(0.01, -0.01);
        float dist = length(uv - center - shadowOffset);
        shadow += smoothstep(0.35, 0.3, dist) * 0.1;
    }
    finalColor.rgb -= shadow;

    // Pressure indicator (subtle glow around edge when active)
    if (isStamping) {
        float edge = 1.0 - length(uv - 0.5) * 1.5;
        finalColor.rgb += vec3(0.1, 0.0, 0.0) * edge * pressureLevel * 0.3;
    }

    fragColor = finalColor;
}

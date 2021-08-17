// Chromatic aberration.
#version 150

uniform sampler2D DiffuseSampler;

in vec2 texCoord;
out vec4 fragColor;

// https://www.handprint.com/ASTRO/IMG/seidel1.gif
#define CHROMATIC_ABERRATION
#define LATERAL_CHROMATIC_ABERRATION
#define LONGITUDINAL_CHROMATIC_ABERRATION
#define SAMPLES 12.0
#define SCALE 5.0

vec4 chromaticAberration( in sampler2D i, in vec2 uv)
{
    vec4 c = vec4(0.0);
    float d = texture2D(i, uv).w;
    for (float s = 0; s < SAMPLES; s++) {
        vec2 a = (s / (SCALE - ((d + 1.0) * SCALE))) * pow(uv - 0.5, vec2(3.0));
        c.r += texture2D(i, uv - a).r;
        c.b += texture2D(i, uv + a).b;
    }
    c.r /= SAMPLES;
    c.g = texture2D(i, uv).g;
    c.b /= SAMPLES;
    c.w = 1.0;
    return c;
}

void main()
{
    #ifdef CHROMATIC_ABERRATION
        fragColor = chromaticAberration(DiffuseSampler, texCoord);
    #else
        fragColor = vec4(texture2D(DiffuseSampler, texCoord).w);
    #endif
}
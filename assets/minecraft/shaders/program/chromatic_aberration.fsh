// Chromatic aberration.

#version 150

uniform sampler2D DiffuseSampler;

in vec2 texCoord;
out vec4 fragColor;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// CONFIG //////////////////////////////////////

//#define DEBUG
// Enable chromatic aberration.
#define CHROMATIC_ABERRATION
// How much to apply lateral chromatic aberration.
#define LATERAL_CHROMATIC_ABERRATION 1.0

// Not implemented yet. Most likely will be implemented in optica.fsh.
//#define LONGITUDINAL_CHROMATIC_ABERRATION

// Amount of samples. You shouldn't change this.
#define SAMPLES 3.0
// Scale of lateral chromatic aberration. Lower is higher.
#define SCALE 30.0
// Multiply bokeh highlight lateral chromatic aberration.
#define BOKEH_HIGHLIGHT_SCALE 5.0

//////////////////////////////////// END ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#define sat(x) clamp(x, 0.0, 1.0)

vec4 chromaticAberration( in sampler2D i, in vec2 uv)
{
    vec4 c = vec4(0.0);
    float d = texture2D(i, uv).w;
    for (float s = 0; s < SAMPLES; s++) {
        vec2 a = ((s / (SAMPLES * SCALE)) * (LATERAL_CHROMATIC_ABERRATION +
        (d * BOKEH_HIGHLIGHT_SCALE))) * pow(uv - 0.5, vec2(3.0));
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
    #ifdef DEBUG
        fragColor = texCoord.x > 0.5 ?
            vec4(texture2D(DiffuseSampler, texCoord).w) :
            vec4(texture2D(DiffuseSampler, texCoord).xyz, 1.0);
    #elif defined(CHROMATIC_ABERRATION)
        fragColor = chromaticAberration(DiffuseSampler, texCoord);
    #else
        fragColor = vec4(texture2D(DiffuseSampler, texCoord).xyz, 1.0);
    #endif
}
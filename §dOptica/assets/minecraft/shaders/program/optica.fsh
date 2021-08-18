////////// Made by Auralius#6109 | Project started August 12, 2021 /////////////
//                                   Optica                                   //
//                               © 2021 Auralius                              //
/*               MIT License https://opensource.org/licenses/MIT              //

Features,
 - Bokeh blur,
 - Smart auto focus,
 - Color correction,
 - Image filters,
 - Dynamic vignette,
 - Lens distortion,
 - Image noise,
 - Extreme customisation,
 - And much more to come!

/////////////////////////////// RESOURCES //////////////////////////////////////

https://photographylife.com/what-is-distortion
https://tuxedolabs.blogspot.com/2018/05/bokeh-depth-of-field-in-single-pass.html
github.com/orthecreedence/ghostie/blob/master/opengl/glsl/dof.bokeh.2.4.frag
https://github.com/trevorvanhoof/ColorGrading/blob/master/grading.glsl
https://dipaola.org/art/wp-content/uploads/2017/09/cgf2012.pdf
https://en.wikipedia.org/wiki/Optical_aberration
www.siliconstudio.co.jp/rd/presentations/files
    siggraph2015/06_MakingYourBokehFascinating_S2015_Kawase_EN.pdf
https://www.handprint.com/ASTRO/ae4.html
 https://www.lenstip.com/upload3/4164_zei85_bokeh.jpg
 https://smallpond.ca/jim/photomicrography/ca/index.html
slidetodoc.com/lenses-realtime-rendering-of-physically-based-optical-effect

//////////////////////////////////////////////////////////////////////////////*/

#version 150

uniform sampler2D DiffuseSampler;

uniform sampler2D DiffuseDepthSampler;
uniform sampler2D ParticlesDepthSampler;
uniform sampler2D WeatherDepthSampler;

uniform vec2 InSize;
uniform float Time;

in vec2 texCoord;
out vec4 fragColor;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// CONFIG //////////////////////////////////////
//                  40 different settings to play with!                       //

// More defines are located in chromatic_aberration.fsh.

// Experimental features.
//#define DEBUG
//#define DEBUG_SIMPLE
//#define DEBUG_LUMEN
//#define DEBUG_DEPTH
//#define DEBUG_GRAY

// Extremely heavy performance impact!
// Allows focus to effect particles / weather.
//#define HIGH_QUALITY_DEPTH

/////////////////////////////////// TOGGLES ////////////////////////////////////
//                      Uncomment lines to enable them.                       //

// Smoother focus.
#define SMART_FOCUS
// Enable bokeh blur.
#define BLUR
// Barrel distortion and whatnot.
#define LENS_DISTORTION
// Vignette that changes based on how close you are to something.
#define DYNAMIC_VIGNETTE
// Adjust hue and saturation.
#define COLOR_CORRECTION
// Luminance noise.
#define NOISE
// Image filters.
//#define FILTER
// Sepia.
//#define SEPIA

////////////////////////////// BOKEH HIGHLIGHTS ////////////////////////////////

// Highlights have to be emulated due to not having natural emissives.
// The minimum luminosity for something to be an emissive.
#define HIGHLIGHT_THRESHOLD 0.95
// How much brighter should highlights be.
#define HIGHLIGHT_GAIN 100.0
// Change bokeh highlight color. Not physically based.
#define BOKEH_HIGHLIGHT_COLOR 1.0, 1.0, 1.0

/////////////////////////////////// FOCUS //////////////////////////////////////

// Higher is worse. I would not change this.
#define SMART_FOCUS_QUALITY 75.0
// Lower is sharper.
#define SMART_FOCUS_SIZE 100.0

// Move the focus point around the screen.
#define FOCUS_POINT 0.5, 0.5

// Change bokeh blur size. Bigger is slower.
#define BLUR_SIZE 20.0
// Higher is worse.
#define BLUR_QUALITY 1.0
// This is a non existing variable in the real world,
// this adjusts how sharp the focus is. Higher is sharper.
#define FOCAL_PLANE_WIDTH 30.0
// Brightness fringing on bokeh.
#define BOKEH_FRINGE

// Low quality blur.
// Not suitable for photos, better for gaming.
//#define LOW_QUALITY_BLUR

/////////////////////// COLOR CORRECTION / ADJUSTMENT //////////////////////////

// Adjust hue. 0.5 = 180deg.
#define ADJUST_HUE 0.0
// Adjust saturation. 2.0 = 2x.
#define ADJUST_SATURATION 1.1
// Adjust exposure.
#define ADJUST_EXPOSURE 1.15
// Adjust contrast.
#define ADJUST_CONTRAST 1.1
// Adjust temperature. Currently not mapped to any real world values.
#define ADJUST_TEMPERATURE 150

////////////////////////////// LENS DISTORTION /////////////////////////////////

// If distortion should be mapped to sensor or lens size.
#define LENS_DISTORTION_CIRCULAR true
// Barrel distortion.
#define LENS_DISTORTION_BARREL 0.05
// Pincusion distortion.
#define LENS_DISTORTION_PINCUSHION 0.0
// This may be removed later and applied automatically, but as of now you will
// have to do it manually. Sorry!
#define LENS_DISTORTION_ZOOM 0.135

////////////////////////////////// FILTERS /////////////////////////////////////

// 0 - B&W     | CONTRAST ???   ???
// 1 - Overlay | RED      GREEN BLUE
#define FILTER_TYPE 1
#define FILTER_PARAMETER 0.0, 0.478, 1.0
// How much to apply the filter.
#define FILTER_STRENGTH 1.0

// Sepia.
#define SEPIA_AMOUNT 2.0

//////////////////////////////////// MISC //////////////////////////////////////

// Noise.
#define NOISE_SENSITIVITY 2.5

//////////////////////////////////// END ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////// MISC DEFINES ////////////////////////////////////

#define sat(x) clamp(x, 0.0, 1.0)
#define GOLDEN_ANGLE 2.39996
#define PI 3.14159

#define NEAR 1.0
#define FAR 100.0

vec2 oneTexel = vec2(1.0) / InSize;

////////////////////////////////// TOOLS ///////////////////////////////////////

// Get luminosity.
float luma( in vec3 c )
{
    return dot(c, vec3(0.299, 0.587, 0.114));
}

// Random float -1.0 - 1.0. | http://www.jstatsoft.org/v08/i14/paper
// Does not work currently and is unused.
float random( in float s )
{
    int x = int(s * 32767);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return float(x) / 2147483647.0;
}

////////////////////////////////// BLUR ////////////////////////////////////////

// Get size of blur based on focal point.
float blurSize( in float d, in float f, in float s )
{
    float o = clamp((1.0 / f - 1.0 / d) * s, -1.0, 1.0);
    return abs(o) * BLUR_SIZE;
}

// Map depth to linear space.
float depth( in sampler2D s, in vec2 uv )
{
    return (2.0 * NEAR) / (FAR + NEAR - texture2D(s, uv).x * (FAR - NEAR));
}

float depth( in float s)
{
    return (2.0 * NEAR) / (FAR + NEAR - s * (FAR - NEAR));
}

// High quality depth.
//Later if possible I want this to be in it's own buffer in the shader pipeline.
float adepth( in vec2 uv )
{
    float s[3];
    s[0] = texture2D(DiffuseDepthSampler, uv).x;
    s[1] = texture2D(ParticlesDepthSampler, uv).x;
    s[2] = texture2D(WeatherDepthSampler, uv).x;

    float d = 1.0;
    d = d > s[0] ? s[0] : d;
    d = d > s[1] ? s[1] : d;
    d = d > s[2] ? s[2] : d;

    return depth(d);
}

// Get dynamic focus.
float smartFocus( in vec2 uv, in float s )
{
    float t = 1.0;
    float r = SMART_FOCUS_QUALITY;

    float ac = texture2D(DiffuseDepthSampler, uv).x;
    for (float a = 0.0; r < s; a += GOLDEN_ANGLE)
    {
        vec2 tc =  uv + vec2(cos(a), sin(a)) * oneTexel * r;
        ac += texture2D(DiffuseDepthSampler, tc).x;
        t += 1.0;
        r += SMART_FOCUS_QUALITY / r;
    }
    return depth(ac /= t);
}

vec3 highlights( in vec3 c, in float i )
{
    c = mix(vec3(0.0), c, max((luma(c) - HIGHLIGHT_THRESHOLD)
        * HIGHLIGHT_GAIN, 0.0) * vec3(BOKEH_HIGHLIGHT_COLOR) *
        #ifdef BOKEH_FRINGE
            max(sqrt(i - (BLUR_SIZE * 0.5)) , 1.5)
        #else
            1.5
        #endif
    ); 
    //c -= max(sqrt(i - (BLUR_SIZE * 0.5)) , 1.5);
    return c;
}

//    Bokeh blur. Based off of this great blog post by Dennis Gustafsson.     //
//  tuxedolabs.blogspot.com/2018/05/bokeh-depth-of-field-in-single-pass.html  //
vec4 blur( in vec2 uv, in float f, in float s )
{
    #ifdef HIGH_QUALITY_DEPTH
        float cd = adepth(uv) * FAR;
    #else
        float cd = depth(DiffuseDepthSampler, uv) * FAR;
    #endif

    float cs = blurSize(cd, f, s);
    vec3 ac = texture2D(DiffuseSampler, uv).xyz;
    float t = 1.0;
    
    float r = BLUR_QUALITY;
    
    vec3 hc = vec3(0.0);

    for (float a = 0.0; r < BLUR_SIZE; a += GOLDEN_ANGLE)
    {
        vec2 tc = uv + vec2(cos(a), sin(a)) * oneTexel * r;
        vec3 sc = texture2D(DiffuseSampler, tc).xyz;

        #ifdef HIGH_QUALITY_DEPTH
            float sd = adepth(tc) * FAR;
        #else
            float sd = depth(DiffuseDepthSampler, tc) * FAR;
        #endif

        float ss = blurSize(sd, f, s);
        ss = sd > cd ? clamp(ss, 0.0, cs * 2.0) : ss;
        float m = smoothstep(r - 0.5, r + 0.5, ss);

        vec3 h = highlights(sc, r);
        hc += h * m;

        ac += mix(ac / t, sc + h, m);
        t += 1.0;
        r += BLUR_QUALITY / r;

        #ifdef LOW_QUALITY_BLUR
            if (m == 0.0) break;
        #endif
    }
    return vec4(ac /= t, hc / t);
}

///////////////////////////// POST PROCESSING //////////////////////////////////
////////////////////////////////// Tools ///////////////////////////////////////

// HSV > RGB.
vec3 hsv2rgb( in vec3 c )
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, sat(p - K.xxx), c.y);
}

// RGB > HSV.
vec3 rgb2hsv( in vec3 rgb )
{
    float Cmax = max(rgb.r, max(rgb.g, rgb.b));
    float Cmin = min(rgb.r, min(rgb.g, rgb.b));
    float delta = Cmax - Cmin;

    vec3 hsv = vec3(0.0, 0.0, Cmax);

    if (Cmax > Cmin) {
        hsv.y = delta / Cmax;

        hsv.x = rgb.r == Cmax ?
            (rgb.g - rgb.b) / delta :
            rgb.g == Cmax ?
                2.0 + (rgb.b - rgb.r) / delta :
                4.0 + (rgb.r - rgb.g) / delta;

        hsv.x = fract(hsv.x / 6.0);
    }
    return hsv;
 }

///////////////////////////////// Filters //////////////////////////////////////

// Hue, saturation, and value.
vec3 adjust( in vec3 c, in float h, in float s, in float v )
{
    vec3 o = rgb2hsv(c);
    o = vec3(o.x += h, o.y *= s, o.z *= v);

    return hsv2rgb(o);
}

// Overlay.
vec3 overlay( in vec3 c, in vec3 o )
{
    return mix(1.0-2.0 * (1.0-c) * (1.0 - o), 2.0 * c * o, step(c, vec3(0.5)));
}

// Applying color filters.
vec3 filter( in vec3 c, int t, float s, vec3 p)
{
    vec3 o = c;
    switch (t) {
        case 0:
            o = vec3(((luma(c) - 0.5) * p) + 0.5);
            break;
        case 1:
            o = overlay( c, hsv2rgb(vec3(p)));
    }
    return mix(c, o, s);
}

// Adjusting contrast.
vec3 contrast( in vec3 c, in float s )
{
    return ((c - 0.5) * s) + 0.5;
}

// Multiplication of floats centered around one.
float middle( in float c, in float s )
{
    return ((c - 1.0) * s) + 1.0;
}

// Adjusting temperature. | https://www.shadertoy.com/view/4sc3D7
vec3 temperature( in float t )
{
    mat3 m = (t <= 6500.0) ?
        mat3(
            vec3(0.0, -2902.19553, -8257.79972),
            vec3(0.0, 1669.58035, 2575.28275),
            vec3(1.0, 1.33026, 1.89937)
        ) : 
        mat3(
            vec3(1745.04252, 1216.61683, -8257.79972),
            vec3(-2666.34742, -2173.10123, 2575.28275),
            vec3(0.55995, 0.70381, 1.89937)
        ); 
    return mix(clamp(vec3(m[0] / (vec3(clamp(t, 1000.0, 40000.0))+ m[1])
    + m[2]), vec3(0.0), vec3(1.0)), vec3(1.0), smoothstep(1000.0, 0.0, t));
}

vec3 sepia( in float s )
{
    return vec3(middle(1.2, s), 1.0, middle(0.8, s));
}

///////////////////////////////// DISTORTION ///////////////////////////////////

// Lens distortion. Based off of https://www.shadertoy.com/view/wtBXRz
vec2 lensDistortion(in vec2 uv, in float k1, in float k2, in float r, in bool f)
{
    uv = uv * 2.0 - 1.0;
    if (f) uv.x *= InSize.x / InSize.y;

    uv *= 1.0 - r;
    
    float r2 = uv.x*uv.x + uv.y*uv.y;
    uv *= 1.0 + k1 * r2 + k2 * r2 * r2;

    if (f) uv.x /= InSize.x / InSize.y;
    uv = uv * 0.5 + 0.5;

    return uv;
}

/////////////////////////////////// OVERLAY ////////////////////////////////////

// Vignette.
float vignette( in vec2 uv, in float s )
{
    uv = (uv * 2.0) - 1.0;
    uv.x *= InSize.x / InSize.y;
    return mix(1.0, 1.0 - sqrt(dot(uv, uv)) * 0.5, s);
}

///////////////////////////// ACES TONE MAPPING ////////////////////////////////
//               Not used, but could be useful in the future.                 //
//   https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl  //

const mat3 ACESInputMat = mat3(

    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777
);

const mat3 ACESOutputMat = mat3(
    1.60475, -0.53108, -0.07367,
    -0.10208,  1.10813, -0.00605,
    -0.00327, -0.07276,  1.07602
);

vec3 RRTAndODTFit(vec3 v)
{
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

vec3 ACESFitted(vec3 c)
{
    c = c * ACESInputMat;
    c = RRTAndODTFit(c);
    c = c * ACESOutputMat;

    c = sat(c);

    return c;
}

////////////////////////////////// Noise ///////////////////////////////////////
//                     Noise helper functions and more.                       //

// https://github.com/mattdesl/glsl-blend-soft-light/blob/master/index.glsl
vec3 blend(vec3 base, vec3 blend) {
    return mix(
        sqrt(base) * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend), 
        2.0 * base * blend + base * base * (1.0 - 2.0 * blend), 
        step(base, vec3(0.5))
    );
}

vec3 mod289( vec3 x )
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289( vec4 x )
{
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute( vec4 x )
{
    return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt( vec4 r )
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

vec3 fade( vec3 t )
{
    return t*t*t*(t*(t*6.0-15.0)+10.0);
}

// Copyright (c) 2011 Stefan Gustavson. All rights reserved.
float pnoise3D( vec3 P, vec3 rep )
{
    vec3 Pi0 = mod(floor(P), rep);
    vec3 Pi1 = mod(Pi0 + vec3(1.0), rep);
    Pi0 = mod289(Pi0);
    Pi1 = mod289(Pi1);
    vec3 Pf0 = fract(P);
    vec3 Pf1 = Pf0 - vec3(1.0);
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz;
    vec4 iz1 = Pi1.zzzz;

    vec4 ixy = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);

    vec4 gx0 = ixy0 * (1.0 / 7.0);
    vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
    gx0 = fract(gx0);
    vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0));
    gx0 -= sz0 * (step(0.0, gx0) - 0.5);
    gy0 -= sz0 * (step(0.0, gy0) - 0.5);

    vec4 gx1 = ixy1 * (1.0 / 7.0);
    vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
    gx1 = fract(gx1);
    vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
    vec4 sz1 = step(gz1, vec4(0.0));
    gx1 -= sz1 * (step(0.0, gx1) - 0.5);
    gy1 -= sz1 * (step(0.0, gy1) - 0.5);

    vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
    vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
    vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
    vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
    vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
    vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
    vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
    vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);

    vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000),dot(g010, g010),
        dot(g100, g100), dot(g110, g110)));
    g000 *= norm0.x;
    g010 *= norm0.y;
    g100 *= norm0.z;
    g110 *= norm0.w;
    vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011),
        dot(g101, g101), dot(g111, g111)));
    g001 *= norm1.x;
    g011 *= norm1.y;
    g101 *= norm1.z;
    g111 *= norm1.w;

    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
    float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
    float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
    float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
    float n111 = dot(g111, Pf1);

    vec3 fade_xyz = fade(Pf0);
    vec4 n_z = mix(vec4(n000, n100, n010, n110),
        vec4(n001, n101, n011, n111), fade_xyz.z);
    vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
    float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
    return 2.2 * n_xyz;
}

float grain(vec2 uv, vec2 r, float t) {
    float n1 = pnoise3D(vec3(uv * r, 0.0), vec3(1.0 / uv + t, 0.0));
    return n1 / 2.0 + 0.5;
}

//                           End of shader functions.                         //
////////////////////////////////////////////////////////////////////////////////

void main()
{
    vec2 uv = texCoord;

    // Dynamic focus.
    #ifdef SMART_FOCUS
        float focus = smartFocus(vec2(FOCUS_POINT), SMART_FOCUS_SIZE);
    #elif defined(BLUR)
        float focus = depth(DiffuseDepthSampler, vec2(FOCUS_POINT));
    #endif

    // Lens distortion.
    #ifdef LENS_DISTORTION
        uv = lensDistortion(
            uv,LENS_DISTORTION_BARREL,
            LENS_DISTORTION_PINCUSHION,
            LENS_DISTORTION_ZOOM,
            LENS_DISTORTION_CIRCULAR
        );
    #endif

    #ifdef DEBUG_LUMEN
        float lum = luma(texture2D(DiffuseSampler, uv).xyz);
    #endif

    // Fallback.
    #ifdef BLUR
        vec3 c = vec3(0.0);
    #else
        vec3 c = texture2D(DiffuseSampler, uv).xyz;
    #endif
    
    // Bokeh blur.
    #ifdef BLUR
        vec4 bokeh = sat(blur(uv, focus * FAR, FOCAL_PLANE_WIDTH));
        c = bokeh.xyz;
    #endif

    // Image noise.
    #ifdef NOISE
        float r = smoothstep(0.05, 0.5, luma(c * NOISE_SENSITIVITY));
        if (luma(c * NOISE_SENSITIVITY) < 0.5) {
            vec3 g = vec3(grain(texCoord, InSize, Time));
            vec3 o = blend(c, g);
            c = mix(o, c, pow(r, 2.0));
        }
    #endif

    // Color corrections.
    #ifndef DEBUG
        #ifdef COLOR_CORRECTION
            c = sat(adjust(c, ADJUST_HUE, ADJUST_SATURATION, ADJUST_EXPOSURE));
            c = contrast(c, ADJUST_CONTRAST);
            c *= temperature(ADJUST_TEMPERATURE);
        #endif
    #endif

    // Image filters.
    #ifdef FILTER
        c = filter(c, FILTER_TYPE, FILTER_STRENGTH, vec3(FILTER_PARAMETER));
    #endif

    // Sepia.
    #ifndef DEBUG
        #ifdef SEPIA
            c = mix(c, vec3(luma(c)) * sepia(SEPIA_AMOUNT), sat(SEPIA_AMOUNT));
        #endif
    #endif

    // Dynamic vignette.
    #ifdef DYNAMIC_VIGNETTE
        #ifdef BLUR
            c *= vignette(uv, clamp(1.0 - focus, 0.5, 1.0));
        #endif
    #endif

    // Debug.
    #ifdef DEBUG
        #ifdef DEBUG_LUMEN
            float ldif = luma(c) - lum;
            c = sat(vec3(-ldif, 0.0, ldif));
        #else
            c = mix(texture2D(DiffuseSampler, texCoord).xyz,
                c, smoothstep(0.499, 0.501, uv.x));
            vec3 c2 = c;
            #ifdef COLOR_CORRECTION
                c2 = adjust(c2, ADJUST_HUE, ADJUST_SATURATION, ADJUST_EXPOSURE);
                c2 = contrast(c2, ADJUST_CONTRAST);
                c2 *= temperature(ADJUST_TEMPERATURE);
            #endif
            #ifdef SEPIA
                c2 = mix(c, vec3(luma(c)) * sepia(SEPIA_AMOUNT),
                    sat(SEPIA_AMOUNT));
            #endif
            #ifndef DEBUG_SIMPLE
                c = mix(c, c2, smoothstep(0.499, 0.501, uv.y));
            #endif
        #endif
        #ifdef DEBUG_DEPTH
            c = mix(c, vec3(depth(texture2D(DiffuseDepthSampler, uv).x)),
                smoothstep(0.499, 0.501, uv.x));
        #endif
        #ifdef DEBUG_GRAY
            c = vec3(luma(c));
        #endif
    #endif

    //c *= 1.0 - depth(texture2D(DiffuseDepthSampler, uv).x);
    #ifdef BLUR
        fragColor = vec4(c, bokeh.w);
    #else
        fragColor = vec4(c, 0.0);
    #endif
}

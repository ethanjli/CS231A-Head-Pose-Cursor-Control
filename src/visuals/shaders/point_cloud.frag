#version 120

// Constants
// ------------------------------------


// Varyings
// ------------------------------------
varying vec4 v_color;
varying float v_size;
varying float v_linewidth;
varying float v_antialias;

// Functions
// ------------------------------------

// ----------------
float disc(vec2 P, float size)
{
    float r = length((P.xy - vec2(0.5,0.5))*size);
    r -= v_size/2;
    return r;
}

// Main
// ------------------------------------
void main()
{
    float size = v_size +2*(v_linewidth + 1.5*v_antialias);
    float t = v_linewidth/2.0-v_antialias;

    float r = disc(gl_PointCoord, size);

    float d = abs(r) - t;
    if( r > (v_linewidth/2.0+v_antialias))
    {
        discard;
    }
    /*
    // Anti-aliasing, doesn't currently work the way we need
    else if( r > 0.0 )
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        gl_FragColor = vec4(v_color.rgb, alpha);
    }
    */
    else
    {
        gl_FragColor = v_color;
    }
}


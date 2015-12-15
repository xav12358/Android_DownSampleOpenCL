#define BLK_SIZE    16
//prefix D for decoding
#define DSHRD_LEN   (BLK_SIZE/2)
#define DSHRD_SIZE  (2*DSHRD_LEN*DSHRD_LEN)

uchar convertYVUtoRGBA(int y, int u, int v)
{
    //uchar4 ret;
    y-=16;
    u-=128;
    v-=128;

    int val = 255-(0.403936*u+0.838316*v+y);

    return val;
}

__kernel void nv21togray( __global uchar* out,
                          __global uchar*  in,
                          int    im_width,
                          int    im_height,
                          int 	 im_offset)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;    int gx	= get_global_id(0);
    int gy	= get_global_id(1);
    
    int inIdx= gy*im_width+gx;
    int uvIdx= im_offset + (gy/2)*im_width + (gx & ~1);
    int shlx = gx/2;
    int shly = gy/2;
    int shIdx= im_offset+(shlx+shly*im_width);
    
    // do some work while others copy
    // uv to shared memory
    int y   =  (int)in[inIdx];
    
    if( gx >= im_width || gy >= im_height )
        return;
    // convert color space
    int v   = ((int)in[shIdx+0]);
    int u   = ((int)in[shIdx+1]);
    // write output to image
    out[inIdx]  = 255-(0.403936*(u-128)+0.838316*(v-128)+(y-16));
}


__kernel void downfilter_x_g( 
    __global uchar *src,
    __global uchar *dst, int w, int h )
{

    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    
    float x0 = src[ix-2+(iy)*w]/16.0;
    float x1 = src[ix-1+(iy)*w]/8.0;
    float x2 = 3*src[ix+(iy)*w]/4.0;
    float x3 = src[ix+1+(iy)*w]/8.0;
    float x4 = src[ix+2+(iy)*w]/16.0;
    

    int output = round( x0 + x1 + x2 + x3 + x4 );
	if(output >255)
		output = 255;
    if( ix < w && iy < h ) {
        dst[iy*w + ix ] = 255-output;
    }
}



__kernel void downfilter_y_g(
    __global uchar* src,
    __global uchar *dst, int w, int h )
{

    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    float x0 = src[2*ix+(2*iy-2)*w*2]/16.0;
    float x1 = src[2*ix+(2*iy-1)*w*2]/8.0;
    float x2 = 3*src[2*ix+(2*iy)*w*2]/4.0;
    float x3 = src[2*ix+(2*iy+1)*w*2]/8.0;
    float x4 = src[2*ix+(2*iy+2)*w*2]/16.0;
    
    int output = round(x0 + x1 + x2 + x3 + x4);

	if(output >255)
		output = 255;
    if( ix < w && iy < h ) {
        dst[iy*w + ix ] = 255-output;
    }
 
}




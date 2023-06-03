#include "template/common.h"
#include "cl/tools.cl"

#define GRIDSIZE 256

typedef struct Point
{
    float2 pos;
    float2 prev_pos;
    float2 fix;
    bool fixed; 
    float restlength[4];
} Point;


typedef struct Rands {
	int rand1;
	float2 rand2; 
} Rands;

__kernel void render( __global Point* points, float magic)
{
	// plot a pixel to outimg
	const int p = get_global_id( 0 );
	const int x = p % GRIDSIZE;
	const int y = p / GRIDSIZE; 
	const int index = x + y * GRIDSIZE;
	const int seed =  WangHash((p+1)*17);

	float2 curpos = points[index].pos, prevpos = points[index].prev_pos;
	points[index].pos += (curpos - prevpos) + (float2)(0, 0.003f); // gravity

	points[index].prev_pos = curpos;

	int rand1 = (int)((RandomFloat(&seed) * 10.0f) < 0.03f);
	float2 rand2 = (float2)(RandomFloat(&seed) * (0.02f + magic), RandomFloat(&seed) * 0.12f);

	points[index].pos += rand1 * rand2;

}

// EOF
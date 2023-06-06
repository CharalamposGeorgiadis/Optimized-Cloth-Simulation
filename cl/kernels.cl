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

typedef struct Rands 
{
	float rand10;
	float rand002;
	float rand012;
} Rands;

int xoffset[4] = { 1, -1, 0, 0 };
int yoffset[4] = { 0, 0, 1, -1 };

float2 gravity = (float2)(0, 0.003f);


__kernel void render( __global Point* points, float magic, __global Rands* rands)
{
	const int p = get_global_id( 0 );
	const int x = p % GRIDSIZE;
	const int y = p / GRIDSIZE; 
	const int index = x + y * GRIDSIZE;

	float2 curpos = points[index].pos;
	float2 prevpos = points[index].prev_pos;
	points[index].pos += (curpos - prevpos) + gravity;

	points[index].prev_pos = curpos;

	int rand1 = (int)(rands[index].rand10 < 0.03f);
	float2 rand2 = (float2)(rands[index].rand002, rands[index].rand012);

	points[index].pos += (rand1 * rand2);
}

__kernel void constraints(__global Point* points, int i, int even)
{
	const int p = get_global_id( 0 );
	const int x = p % (GRIDSIZE - 2) + 1;
	const int y = p / (GRIDSIZE - 2) + 1; 
	const int index = x + y * GRIDSIZE;

	if (index % 2 == even)
        return;

	int neighbor_index = x + xoffset[i] + (y + yoffset[i]) * GRIDSIZE;
	Point neighbor = points[neighbor_index];

	float2 diff = neighbor.pos - points[index].pos;
	float distance = length(diff);

	int if_1 = (int)isfinite(distance);

	if (distance > points[index].restlength[i])
	{
		float extra = distance / points[index].restlength[i] - 1;
		float2 mult = if_1 * extra * diff * 0.5f;

		points[index].pos += mult;
		points[neighbor_index].pos -= mult;
	}
}

__kernel void fix(__global Point* points)
{
	const int x = get_global_id( 0 );
	points[x].pos = points[x].fix;
}

// EOF
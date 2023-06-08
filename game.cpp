#include "precomp.h"
#include "game.h"

#define GRIDSIZE 256

// VERLET CLOTH SIMULATION DEMO
// High-level concept: a grid consists of points, each connected to four 
// neighbours. For a simulation step, the position of each point is affected
// by its speed, expressed as (current position - previous position), a
// constant gravity force downwards, and random impulses ("wind").
// The final force is provided by the bonds between points, via the four
// connections.
// Together, this simple scheme yields a pretty convincing cloth simulation.
// The algorithm has been used in games since the game "Thief".

// ASSIGNMENT STEPS:
// 1. SIMD, part 1: in Game::Simulation, convert lines 119 to 126 to SIMD.
//    You receive 2 points if the resulting code is faster than the original.
//    This will probably require a reorganization of the data layout, which
//    may in turn require changes to the rest of the code.
// 2. SIMD, part 2: for an additional 4 points, convert the full Simulation
//    function to SSE. This may require additional changes to the data to
//    avoid concurrency issues when operating on neighbouring points.
//    The resulting code must be at least 2 times faster (using SSE) or 4
//    times faster (using AVX) than the original  to receive the full 4 points.
// 3. GPGPU, part 1: modify Game::Simulation so that it sends the cloth data
//    to the GPU, and execute lines 119 to 126 on the GPU. After this, bring
//    back the cloth data to the CPU and execute the remainder of the Verlet
//    simulation code. You receive 2 points if the code *works* correctly;
//    note that this is expected to be slower due to the data transfers.
// 4. GPGPU, part 2: execute the full Game::Simulation function on the GPU.
//    You receive 4 additional points if this yields a correct simulation
//    that is at least 5x faster than the original code. DO NOT draw the
//    cloth on the GPU; this is (for now) outside the scope of the assignment.
// Note that the GPGPU tasks will benefit from the SIMD tasks.
// Also note that your final grade will be capped at 10.

// create a new key
avx_xorshift128plus_key_t mykey;


struct Point
{
	float2 pos;				// current position of the point
	float2 prev_pos;		// position of the point in the previous frame
	float2 fix;				// stationary position; used for the top line of points
	bool fixed;				// true if this is a point in the top line of the cloth
	float restlength[4];	// initial distance to neighbours
};


struct Points
{
	float* pos_x;
	float* pos_y;
	float* prev_pos_x;
	float* prev_pos_y;
	float* fix_x;
	float* fix_y;
	bool* fixed;
	float* restlength0;
	float* restlength1;
	float* restlength2;
	float* restlength3;
};
Points points;


// grid access convenience
Point* pointGrid = new Point[GRIDSIZE * GRIDSIZE];
Point& grid(const uint x, const uint y) { return pointGrid[x + y * GRIDSIZE]; }


// grid offsets for the neighbours via the four links
int xoffset[4] = { 1, -1, 0, 0 }, yoffset[4] = { 0, 0, 1, -1 };

// initialization
void Game::Init()
{
	avx_xorshift128plus_init(Rand(1000), Rand(10000), &mykey);

	points.pos_x = new float[GRIDSIZE * GRIDSIZE];
	points.pos_y = new float[GRIDSIZE * GRIDSIZE];
	points.prev_pos_x = new float[GRIDSIZE * GRIDSIZE];
	points.prev_pos_y = new float[GRIDSIZE * GRIDSIZE];
	points.fix_x = new float[GRIDSIZE * GRIDSIZE];
	points.fix_y = new float[GRIDSIZE * GRIDSIZE];
	points.fixed = new bool[GRIDSIZE * GRIDSIZE];

	points.restlength0 = new float[GRIDSIZE * GRIDSIZE];
	points.restlength1 = new float[GRIDSIZE * GRIDSIZE];
	points.restlength2 = new float[GRIDSIZE * GRIDSIZE];
	points.restlength3 = new float[GRIDSIZE * GRIDSIZE];

	for (int y = 0; y < GRIDSIZE; y++)
		for (int x = 0; x < GRIDSIZE; x++)
		{
			int index = x + y * GRIDSIZE;
			points.pos_x[index] = 10 + (float)x * ((SCRWIDTH - 100) / GRIDSIZE) + y * 0.9f + Rand(2);
			points.pos_y[index] = 10 + (float)y * ((SCRHEIGHT - 180) / GRIDSIZE) + Rand(2);
			points.prev_pos_x[index] = points.pos_x[index];
			points.prev_pos_y[index] = points.pos_y[index];

			if (y == 0)
			{
				points.fixed[index] = true;
				points.fix_x[index] = points.pos_x[index];
				points.fix_y[index] = points.pos_y[index];
			}
			else
			{
				points.fixed[index] = false;
			}
		}

	for (int y = 1; y < GRIDSIZE - 1; y++)
		for (int x = 1; x < GRIDSIZE - 1; x++)
		{
			int index = x + y * GRIDSIZE;
			for (int c = 0; c < 4; c++)
			{
				int neighbourIndex = (x + xoffset[c]) + (y + yoffset[c]) * GRIDSIZE;
				float dx = points.pos_x[index] - points.pos_x[neighbourIndex];
				float dy = points.pos_y[index] - points.pos_y[neighbourIndex];
				float distance = sqrtf(dx * dx + dy * dy);

				switch (c)
				{
				case 0:
					points.restlength0[index] = distance * 1.15f;
					break;
				case 1:
					points.restlength1[index] = distance * 1.15f;
					break;
				case 2:
					points.restlength2[index] = distance * 1.15f;
					break;
				case 3:
					points.restlength3[index] = distance * 1.15f;
					break;
				}
			}
		}
}

// cloth rendering
// NOTE: For this assignment, please do not attempt to render directly on
// the GPU. Instead, if you use GPGPU, retrieve simulation results each frame
// and render using the function below. Do not modify / optimize it.
void Game::DrawGrid()
{
	// draw the grid
	screen->Clear(0);

	for (int y = 0; y < (GRIDSIZE - 1); y++)
		for (int x = 1; x < (GRIDSIZE - 2); x++)
		{
			int index = x + y * GRIDSIZE;
			int nextXIndex = (x + 1) + y * GRIDSIZE;
			int nextYIndex = x + (y + 1) * GRIDSIZE;

			float p1_x = points.pos_x[index];
			float p1_y = points.pos_y[index];
			float p2_x = points.pos_x[nextXIndex];
			float p2_y = points.pos_y[nextXIndex];
			float p3_x = points.pos_x[nextYIndex];
			float p3_y = points.pos_y[nextYIndex];

			screen->Line(p1_x, p1_y, p2_x, p2_y, 0xffffff);
			screen->Line(p1_x, p1_y, p3_x, p3_y, 0xffffff);
		}

	for (int y = 0; y < (GRIDSIZE - 1); y++)
	{
		int index = (GRIDSIZE - 2) + y * GRIDSIZE;
		int nextYIndex = (GRIDSIZE - 2) + (y + 1) * GRIDSIZE;

		float p1_x = points.pos_x[index];
		float p1_y = points.pos_y[index];
		float p2_x = points.pos_x[nextYIndex];
		float p2_y = points.pos_y[nextYIndex];

		screen->Line(p1_x, p1_y, p2_x, p2_y, 0xffffff);
	}
}

// cloth simulation
// This function implements Verlet integration (see notes at top of file).
// Important: when constraints are applied, typically two points are
// drawn together to restore the rest length. When running on the GPU or
// when using SIMD, this will only work if the two vertices are not
// operated upon simultaneously (in a vector register, or in a warp).
float magic = 0.11f;
__m256 intmax = _mm256_set1_ps(INT_MAX);

__m256 Rand8(float max_range) {
	__m256i random_int = avx_xorshift128plus(&mykey);
	__m256 random_float = _mm256_cvtepi32_ps(random_int);
	random_float = _mm256_div_ps(random_float, intmax);
	return _mm256_mul_ps(random_float, _mm256_set1_ps(max_range));
}

__m256 gravity = _mm256_set1_ps(0.003f);
__m256 zero = _mm256_setzero_ps();
__m256 one = _mm256_set1_ps(1);
__m256 zero_point_five = _mm256_set1_ps(0.5f);
__m256 infinite = _mm256_set1_ps(INFINITY);
__m256 zero_point_3 = _mm256_set1_ps(0.03f);


// Updates points and dependent neighbhors
void update_dependent_neighbors(float* restlengthList, int linknr, __m256i indices, __m256& pointpos_x, __m256& pointpos_y)
{
	// Calculate neighbour indices
	__m256i neighbourIndices = _mm256_add_epi32(indices, _mm256_set1_epi32(xoffset[linknr] + yoffset[linknr] * GRIDSIZE));

	// Load neighbour positions
	__m256 neighbourPosX = _mm256_i32gather_ps(points.pos_x, neighbourIndices, 4);
	__m256 neighbourPosY = _mm256_i32gather_ps(points.pos_y, neighbourIndices, 4);

	// Calculate dx, dy, and distance
	__m256 dx = _mm256_sub_ps(neighbourPosX, pointpos_x);
	__m256 dy = _mm256_sub_ps(neighbourPosY, pointpos_y);
	__m256 distance = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)));

	// Create a mask that has true where distance is finite
	__m256 not_inf_mask = _mm256_and_ps(_mm256_cmp_ps(distance, distance, _CMP_EQ_OQ), _mm256_cmp_ps(distance, infinite, _CMP_NEQ_OQ));

	// Load restlength
	__m256 restlength = _mm256_i32gather_ps(restlengthList, indices, 4);

	// Calculate condition mask (distance > restlength)
	__m256 mask = _mm256_cmp_ps(distance, restlength, _CMP_GT_OQ);

	// Calculate extra, dir_x, dir_y, and updates
	__m256 extra = _mm256_sub_ps(_mm256_div_ps(distance, restlength), one);
	extra = _mm256_and_ps(_mm256_sub_ps(_mm256_div_ps(distance, restlength), one), mask); // Set extra to 0 where condition is false

	__m256 update = _mm256_mul_ps(_mm256_and_ps(_mm256_sub_ps(_mm256_div_ps(distance, restlength), one), mask), zero_point_five);

	pointpos_x = _mm256_blendv_ps(pointpos_x, _mm256_add_ps(pointpos_x, _mm256_mul_ps(update, dx)), not_inf_mask);
	pointpos_y = _mm256_blendv_ps(pointpos_y, _mm256_add_ps(pointpos_y, _mm256_mul_ps(update, dy)), not_inf_mask);

	// Subtract from neighbour positions
	neighbourPosX = _mm256_blendv_ps(neighbourPosX, _mm256_sub_ps(neighbourPosX, _mm256_mul_ps(update, dx)), not_inf_mask);
	neighbourPosY = _mm256_blendv_ps(neighbourPosY, _mm256_sub_ps(neighbourPosY, _mm256_mul_ps(update, dy)), not_inf_mask);

	// Scatter updated positions
	for (int i = 0; i < 8; i++)
	{
		points.pos_x[indices.m256i_i32[i]] = ((float*)&pointpos_x)[i];
		points.pos_y[indices.m256i_i32[i]] = ((float*)&pointpos_y)[i];
		points.pos_x[neighbourIndices.m256i_i32[i]] = ((float*)&neighbourPosX)[i];
		points.pos_y[neighbourIndices.m256i_i32[i]] = ((float*)&neighbourPosY)[i];
	}
}

// Updates points and non-dependent neighbhors
void update_non_dependent_neighbors(float* restlengthList, int linknr, int x, int y)
{
	// Calculate indices for point positions
	int index = x + y * GRIDSIZE;

	// Load point positions
	__m256 pointpos_x = _mm256_loadu_ps(&points.pos_x[index]);
	__m256 pointpos_y = _mm256_loadu_ps(&points.pos_y[index]);

	// Calculate neighbour index
	int neighbourIndex = index + xoffset[linknr] + yoffset[linknr] * GRIDSIZE;

	// Load neighbour positions
	__m256 neighbourPosX = _mm256_loadu_ps(&points.pos_x[neighbourIndex]);
	__m256 neighbourPosY = _mm256_loadu_ps(&points.pos_y[neighbourIndex]);

	// Calculate dx, dy, and distance
	__m256 dx = _mm256_sub_ps(neighbourPosX, pointpos_x);
	__m256 dy = _mm256_sub_ps(neighbourPosY, pointpos_y);
	__m256 distance = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)));

	// Create a mask that has true where distance is finite
	__m256 not_inf_mask = _mm256_and_ps(_mm256_cmp_ps(distance, distance, _CMP_EQ_OQ), _mm256_cmp_ps(distance, infinite, _CMP_NEQ_OQ));

	// Load restlength
	__m256 restlength = _mm256_loadu_ps(&restlengthList[index]);

	// Calculate condition mask (distance > restlength)
	__m256 mask = _mm256_cmp_ps(distance, restlength, _CMP_GT_OQ);

	// Calculate extra, dir_x, dir_y, and updates
	__m256 extra = _mm256_sub_ps(_mm256_div_ps(distance, restlength), one);
	extra = _mm256_and_ps(extra, mask); // Set extra to 0 where condition is false

	__m256 update = _mm256_mul_ps(extra, zero_point_five);

	// Apply finite masks to point positions
	pointpos_x = _mm256_blendv_ps(pointpos_x, _mm256_add_ps(pointpos_x, _mm256_mul_ps(update, dx)), not_inf_mask);
	pointpos_y = _mm256_blendv_ps(pointpos_y, _mm256_add_ps(pointpos_y, _mm256_mul_ps(update, dy)), not_inf_mask);

	// Subtract from neighbour positions
	neighbourPosX = _mm256_blendv_ps(neighbourPosX, _mm256_sub_ps(neighbourPosX, _mm256_mul_ps(update, dx)), not_inf_mask);
	neighbourPosY = _mm256_blendv_ps(neighbourPosY, _mm256_sub_ps(neighbourPosY, _mm256_mul_ps(update, dy)), not_inf_mask);

	// Store updated positions
	_mm256_storeu_ps(&points.pos_x[index], pointpos_x);
	_mm256_storeu_ps(&points.pos_y[index], pointpos_y);
	_mm256_storeu_ps(&points.pos_x[neighbourIndex], neighbourPosX);
	_mm256_storeu_ps(&points.pos_y[neighbourIndex], neighbourPosY);
}

void Game::Simulation()
{
	// simulation is exected three times per frame; do not change this.
	for (int steps = 0; steps < 3; steps++)
	{
		for (int y = 0; y < GRIDSIZE; y++)
			for (int x = 0; x < GRIDSIZE; x += 8)
			{
				int index = x + y * GRIDSIZE;

				__m256 curpos_x = _mm256_loadu_ps(&points.pos_x[index]);
				__m256 curpos_y = _mm256_loadu_ps(&points.pos_y[index]);
				__m256 prevpos_x = _mm256_loadu_ps(&points.prev_pos_x[index]);
				__m256 prevpos_y = _mm256_loadu_ps(&points.prev_pos_y[index]);

				__m256 newpos_x = _mm256_add_ps(curpos_x, _mm256_sub_ps(curpos_x, prevpos_x));
				__m256 newpos_y = _mm256_add_ps(_mm256_add_ps(curpos_y, _mm256_sub_ps(curpos_y, prevpos_y)), gravity);

				_mm256_storeu_ps(&points.prev_pos_x[index], curpos_x);
				_mm256_storeu_ps(&points.prev_pos_y[index], curpos_y);

				__m256 mask = _mm256_cmp_ps(Rand8(10), zero_point_3, _CMP_LT_OQ);

				__m256 impx = _mm256_blendv_ps(zero, Rand8(0.02f + magic), mask);
				__m256 impy = _mm256_blendv_ps(zero, Rand8(0.12f), mask);

				newpos_x = _mm256_add_ps(newpos_x, impx);
				newpos_y = _mm256_add_ps(newpos_y, impy);

				_mm256_storeu_ps(&points.pos_x[index], newpos_x);
				_mm256_storeu_ps(&points.pos_y[index], newpos_y);
			}

		magic += 0.0002f; // slowly increases the chance of anomalies
		// apply constraints; 4 simulation steps: do not change this number.

		for (int i = 0; i < 4; i++)
		{
				for (int y = 1; y < GRIDSIZE - 1; y++)
				{
					__m256i ymask = _mm256_set1_epi32(y * GRIDSIZE);

					// Dependent neighbhors, skip indices and use gather to avoid concurrency issues. 
					for (int checker = 0; checker < 2; checker++)
					{
						for (int x = 1 + checker; x < GRIDSIZE - 1; x += 16) // Two passes over the data, offset by 1 on the second pass
						{
							// Create indices for SIMD gather
							__m256i indices = _mm256_setr_epi32(x, x + 2, x + 4, x + 6, x + 8, x + 10, x + 12, x + 14);

							indices = _mm256_add_epi32(indices, ymask);

							// Load point positions
							__m256 pointpos_x = _mm256_i32gather_ps(points.pos_x, indices, 4);
							__m256 pointpos_y = _mm256_i32gather_ps(points.pos_y, indices, 4);

							update_dependent_neighbors(points.restlength0, 0, indices, pointpos_x, pointpos_y);

							update_dependent_neighbors(points.restlength1, 1, indices, pointpos_x, pointpos_y);
						}
					}			
					// Independent neighbhors, no need to use gather, or skip indices
					for (int x = 1; x < GRIDSIZE - 1; x += 8)
					{
						update_non_dependent_neighbors(points.restlength2, 2, x, y);

						update_non_dependent_neighbors(points.restlength3, 3, x, y);
					}					
				}

				for (int x = 0; x < GRIDSIZE; x += 8) 
				{
					__m256 fix_x = _mm256_loadu_ps(&points.fix_x[x]);
					__m256 fix_y = _mm256_loadu_ps(&points.fix_y[x]);

					_mm256_storeu_ps(&points.pos_x[x], fix_x);
					_mm256_storeu_ps(&points.pos_y[x], fix_y);
				}			
		}
	}
}

void Game::Tick(float a_DT)
{
	// update the simulation
	Timer tm;
	tm.reset();
	Simulation();
	float elapsed1 = tm.elapsed();

	// draw the grid
	tm.reset();
	DrawGrid();
	float elapsed2 = tm.elapsed();

	// display statistics
	char t[128];
	sprintf(t, "ye olde ruggeth cloth simulation: %5.1f ms", elapsed1 * 1000);
	screen->Print(t, 2, SCRHEIGHT - 24, 0xffffff);
	sprintf(t, "                       rendering: %5.1f ms", elapsed2 * 1000);
	screen->Print(t, 2, SCRHEIGHT - 14, 0xffffff);
}
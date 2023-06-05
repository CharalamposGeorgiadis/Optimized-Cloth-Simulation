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

struct Point
{
	float2 pos;				// current position of the point
	float2 prev_pos;		// position of the point in the previous frame
	float2 fix;				// stationary position; used for the top line of points
	bool fixed;				// true if this is a point in the top line of the cloth
	float restlength[4];	// initial distance to neighbours
};

// grid access convenience
Point* pointGrid = new Point[GRIDSIZE * GRIDSIZE];

Point& grid(const uint x, const uint y) { return pointGrid[x + y * GRIDSIZE]; }

// grid offsets for the neighbours via the four links
int xoffset[4] = { 1, -1, 0, 0 }, yoffset[4] = { 0, 0, 1, -1 };

Kernel* kernel_render;
Kernel* kernel_constraints;

Buffer* buffer;
Buffer* randsBuffer;

cl_command_queue queue_render;
cl_command_queue queue_constraints;

struct Rands {
	float rand10;
	float rand002;
	float rand012;
};

Rands* rands = new Rands[GRIDSIZE * GRIDSIZE];

// initialization
void Game::Init()
{
	kernel_render = new Kernel("cl\\kernels.cl", "render");
	kernel_constraints = new Kernel("cl\\kernels.cl", "constraints");

	buffer = new Buffer(GRIDSIZE * GRIDSIZE * sizeof(Point), pointGrid, buffer->WRITEONLY);
	randsBuffer = new Buffer(GRIDSIZE * GRIDSIZE * sizeof(Rands), rands, randsBuffer->READONLY);

	queue_render = kernel_render->GetQueue();
	queue_constraints = kernel_constraints->GetQueue();

	// create the cloth
	for (int y = 0; y < GRIDSIZE; y++) 
		for (int x = 0; x < GRIDSIZE; x++)
		{
			grid(x, y).pos.x = 10 + (float)x * ((SCRWIDTH - 100) / GRIDSIZE) + y * 0.9f + Rand(2);
			grid(x, y).pos.y = 10 + (float)y * ((SCRHEIGHT - 180) / GRIDSIZE) + Rand(2);
			grid(x, y).prev_pos = grid(x, y).pos; // all points start stationary
			if (y == 0)
			{
				grid(x, y).fixed = true;
				grid(x, y).fix = grid(x, y).pos;
			}
			else
			{
				grid(x, y).fixed = false;
			}
		}
	for (int y = 1; y < GRIDSIZE - 1; y++) 
		for (int x = 1; x < GRIDSIZE - 1; x++)
		{
			// calculate and store distance to four neighbours, allow 15% slack
			for (int c = 0; c < 4; c++)
			{
				grid(x, y).restlength[c] = length(grid(x, y).pos - grid(x + xoffset[c], y + yoffset[c]).pos) * 1.15f;
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
	for (int y = 0; y < (GRIDSIZE - 1); y++) for (int x = 1; x < (GRIDSIZE - 2); x++)
	{
		const float2 p1 = grid(x, y).pos;
		const float2 p2 = grid(x + 1, y).pos;
		const float2 p3 = grid(x, y + 1).pos;
		screen->Line(p1.x, p1.y, p2.x, p2.y, 0xffffff);
		screen->Line(p1.x, p1.y, p3.x, p3.y, 0xffffff);
	}
	for (int y = 0; y < (GRIDSIZE - 1); y++)
	{
		const float2 p1 = grid(GRIDSIZE - 2, y).pos;
		const float2 p2 = grid(GRIDSIZE - 2, y + 1).pos;
		screen->Line(p1.x, p1.y, p2.x, p2.y, 0xffffff);
	}
}

// cloth simulation
// This function implements Verlet integration (see notes at top of file).
// Important: when constraints are applied, typically two points are
// drawn together to restore the rest length. When running on the GPU or
// when using SIMD, this will only work if the two vertices are not
// operated upon simultaneously (in a vector register, or in a warp).
float magic = 0.11f;
void Game::Simulation()
{
	clEnqueueWriteBuffer(queue_render, *(buffer->GetDevicePtr()), CL_TRUE, 0, GRIDSIZE * GRIDSIZE * sizeof(Point), pointGrid, 0, nullptr, nullptr);

	kernel_constraints->SetArguments(buffer);

	// simulation is exected three times per frame; do not change this.
	for (int steps = 0; steps < 3; steps++)
	{
		for (int y = 0; y < GRIDSIZE; y++)
			for (int x = 0; x < GRIDSIZE; x++)
			{
				rands[x + y * GRIDSIZE].rand10 = Rand(10);
				rands[x + y * GRIDSIZE].rand002 = Rand(0.02f + magic);
				rands[x + y * GRIDSIZE].rand012 = Rand(0.12f);
			}

		clEnqueueWriteBuffer(queue_render, *(randsBuffer->GetDevicePtr()), CL_TRUE, 0, GRIDSIZE * GRIDSIZE * sizeof(Rands), rands, 0, nullptr, nullptr);
		kernel_render->SetArguments(buffer, magic, randsBuffer);
		kernel_render->Run(GRIDSIZE * GRIDSIZE);

		buffer->CopyFromDevice();

		magic += 0.0002f; // slowly increases the chance of anomalies

		// apply constraints; 4 simulation steps: do not change this number.
		for (int i = 0; i < 4; i++)
		{
			clEnqueueWriteBuffer(queue_constraints, *(buffer->GetDevicePtr()), CL_TRUE, 0, GRIDSIZE * GRIDSIZE * sizeof(Point), pointGrid, 0, nullptr, nullptr);
			
			kernel_constraints->SetArguments(buffer, 0);
			kernel_constraints->Run(GRIDSIZE * GRIDSIZE);	

			kernel_constraints->SetArguments(buffer, 1);
			kernel_constraints->Run(GRIDSIZE * GRIDSIZE);

			kernel_constraints->SetArguments(buffer, 2);
			kernel_constraints->Run(GRIDSIZE * GRIDSIZE);

			kernel_constraints->SetArguments(buffer, 3);
			kernel_constraints->Run(GRIDSIZE * GRIDSIZE);

			buffer->CopyFromDevice();

			//buffer->CopyFromDevice();
			//for (int y = 1; y < GRIDSIZE - 1; y++)
			//	for (int x = 1; x < GRIDSIZE - 1; x++)
			//	{
			//		float2 pointpos = grid(x, y).pos;
			//		// use springs to four neighbouring points
			//		for (int linknr = 0; linknr < 4; linknr++)
			//		{
			//			Point& neighbour = grid(x + xoffset[linknr], y + yoffset[linknr]);
			//			float distance = length(neighbour.pos - pointpos);
			//			if (!isfinite(distance))
			//			{
			//				// warning: this happens; sometimes vertex positions 'explode'.
			//				continue;
			//			}
			//			if (distance > grid(x, y).restlength[linknr])
			//			{
			//				// pull points together
			//				float extra = distance / (grid(x, y).restlength[linknr]) - 1;
			//				float2 dir = neighbour.pos - pointpos;
			//				pointpos += extra * dir * 0.5f;
			//				neighbour.pos -= extra * dir * 0.5f;
			//			}
			//		}
			//		grid(x, y).pos = pointpos;
			//	}
			// fixed line of points is fixed.

			//for (int x = 0; x < GRIDSIZE; x++)
			//	grid(x, 0).pos = grid(x, 0).fix;
		}
	}
	buffer->CopyFromDevice();
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
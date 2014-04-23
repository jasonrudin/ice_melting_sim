/*
  FLUIDS v.1 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2008. Rama Hoetzlein, http://www.rchoetzlein.com

  ZLib license
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/



#include <conio.h>

#ifdef _MSC_VER
	#include <gl/glut.h>
#else
	#include <GL/glut.h>
#endif

#include "common_defs.h"
#include "mtime.h"
#include "fluid_system.h"
#include "../definitions.h"

#ifdef BUILD_CUDA
	#include "fluid_system_host.cuh"
#endif

#define EPSILON			0.00001f			//for collision detection



FluidSystem::FluidSystem ()
{
}

void FluidSystem::Initialize ( int mode, int total )
{
	if ( mode != BFLUID ) {
		printf ( "ERROR: FluidSystem not initialized as BFLUID.\n");
	}
	PointSet::Initialize ( mode, total );
	
	FreeBuffers ();
	AddBuffer ( BFLUID, sizeof ( Fluid ), total );
	AddAttribute ( 0, "pos", sizeof ( Vector3DF ), false );	
	AddAttribute ( 0, "color", sizeof ( DWORD ), false );
	AddAttribute ( 0, "vel", sizeof ( Vector3DF ), false );
	AddAttribute ( 0, "ndx", sizeof ( unsigned short ), false );
	AddAttribute ( 0, "age", sizeof ( unsigned short ), false );

	AddAttribute ( 0, "pressure", sizeof ( double ), false );
	AddAttribute ( 0, "density", sizeof ( double ), false );
	AddAttribute ( 0, "sph_force", sizeof ( Vector3DF ), false );
	AddAttribute ( 0, "next", sizeof ( Fluid* ), false );
	AddAttribute ( 0, "tag", sizeof ( bool ), false );	

	AddAttribute ( 0, "total_surrounding_heat", sizeof ( float ), false );
	AddAttribute ( 0, "temp", sizeof ( float ), false );
	AddAttribute ( 0, "state", sizeof ( enum State ), false );
	AddAttribute ( 0, "mass", sizeof ( float ), false );
	AddAttribute(0, "index", sizeof(Vector3DI), false);
		
	SPH_Setup ();
	Reset ( total );	
}

void FluidSystem::Reset ( int nmax )
{
	ResetBuffer ( 0, nmax );

	m_DT = 0.003; //  0.001;			// .001 = for point grav

	// Reset parameters
	m_Param [ MAX_FRAC ] = 1.0;
	m_Param [ POINT_GRAV ] = 0.0;
	m_Param [ PLANE_GRAV ] = 1.0;

	m_Param [ BOUND_ZMIN_SLOPE ] = 0.0;
	m_Param [ FORCE_XMAX_SIN ] = 0.0;
	m_Param [ FORCE_XMIN_SIN ] = 0.0;	
	m_Toggle [ WRAP_X ] = false;
	m_Toggle [ WALL_BARRIER ] = false;
	m_Toggle [ LEVY_BARRIER ] = false;
	m_Toggle [ DRAIN_BARRIER ] = false;
	m_Param [ SPH_INTSTIFF ] = 1.00;
	m_Param [ SPH_VISC ] = 0.2;
	m_Param [ SPH_INTSTIFF ] = 0.50;
	m_Param [ SPH_EXTSTIFF ] = 20000;
	m_Param [ SPH_SMOOTHRADIUS ] = 0.0043; //.001
	
	m_Vec [ POINT_GRAV_POS ].Set ( 0, 0, 50 );
	m_Vec [ PLANE_GRAV_DIR ].Set ( 0, 0, -9.8 );
	m_Vec [ EMIT_POS ].Set ( 0, 0, 0 );
	m_Vec [ EMIT_RATE ].Set ( 0, 0, 0 );
	m_Vec [ EMIT_ANG ].Set ( 0, 90, 1.0 );
	m_Vec [ EMIT_DANG ].Set ( 0, 0, 0 );
}

int count = 0;

int FluidSystem::AddPoint ()
{
	xref ndx;	
	Fluid* f = (Fluid*) AddElem ( 0, ndx );	
	f->sph_force.Set(0,0,0);
	f->vel.Set(0,0,0);
	f->vel_eval.Set(0,0,0);
	f->next = 0x0;
	f->pressure = 0;
	f->density = 0;
	f->temp = 20;
	f->total_surrounding_heat = 0;
	f->state = ICE;
	f->mass = MASS;
	return ndx;
}



int FluidSystem::AddPointReuse ()
{
	xref ndx;	
	Fluid* f;
	if ( NumPoints() <= mBuf[0].max-2 )
		f = (Fluid*) AddElem ( 0, ndx );
	else
		f = (Fluid*) RandomElem ( 0, ndx );

	f->sph_force.Set(0,0,0);
	f->vel.Set(0,0,0);
	f->vel_eval.Set(0,0,0);
	f->next = 0x0;
	f->pressure = 0;
	f->density = 0;
	f->temp = 20;
	f->total_surrounding_heat = 0;
	f->state = ICE;
	f->mass = MASS;
	return ndx;
}

void FluidSystem::Run ()
{
	bool bTiming = true;

	mint::Time start, stop;
	
	float ss = m_Param [ SPH_PDIST ] / m_Param[ SPH_SIMSCALE ];		// simulation scale (not Schutzstaffel)

	if ( m_Vec[EMIT_RATE].x > 0 && (++m_Frame) % (int) m_Vec[EMIT_RATE].x == 0 ) {
		//m_Frame = 0;
		Emit ( ss ); 
	}
	
	#ifdef NOGRID
		// Slow method - O(n^2)
		SPH_ComputePressureSlow ();
		SPH_ComputeForceSlow ();
	#else

		if ( m_Toggle[USE_CUDA] ) {
			
			#ifdef BUILD_CUDA
				// -- GPU --
				start.SetSystemTime ( ACC_NSEC );		
				TransferToCUDA ( mBuf[0].data, (int*) &m_Grid[0], NumPoints() );
				if ( bTiming) { stop.SetSystemTime ( ACC_NSEC ); stop = stop - start; printf ( "TO: %s\n", stop.GetReadableTime().c_str() ); }
			
				start.SetSystemTime ( ACC_NSEC );		
				Grid_InsertParticlesCUDA ();
				if ( bTiming) { stop.SetSystemTime ( ACC_NSEC ); stop = stop - start; printf ( "INSERT (CUDA): %s\n", stop.GetReadableTime().c_str() ); }

				start.SetSystemTime ( ACC_NSEC );
				SPH_ComputePressureCUDA ();
				if ( bTiming) { stop.SetSystemTime ( ACC_NSEC ); stop = stop - start; printf ( "PRESS (CUDA): %s\n", stop.GetReadableTime().c_str() ); }

				start.SetSystemTime ( ACC_NSEC );
				SPH_ComputeForceCUDA (); 
				if ( bTiming) { stop.SetSystemTime ( ACC_NSEC ); stop = stop - start; printf ( "FORCE (CUDA): %s\n", stop.GetReadableTime().c_str() ); }

				//** CUDA integrator is incomplete..
				// Once integrator is done, we can remove TransferTo/From steps
				/*start.SetSystemTime ( ACC_NSEC );
				SPH_AdvanceCUDA( m_DT, m_DT/m_Param[SPH_SIMSCALE] );
				if ( bTiming) { stop.SetSystemTime ( ACC_NSEC ); stop = stop - start; printf ( "ADV (CUDA): %s\n", stop.GetReadableTime().c_str() ); }*/

				start.SetSystemTime ( ACC_NSEC );		
				TransferFromCUDA ( mBuf[0].data, (int*) &m_Grid[0], NumPoints() );
				if ( bTiming) { stop.SetSystemTime ( ACC_NSEC ); stop = stop - start; printf ( "FROM: %s\n", stop.GetReadableTime().c_str() ); }

				// .. Do advance on CPU 
				Advance();

			#endif
			
		} else {
			// -- CPU only --

			start.SetSystemTime ( ACC_NSEC );
			Grid_InsertParticles ();
			if ( bTiming) { stop.SetSystemTime ( ACC_NSEC ); stop = stop - start; printf ( "INSERT: %s\n", stop.GetReadableTime().c_str() ); }
		
			start.SetSystemTime ( ACC_NSEC );
			SPH_ComputePressureGrid ();
			if ( bTiming) { stop.SetSystemTime ( ACC_NSEC ); stop = stop - start; printf ( "PRESS: %s\n", stop.GetReadableTime().c_str() ); }

			TemperatureAdvection();

			start.SetSystemTime ( ACC_NSEC );
			SPH_ComputeForceGridNC ();		
			if ( bTiming) { stop.SetSystemTime ( ACC_NSEC ); stop = stop - start; printf ( "FORCE: %s\n", stop.GetReadableTime().c_str() ); }

			start.SetSystemTime ( ACC_NSEC );
			Advance();
			if ( bTiming) { stop.SetSystemTime ( ACC_NSEC ); stop = stop - start; printf ( "ADV: %s\n", stop.GetReadableTime().c_str() ); }
		}		
		
	#endif
}



void FluidSystem::SPH_DrawDomain ()
{
	Vector3DF min, max;
	min = m_Vec[SPH_VOLMIN];
	max = m_Vec[SPH_VOLMAX];
	min.z += 0.5;

	glColor3f ( 0.0, 0.0, 1.0 );
	glBegin ( GL_LINES );
	glVertex3f ( min.x, min.y, min.z );	glVertex3f ( max.x, min.y, min.z );
	glVertex3f ( min.x, max.y, min.z );	glVertex3f ( max.x, max.y, min.z );
	glVertex3f ( min.x, min.y, min.z );	glVertex3f ( min.x, max.y, min.z );
	glVertex3f ( max.x, min.y, min.z );	glVertex3f ( max.x, max.y, min.z );
	glEnd ();
}


void FluidSystem::Advance ()
{
	char *dat1, *dat1_end;
	Fluid* p;
	Vector3DF norm, z;
	Vector3DF dir, accel;
	Vector3DF vnext;
	Vector3DF min, max;
	double adj;
	float SL, SL2, ss, radius;
	float stiff, damp, speed, diff; 
	SL = m_Param[SPH_LIMIT];
	SL2 = SL*SL;
	
	stiff = m_Param[SPH_EXTSTIFF];
	damp = m_Param[SPH_EXTDAMP];
	radius = m_Param[SPH_PRADIUS];
	min = m_Vec[SPH_VOLMIN];
	max = m_Vec[SPH_VOLMAX];
	ss = m_Param[SPH_SIMSCALE];

	dat1_end = mBuf[0].data + NumPoints()*mBuf[0].stride;
	for ( dat1 = mBuf[0].data; dat1 < dat1_end; dat1 += mBuf[0].stride ) {
		p = (Fluid*) dat1;		

		// Compute Acceleration		
		accel = p->sph_force;
		accel *= m_Param[SPH_PMASS];

		// Velocity limiting 
		speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
		if ( speed > SL2 ) {
			accel *= SL / sqrt(speed);
		}		
	
		// Boundary Conditions

		// Z-axis walls
		diff = 2 * radius - ( p->pos.z - min.z - (p->pos.x - m_Vec[SPH_VOLMIN].x) * m_Param[BOUND_ZMIN_SLOPE] )*ss;
		if (diff > EPSILON && p->state == WATER ) {			
			norm.Set ( -m_Param[BOUND_ZMIN_SLOPE], 0, 1.0 - m_Param[BOUND_ZMIN_SLOPE] );
			adj = stiff * diff - damp * norm.Dot ( p->vel_eval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}		

		diff = 2 * radius - ( max.z - p->pos.z )*ss;
		if (diff > EPSILON) {
			norm.Set ( 0, 0, -1 );
			adj = stiff * diff - damp * norm.Dot ( p->vel_eval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}
		
		// X-axis walls
		if ( !m_Toggle[WRAP_X] ) {
			diff = 2 * radius - ( p->pos.x - min.x + (sin(m_Time*10.0)-1+(p->pos.y*0.025)*0.25) * m_Param[FORCE_XMIN_SIN] )*ss;	
			//diff = 2 * radius - ( p->pos.x - min.x + (sin(m_Time*10.0)-1) * m_Param[FORCE_XMIN_SIN] )*ss;	
			if (diff > EPSILON ) {
				norm.Set ( 1.0, 0, 0 );
				adj = (m_Param[ FORCE_XMIN_SIN ] + 1) * stiff * diff - damp * norm.Dot ( p->vel_eval ) ;
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}

			diff = 2 * radius - ( max.x - p->pos.x + (sin(m_Time*10.0)-1) * m_Param[FORCE_XMAX_SIN] )*ss;	
			if (diff > EPSILON) {
				norm.Set ( -1, 0, 0 );
				adj = (m_Param[ FORCE_XMAX_SIN ]+1) * stiff * diff - damp * norm.Dot ( p->vel_eval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}
		}

		// Y-axis walls
		diff = 2 * radius - ( p->pos.y - min.y )*ss;			
		if (diff > EPSILON) {
			norm.Set ( 0, 1, 0 );
			adj = stiff * diff - damp * norm.Dot ( p->vel_eval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}
		diff = 2 * radius - ( max.y - p->pos.y )*ss;
		if (diff > EPSILON) {
			norm.Set ( 0, -1, 0 );
			adj = stiff * diff - damp * norm.Dot ( p->vel_eval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}

		// Wall barrier
		if ( m_Toggle[WALL_BARRIER] ) {
			diff = 2 * radius - ( p->pos.x - 0 )*ss;					
			if (diff < 2*radius && diff > EPSILON && fabs(p->pos.y) < 3 && p->pos.z < 10) {
				norm.Set ( 1.0, 0, 0 );
				adj = 2*stiff * diff - damp * norm.Dot ( p->vel_eval ) ;	
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}
		}
		
		// Levy barrier
		if ( m_Toggle[LEVY_BARRIER] ) {
			diff = 2 * radius - ( p->pos.x - 0 )*ss;					
			if (diff < 2*radius && diff > EPSILON && fabs(p->pos.y) > 5 && p->pos.z < 10) {
				norm.Set ( 1.0, 0, 0 );
				adj = 2*stiff * diff - damp * norm.Dot ( p->vel_eval ) ;	
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}
		}
		// Drain barrier
		if ( m_Toggle[DRAIN_BARRIER] ) {
			diff = 2 * radius - ( p->pos.z - min.z-15 )*ss;
			if (diff < 2*radius && diff > EPSILON && (fabs(p->pos.x)>3 || fabs(p->pos.y)>3) ) {
				norm.Set ( 0, 0, 1);
				adj = stiff * diff - damp * norm.Dot ( p->vel_eval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}
		}

		// Plane gravity
		if ( m_Param[PLANE_GRAV] > 0) 
			accel += m_Vec[PLANE_GRAV_DIR];

		// Point gravity
		if ( m_Param[POINT_GRAV] > 0 ) {
			norm.x = ( p->pos.x - m_Vec[POINT_GRAV_POS].x );
			norm.y = ( p->pos.y - m_Vec[POINT_GRAV_POS].y );
			norm.z = ( p->pos.z - m_Vec[POINT_GRAV_POS].z );
			norm.Normalize ();
			norm *= m_Param[POINT_GRAV];
			accel -= norm;
		}

		// Leapfrog Integration ----------------------------
		vnext = accel;							
		vnext *= m_DT;
		vnext += p->vel;						// v(t+1/2) = v(t-1/2) + a(t) dt
		p->vel_eval = p->vel;
		p->vel_eval += vnext;
		p->vel_eval *= 0.5;					// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5		used to compute forces later
		p->vel = vnext;
		vnext *= m_DT/ss;
		p->pos += vnext;						// p(t+1) = p(t) + v(t+1/2) dt

		//update temperature
		p->temp = p->temp + p->total_surrounding_heat * m_DT;
		p->total_surrounding_heat = 0;


		if(p->temp > 32 && p->state == ICE){
			p->state = WATER;
			AdjustNeighbors(p);
		}

		//The default color mode will display color that represents the temperature of the particles
		if(m_Param[CLR_MODE] == 0){
			float point_temp = p->temp;
			p->clr = COLORA(point_temp/100,point_temp/100,point_temp/100,1);

						// Color according to temperature
			float v = p->temp;
            float dv = MAX_TEMPERATURE - MIN_TEMPERATURE;
            float rgba[4] = {1.0f, 1.0f, 1.0f, 1.0f};
		    //float rgba[4] = {0.0f, 0.0f, 0.0f, 1.0f};
         
			if (v < MIN_TEMPERATURE) v = MIN_TEMPERATURE;
            if (v > MAX_TEMPERATURE) v = MAX_TEMPERATURE;

            if (v < (MIN_TEMPERATURE + 0.25 * dv)) {
                rgba[0] = 0.0;
                rgba[1] = 4 * (v - MIN_TEMPERATURE) / dv;
            } else if (v < (MIN_TEMPERATURE + 0.5 * dv)) {
                rgba[0] = 0.0;
                rgba[2] = 1.0 + 4.0 * (MIN_TEMPERATURE + 0.25 * dv - v) / dv;
            } else if (v < (MIN_TEMPERATURE + 0.75 * dv)) {
                rgba[0] = 4.0 * (v - MIN_TEMPERATURE - 0.5 * dv) / dv;
                rgba[2] = 0.0;
            } else {
                rgba[1] = 1.0 + 4.0 * (MIN_TEMPERATURE + 0.75 * dv - v) / dv;
                rgba[2] = 0.0;
            }

            p->clr = COLORA(rgba[0], rgba[1], rgba[2], rgba[3]);
		}
		if ( m_Param[CLR_MODE]==1.0 ) {
			adj = fabs(vnext.x)+fabs(vnext.y)+fabs(vnext.z) / 7000.0;
			adj = (adj > 1.0) ? 1.0 : adj;
			p->clr = COLORA( 0, adj, adj, 1 );
		}
		if ( m_Param[CLR_MODE]==2.0 ) {
			float v = 0.5 + ( p->pressure / 1500.0); 
			if ( v < 0.1 ) v = 0.1;
			if ( v > 1.0 ) v = 1.0;
			p->clr = COLORA ( v, 1-v, 0, 1 );
		}


		// Euler integration -------------------------------
		/* accel += m_Gravity;
		accel *= m_DT;
		p->vel += accel;				// v(t+1) = v(t) + a(t) dt
		p->vel_eval += accel;
		p->vel_eval *= m_DT/d;
		p->pos += p->vel_eval;
		p->vel_eval = p->vel;  */	


		if ( m_Toggle[WRAP_X] ) {
			diff = p->pos.x - (m_Vec[SPH_VOLMIN].x + 2);			// -- Simulates object in center of flow
			if ( diff <= 0 ) {
				p->pos.x = (m_Vec[SPH_VOLMAX].x - 2) + diff*2;				
				p->pos.z = 10;
			}
		}	
	}

	m_Time += m_DT;
}



//------------------------------------------------------ SPH Setup 
//
//  Range = +/- 10.0 * 0.006 (r) =	   0.12			m (= 120 mm = 4.7 inch)
//  Container Volume (Vc) =			   0.001728		m^3
//  Rest Density (D) =				1000.0			kg / m^3
//  Particle Mass (Pm) =			   0.00020543	kg						(mass = vol * density)
//  Number of Particles (N) =		4000.0
//  Water Mass (M) =				   0.821		kg (= 821 grams)
//  Water Volume (V) =				   0.000821     m^3 (= 3.4 cups, .21 gals)
//  Smoothing Radius (R) =             0.02			m (= 20 mm = ~3/4 inch)
//  Particle Radius (Pr) =			   0.00366		m (= 4 mm  = ~1/8 inch)
//  Particle Volume (Pv) =			   2.054e-7		m^3	(= .268 milliliters)
//  Rest Distance (Pd) =			   0.0059		m
//
//  Given: D, Pm, N
//    Pv = Pm / D			0.00020543 kg / 1000 kg/m^3 = 2.054e-7 m^3	
//    Pv = 4/3*pi*Pr^3    cuberoot( 2.054e-7 m^3 * 3/(4pi) ) = 0.00366 m
//     M = Pm * N			0.00020543 kg * 4000.0 = 0.821 kg		
//     V =  M / D              0.821 kg / 1000 kg/m^3 = 0.000821 m^3
//     V = Pv * N			 2.054e-7 m^3 * 4000 = 0.000821 m^3
//    Pd = cuberoot(Pm/D)    cuberoot(0.00020543/1000) = 0.0059 m 
//
// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
// Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
//    (k = number of cells, gs = cell size, d = simulation scale)

void FluidSystem::SPH_Setup ()
{
	m_Param [ SPH_SIMSCALE ] =		0.004;			// unit size
	m_Param [ SPH_VISC ] =			0.2;			// pascal-second (Pa.s) = 1 kg m^-1 s^-1  (see wikipedia page on viscosity)
	m_Param [ SPH_RESTDENSITY ] =	600.0;			// kg / m^3
	m_Param [ SPH_PMASS ] =			0.00020543;		// kg
	m_Param [ SPH_PRADIUS ] =		0.002;			//.004 m
	m_Param [ SPH_PDIST ] =			0.0059;			// m
	m_Param [ SPH_SMOOTHRADIUS ] =	0.01;			// m 
	m_Param [ SPH_INTSTIFF ] =		.5;              // 1.00;
	m_Param [ SPH_EXTSTIFF ] =		 15000; 
	m_Param [ SPH_EXTDAMP ] =		256.0;
	m_Param [ SPH_LIMIT ] =			200.0;			// m / s

	m_Toggle [ SPH_GRID ] =		false;
	m_Toggle [ SPH_DEBUG ] =	false;

	SPH_ComputeKernels ();
}

void FluidSystem::SPH_ComputeKernels ()
{
	m_Param [ SPH_PDIST ] = pow ( m_Param[SPH_PMASS] / m_Param[SPH_RESTDENSITY], 1/3.0 );
	m_R2 = m_Param [SPH_SMOOTHRADIUS] * m_Param[SPH_SMOOTHRADIUS];
	m_Poly6Kern = 315.0f / (64.0f * 3.141592 * pow( m_Param[SPH_SMOOTHRADIUS], 9) );	// Wpoly6 kernel (denominator part) - 2003 Muller, p.4
	m_SpikyKern = -45.0f / (3.141592 * pow( m_Param[SPH_SMOOTHRADIUS], 6) );			// Laplacian of viscocity (denominator): PI h^6
	m_LapKern = 45.0f / (3.141592 * pow( m_Param[SPH_SMOOTHRADIUS], 6) );
}

void FluidSystem::SPH_CreateExample ( int n, int nmax )
{
	Vector3DF pos;
	Vector3DF min, max;
	
	Reset ( nmax );
	
	switch ( n ) {

	case 0:		// Square Ice

		m_Vec [ SPH_VOLMIN ].Set ( -15, -15, 0 );
		m_Vec [ SPH_VOLMAX ].Set ( 40, 40, 40 );
		m_Vec [ SPH_INITMIN ].Set ( 0, 0, 5 );
		m_Vec [ SPH_INITMAX ].Set ( 35, 35, 35 );
		m_Vec [ PLANE_GRAV_DIR ].Set ( 0.0, 0, -9.8 );
		voxelGrid = new VoxelGrid("Objects/cube_20.voxels");
		break;

	case 1:		// Dragon Ice

		m_Vec [ SPH_VOLMIN ].Set ( -15, -15, 0 );
		m_Vec [ SPH_VOLMAX ].Set ( 40, 40, 40 );
		m_Vec [ SPH_INITMIN ].Set ( 0, 0, 5 );
		m_Vec [ SPH_INITMAX ].Set ( 35, 35, 35 );;
		voxelGrid = new VoxelGrid("Objects/dragon.voxels");
		break;
	}


	nmax = voxelGrid->theDim[0] * voxelGrid->theDim[1] * voxelGrid->theDim[2];
	Reset(nmax);
	SetNeighbors();

	//SPH_ComputeKernels ();

	m_Param [ SPH_SIMSIZE ] = m_Param [ SPH_SIMSCALE ] * (m_Vec[SPH_VOLMAX].z - m_Vec[SPH_VOLMIN].z);
	m_Param [ SPH_PDIST ] = pow ( m_Param[SPH_PMASS] / m_Param[SPH_RESTDENSITY], 1/3.0 );	

	float ss = m_Param [ SPH_PDIST ]*0.87 / m_Param[ SPH_SIMSCALE ];	
	printf ( "Spacing: %f\n", ss);
	//AddVolume ( m_Vec[SPH_INITMIN], m_Vec[SPH_INITMAX], ss );	// Create the particles
	AddVolume ( m_Vec[SPH_INITMIN], m_Vec[SPH_INITMAX], voxelGrid->voxelSize[0], voxelGrid); //create the particles (WHEN LOADING IN)

	float cell_size = m_Param[SPH_SMOOTHRADIUS]*2.0;			// Grid cell size (2r)	
	Grid_Setup ( m_Vec[SPH_VOLMIN], m_Vec[SPH_VOLMAX], m_Param[SPH_SIMSCALE], cell_size, 1.0 );												// Setup grid
	Grid_InsertParticles ();									// Insert particles

	Vector3DF vmin, vmax;
	vmin =  m_Vec[SPH_VOLMIN];
	vmin -= Vector3DF(2,2,2);
	vmax =  m_Vec[SPH_VOLMAX];
	vmax += Vector3DF(2,2,-2);

	#ifdef BUILD_CUDA
		FluidClearCUDA ();
		Sleep ( 500 );

		FluidSetupCUDA ( NumPoints(), sizeof(Fluid), *(float3*)& m_GridMin, *(float3*)& m_GridMax, *(float3*)& m_GridRes, *(float3*)& m_GridSize, (int) m_Vec[EMIT_RATE].x );

		Sleep ( 500 );

		FluidParamCUDA ( m_Param[SPH_SIMSCALE], m_Param[SPH_SMOOTHRADIUS], m_Param[SPH_PMASS], m_Param[SPH_RESTDENSITY], m_Param[SPH_INTSTIFF], m_Param[SPH_VISC] );
	#endif

}

// Compute Pressures - Very slow yet simple. O(n^2)
void FluidSystem::SPH_ComputePressureSlow ()
{
	char *dat1, *dat1_end;
	char *dat2, *dat2_end;
	Fluid *p, *q;
	int cnt = 0;
	double dx, dy, dz, sum, dsq, c;
	double d, d2, mR, mR2;
	d = m_Param[SPH_SIMSCALE];
	d2 = d*d;
	mR = m_Param[SPH_SMOOTHRADIUS];
	mR2 = mR*mR;	

	dat1_end = mBuf[0].data + NumPoints()*mBuf[0].stride;
	for ( dat1 = mBuf[0].data; dat1 < dat1_end; dat1 += mBuf[0].stride ) {
		p = (Fluid*) dat1;

		sum = 0.0;
		cnt = 0;
		
		dat2_end = mBuf[0].data + NumPoints()*mBuf[0].stride;
		for ( dat2 = mBuf[0].data; dat2 < dat2_end; dat2 += mBuf[0].stride ) {
			q = (Fluid*) dat2;

			if ( p==q ) continue;
			dx = ( p->pos.x - q->pos.x)*d;		// dist in cm
			dy = ( p->pos.y - q->pos.y)*d;
			dz = ( p->pos.z - q->pos.z)*d;
			dsq = (dx*dx + dy*dy + dz*dz);
			if ( mR2 > dsq ) {
				c =  m_R2 - dsq;
				sum += c * c * c;
				cnt++;
				//if ( p == m_CurrP ) q->tag = true;
			}
		}	
		p->density = sum * m_Param[SPH_PMASS] * m_Poly6Kern ;	
		p->pressure = ( p->density - m_Param[SPH_RESTDENSITY] ) * m_Param[SPH_INTSTIFF];
		p->density = 1.0f / p->density;
	}
}


// Compute Pressures - Using spatial grid, and also create neighbor table
void FluidSystem::SPH_ComputePressureGrid ()
{
	char *dat1, *dat1_end;
	Fluid* p;
	Fluid* pcurr;
	int pndx;
	int i, cnt = 0;
	float dx, dy, dz, sum, dsq, c;
	float d, d2, mR, mR2;
	float radius = m_Param[SPH_SMOOTHRADIUS] / m_Param[SPH_SIMSCALE];
	d = m_Param[SPH_SIMSCALE];
	d2 = d*d;
	mR = m_Param[SPH_SMOOTHRADIUS];
	mR2 = mR*mR;	

	dat1_end = mBuf[0].data + NumPoints()*mBuf[0].stride;
	i = 0;
	for ( dat1 = mBuf[0].data; dat1 < dat1_end; dat1 += mBuf[0].stride, i++ ) {
		p = (Fluid*) dat1;

		sum = 0.0;	
		m_NC[i] = 0;

		//find all of the cells within a radius r from the particle
		Grid_FindCells ( p->pos, radius );
		for (int cell=0; cell < 8; cell++) {	//for each potential neighboring cell
			if ( m_GridCell[cell] != -1 ) {		//if the cell is in range
				pndx = m_Grid [ m_GridCell[cell] ];			//find the index of the particle in the world grid (as opposed to the local grid)	
				while ( pndx != -1 ) {					
					pcurr = (Fluid*) (mBuf[0].data + pndx*mBuf[0].stride);	 //find the particle				
					if ( pcurr == p ) {pndx = pcurr->next; continue; }		 //ignores itself
					dx = ( p->pos.x - pcurr->pos.x)*d;		// dist in cm
					dy = ( p->pos.y - pcurr->pos.y)*d;
					dz = ( p->pos.z - pcurr->pos.z)*d;
					dsq = (dx*dx + dy*dy + dz*dz);
					if ( mR2 > dsq ) {
						c =  m_R2 - dsq;
						sum += c * c * c;
						if ( m_NC[i] < MAX_NEIGHBOR ) {
							m_Neighbor[i][ m_NC[i] ] = pndx;			//store the index of particle i in an array indexed by the particle and 
																		//the number neighbor it is
							m_NDist[i][ m_NC[i] ] = sqrt(dsq);			//store the distance between particle i and the current particle
							m_NC[i]++;									//increase the number of neighbors for particle i
						}
					}
					pndx = pcurr->next;
				}
			}
			m_GridCell[cell] = -1;
		}
		p->density = sum * m_Param[SPH_PMASS] * m_Poly6Kern ;
		if(p->state == WATER){
			p->pressure = ( p->density - m_Param[SPH_RESTDENSITY] ) * m_Param[SPH_INTSTIFF];
		}
		else{
			p->pressure = ( p->density - m_Param[SPH_RESTDENSITY] ) * .3;
		}
		p->density = 1.0f / p->density;		
	}
}

// Compute Forces - Very slow, but simple. O(n^2)

void FluidSystem::SPH_ComputeForceSlow ()
{
	char *dat1, *dat1_end;
	char *dat2, *dat2_end;
	Fluid *p, *q;
	Vector3DF force, fcurr;
	register double pterm, vterm, dterm;
	double c, r, d, sum, dsq;
	double dx, dy, dz;
	double mR, mR2, visc;

	d = m_Param[SPH_SIMSCALE];
	mR = m_Param[SPH_SMOOTHRADIUS];
	mR2 = (mR*mR);
	visc = m_Param[SPH_VISC];
	vterm = m_LapKern * visc;

	dat1_end = mBuf[0].data + NumPoints()*mBuf[0].stride;
	for ( dat1 = mBuf[0].data; dat1 < dat1_end; dat1 += mBuf[0].stride ) {
		p = (Fluid*) dat1;

		sum = 0.0;
		force.Set ( 0, 0, 0 );
		
		dat2_end = mBuf[0].data + NumPoints()*mBuf[0].stride;
		for ( dat2 = mBuf[0].data; dat2 < dat2_end; dat2 += mBuf[0].stride ) {
			q = (Fluid*) dat2;

			if ( p == q ) continue;
			dx = ( p->pos.x - q->pos.x )*d;			// dist in cm
			dy = ( p->pos.y - q->pos.y )*d;
			dz = ( p->pos.z - q->pos.z )*d;
			dsq = (dx*dx + dy*dy + dz*dz);
			if ( mR2 > dsq ) {
				r = sqrt ( dsq );
				c = (mR - r);
				pterm = -0.5f * c * m_SpikyKern * ( p->pressure + q->pressure) / r;
				dterm = c * p->density * q->density;
				force.x += ( pterm * dx + vterm * (q->vel_eval.x - p->vel_eval.x) ) * dterm;
				force.y += ( pterm * dy + vterm * (q->vel_eval.y - p->vel_eval.y) ) * dterm;
				force.z += ( pterm * dz + vterm * (q->vel_eval.z - p->vel_eval.z) ) * dterm;
			}
		}			
		p->sph_force = force;		
	}
}

// Compute Forces - Using spatial grid. Faster.
void FluidSystem::SPH_ComputeForceGrid ()
{
	char *dat1, *dat1_end;	
	Fluid *p;
	Fluid *pcurr;
	int pndx;
	Vector3DF force, fcurr;
	register double pterm, vterm, dterm;
	double c, d, dsq, r;
	double dx, dy, dz;
	double mR, mR2, visc;
	float radius = m_Param[SPH_SMOOTHRADIUS] / m_Param[SPH_SIMSCALE];

	d = m_Param[SPH_SIMSCALE];
	mR = m_Param[SPH_SMOOTHRADIUS];
	mR2 = (mR*mR);
	visc = m_Param[SPH_VISC];

	dat1_end = mBuf[0].data + NumPoints()*mBuf[0].stride;
	for ( dat1 = mBuf[0].data; dat1 < dat1_end; dat1 += mBuf[0].stride ) {
		p = (Fluid*) dat1;

		force.Set ( 0, 0, 0 );

		Grid_FindCells ( p->pos, radius );
		for (int cell=0; cell < 8; cell++) {
			if ( m_GridCell[cell] != -1 ) {
				pndx = m_Grid [ m_GridCell[cell] ];				
				while ( pndx != -1 ) {					
					pcurr = (Fluid*) (mBuf[0].data + pndx*mBuf[0].stride);					
					if ( pcurr == p ) {pndx = pcurr->next; continue; }
			
					dx = ( p->pos.x - pcurr->pos.x)*d;		// dist in cm
					dy = ( p->pos.y - pcurr->pos.y)*d;
					dz = ( p->pos.z - pcurr->pos.z)*d;
					dsq = (dx*dx + dy*dy + dz*dz);
					if ( mR2 > dsq ) {
						r = sqrt ( dsq );
						c = (mR - r);
						pterm = -0.5f * c * m_SpikyKern * ( p->pressure + pcurr->pressure) / r;
						dterm = c * p->density * pcurr->density;
						vterm =	m_LapKern * visc;
						force.x += ( pterm * dx + vterm * (pcurr->vel_eval.x - p->vel_eval.x) ) * dterm;
						force.y += ( pterm * dy + vterm * (pcurr->vel_eval.y - p->vel_eval.y) ) * dterm;
						force.z += ( pterm * dz + vterm * (pcurr->vel_eval.z - p->vel_eval.z) ) * dterm;
					}
					pndx = pcurr->next;
				}
			}
		}

		if(p->state = WATER){
			p->sph_force = force;
		}
		else{
			p->sph_force = 0;
		}
	}
}


// Compute Forces - Using spatial grid with saved neighbor table. Fastest.
void FluidSystem::SPH_ComputeForceGridNC ()
{
	char *dat1, *dat1_end;	
	Fluid *p;
	Fluid *pcurr;
	Vector3DF force, fcurr;
	register float pterm, vterm, dterm;
	int i;
	float c, d;
	float dx, dy, dz;
	float mR, mR2, visc;	

	d = m_Param[SPH_SIMSCALE];
	mR = m_Param[SPH_SMOOTHRADIUS];
	mR2 = (mR*mR);
	visc = m_Param[SPH_VISC];

	dat1_end = mBuf[0].data + NumPoints()*mBuf[0].stride;
	i = 0;

	//find the correct anti-gravity force
	bool touch_ground = adjustGravity();
	
	for ( dat1 = mBuf[0].data; dat1 < dat1_end; dat1 += mBuf[0].stride, i++ ) {
		p = (Fluid*) dat1;

		if (touch_ground && p->state == ICE) {
            force.Set(anti_gravity.x, anti_gravity.y, anti_gravity.z);
        } 
		else {
            force.Set (0, 0, 0);
        }

		for (int j=0; j < m_NC[i]; j++ ) {
			pcurr = (Fluid*) (mBuf[0].data + m_Neighbor[i][j]*mBuf[0].stride);
			dx = ( p->pos.x - pcurr->pos.x)*d;		// dist in cm
			dy = ( p->pos.y - pcurr->pos.y)*d;
			dz = ( p->pos.z - pcurr->pos.z)*d;				
			c = ( mR - m_NDist[i][j] );
			pterm = -0.5f * c * m_SpikyKern * ( p->pressure + pcurr->pressure) / m_NDist[i][j];
			dterm = c * p->density * pcurr->density;
			vterm = m_LapKern * visc;
			Vector3DF dist = p->pos;
            dist -= pcurr->pos;
            float length = dist.Length();

			if(p->state == WATER){
				force.x += ( pterm * dx + vterm * (pcurr->vel_eval.x - p->vel_eval.x) ) * dterm;
				force.y += ( pterm * dy + vterm * (pcurr->vel_eval.y - p->vel_eval.y) ) * dterm;
				force.z += ( pterm * dz + vterm * (pcurr->vel_eval.z - p->vel_eval.z) ) * dterm;

				if (pcurr->state == WATER) {
					force.x += K_WATER * dist.x;
					force.y += K_WATER * dist.y;
					force.z += K_WATER * dist.z;
				} 
				else {
					force.x += K_ICE * dist.x;
					force.y += K_ICE * dist.y;
					force.z += K_ICE * dist.z;
				}
			}
		}
		if(p->state == ICE){
		//	force -= m_Vec[PLANE_GRAV_DIR];
		//	force *= 1/m_Param[SPH_PMASS];
		}
		//p->sph_force = 0;
		p->sph_force = force;
	}
	}


void FluidSystem::TemperatureAdvection(){
	
char *dat1, *dat1_end;	
	Fluid *p;
	Fluid *pcurr;
	float SmoothingKernelFunction;
	float tempDiff, totalNeighborTemp;
	float heatValue, amountExposed, ambientTempEffect;
	int i;
	float c, d;
	float dx, dy, dz;
	float mR, mR2, visc;	

	d = m_Param[SPH_SIMSCALE];
	mR = m_Param[SPH_SMOOTHRADIUS];
	mR2 = (mR*mR);
	visc = m_Param[SPH_VISC];

	dat1_end = mBuf[0].data + NumPoints()*mBuf[0].stride;
	i = 0;
	
	
	//temperature change between particles
	for ( dat1 = mBuf[0].data; dat1 < dat1_end; dat1 += mBuf[0].stride, i++ ) {
		p = (Fluid*) dat1;
		totalNeighborTemp = 0;
		ambientTempEffect = 0;

		for (int j=0; j < m_NC[i]; j++ ) {
			pcurr = (Fluid*) (mBuf[0].data + m_Neighbor[i][j]*mBuf[0].stride);
			dx = ( p->pos.x - pcurr->pos.x)*d;		// dist in cm
			dy = ( p->pos.y - pcurr->pos.y)*d;
			dz = ( p->pos.z - pcurr->pos.z)*d;

			//calculate the difference in temperatures
			tempDiff = pcurr->temp - p->temp;
			if(abs(tempDiff) < .00001){
				tempDiff = 0;
			}

			Vector3DF dist = p->pos;
            dist -= pcurr->pos;
            double length = dist.Length();

			c = ( mR - m_NDist[i][j] );
			//SmoothingKernelFunction = -1.0 * c * m_SpikyKern;
			//SmoothingKernelFunction =  45.0f/(3.141592 * pow(PARTICLE_PRADIUS, 6)) * (PARTICLE_PRADIUS - m_NDist[i][j]);
			SmoothingKernelFunction =  45.0f/(3.141592 * pow(PARTICLE_PRADIUS, 6)) * (PARTICLE_PRADIUS - length);
			//totalNeighborTemp = totalNeighborTemp + pcurr->mass * (tempDiff / pcurr->density) * SmoothingKernelFunction;
			float tempTerm = ((pcurr->temp - p->temp)/pcurr->density);
			totalNeighborTemp = totalNeighborTemp + m_Param [ SPH_PMASS ] * tempTerm * SmoothingKernelFunction;

			if(abs(totalNeighborTemp) < .000001){
				totalNeighborTemp = 0;
			}
		}

		//multiply by relevant thermal diffusion constant
		if(p->state == WATER){
			totalNeighborTemp = totalNeighborTemp * WATER_THERMAL_DIFFUSION_CONSTANT;
		}
		else{
			totalNeighborTemp = totalNeighborTemp * ICE_THERMAL_DIFFUSION_CONSTANT;
		}

		//find temperature change due to surrounding air
		amountExposed = 1; //need to update this using a neighbor table and a grid
		amountExposed = (voxelGrid->voxelSize[0] * voxelGrid->voxelSize[0])*(6.0 - voxelGrid->adjacencyList[p->index.x][p->index.y][p->index.z]);
		heatValue = THERMAL_CONDUCTIVITY * (AMBIENT_TEMPERATURE - p->temp) * amountExposed;

		if(p->state == WATER){
			heatValue = THERMAL_CONDUCTIVITY * (AMBIENT_TEMPERATURE - p->temp);
			ambientTempEffect = heatValue / (SPECIFIC_HEAT_CAPACITY_WATER * MASS);
		}
		else{
			double radius = m_Param[SPH_PRADIUS];
			double ss = m_Param[SPH_SIMSCALE];
			Vector3DF min = m_Vec[SPH_VOLMIN];
			double diff = 2 * radius - ( p->pos.z - min.z - (p->pos.x - m_Vec[SPH_VOLMIN].x) * m_Param[BOUND_ZMIN_SLOPE] )*ss;
			amountExposed = (voxelGrid->voxelSize[0] * voxelGrid->voxelSize[0])*(6.0 - voxelGrid->adjacencyList[p->index.x][p->index.y][p->index.z]);
			heatValue = THERMAL_CONDUCTIVITY * (AMBIENT_TEMPERATURE - p->temp) * amountExposed;
			if(diff > EPSILON){
				heatValue += .05;
			}

			ambientTempEffect = heatValue / (SPECIFIC_HEAT_CAPACITY_ICE * MASS);
			//ambientTempEffect = 0;
		}

		if(totalNeighborTemp > 50){
			int abc = 2;
		}

		if(ambientTempEffect > 0){
			int abc = 2;
		}

		p->total_surrounding_heat = totalNeighborTemp + ambientTempEffect;

		

	}
}



void FluidSystem::SetNeighbors(){
	short neighbors;
    voxelGrid->adjacencyList[1][1][1] = 0;
    for (int i = 0; i < voxelGrid->theDim[0]; i++) {
        for (int j = 0; j < voxelGrid->theDim[2]; j++) {
            for (int k = 0; k < voxelGrid->theDim[1]; k++) {
                neighbors = 0;
                if (voxelGrid->data[i][j][k]) { // if there is a voxel in that location
                    if (i > 0 && voxelGrid->data[i-1][j][k]) neighbors++;
                    if (i < voxelGrid->theDim[0] - 1 && voxelGrid->data[i+1][j][k]) neighbors++;

                    if (j > 0 && voxelGrid->data[i][j-1][k]) neighbors++;
                    if (j < voxelGrid->theDim[2] - 1 && voxelGrid->data[i][j+1][k]) neighbors++;

                    if (k > 0 && voxelGrid->data[i][j][k-1]) neighbors++;
                    if (k < voxelGrid->theDim[1] - 1 && voxelGrid->data[i][j][k+1]) neighbors++;
                } else {
                    neighbors = -1; //error state
                }
                voxelGrid->adjacencyList[i][j][k] = neighbors;
            }
        }
    }


}

void FluidSystem::AdjustNeighbors(Fluid* p){
	int pi = p->index.x;
	int pj = p->index.y;
	int pk = p->index.z;


	voxelGrid->data[pi][pj][pk] = 0;
	voxelGrid->adjacencyList[pi][pj][pk] = -1;

    if (pi + 1 < voxelGrid->theDim[0]) voxelGrid->adjacencyList[pi+1][pj][pk]--;
    if (pi - 1 > 0) voxelGrid->adjacencyList[pi-1][pj][pk]--;
    if (pj + 1 < voxelGrid->theDim[2]) voxelGrid->adjacencyList[pi][pj+1][pk]--;
    if (pj - 1 > 0) voxelGrid->adjacencyList[pi][pj-1][pk]--;
    if (pk + 1 < voxelGrid->theDim[1]) voxelGrid->adjacencyList[pi][pj][pk+1]--;
    if (pk - 1 > 0) voxelGrid->adjacencyList[pi][pj][pk-1]--;

	p->state = WATER;
}

void FluidSystem::AddVolume ( Vector3DF min, Vector3DF max, float spacing, VoxelGrid* vgrid)
{
	Vector3DF pos;
	Fluid* p;
	float dx, dy, dz;
	dx = max.x-min.x;
	dy = max.y-min.y;
	dz = max.z-min.z;

	// temp counter
	int count = 0;
	for (float z = min.z; z <= max.z; z += spacing ) {
		for (float y = min.y; y <= max.y; y += spacing ) {	
			for (float x = min.x; x <= max.x; x += spacing ) {
                Vector3DF index = vgrid->inVoxelGrid(x,y,z);
				if(index.x >= 0 && index.y >= 0 && index.z >= 0){
					count++;
					p = (Fluid*)GetPoint ( AddPointReuse () );
					pos.Set ( x, y, z);
                    p->index = index;
					p->pos = pos;
					p->clr = COLORA( (x-min.x)/dx, (y-min.y)/dy, (z-min.z)/dz, 1);
				}
			}
		}
	}
}

//Determines the anti-gravity force to apply to particles that are on the ground
bool FluidSystem::adjustGravity(){
	Fluid *p;
	char *dat1_end, *dat2;
	dat1_end = mBuf[0].data + NumPoints()*mBuf[0].stride;

	// Checking the boundary
    double diff, adj;
	double stiff = m_Param[SPH_EXTSTIFF];
	double damp = m_Param[SPH_EXTDAMP];
	double radius = m_Param[SPH_PRADIUS];
    double ss = m_Param[SPH_SIMSCALE];

    bool touch_ground = false;
    //Vector3DF anti_gravity;
    Vector3DF norm;
	Vector3DF min = m_Vec[SPH_VOLMIN];
	Vector3DF max = m_Vec[SPH_VOLMAX];

	

	for( dat2 = mBuf[0].data; dat2 < dat1_end; dat2 += mBuf[0].stride) {
    	// Z-axis walls
        p = (Fluid*) dat2;
		diff = 2 * radius - ( p->pos.z - min.z - (p->pos.x - m_Vec[SPH_VOLMIN].x) * m_Param[BOUND_ZMIN_SLOPE] )*ss;
		if (diff > EPSILON && p->state == 0) {	
			//sets the normal direction
			norm.Set ( -m_Param[BOUND_ZMIN_SLOPE], 0, 1.0 - m_Param[BOUND_ZMIN_SLOPE] );

			//amount of "rebound"
			adj = stiff * diff - damp * norm.Dot ( p->vel_eval );
            anti_gravity = norm;
            anti_gravity *= adj;
            anti_gravity /= m_Param[SPH_PMASS];
            touch_ground = true;
		}
    }

	return touch_ground;
}

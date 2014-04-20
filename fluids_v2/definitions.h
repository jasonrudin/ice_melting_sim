#ifndef DEFINITIONS
#define DEFINITIONS


// State of particle

enum State { ICE, WATER};

static float ICE_THERMAL_DIFFUSION_CONSTANT = .5;
static float WATER_THERMAL_DIFFUSION_CONSTANT = .1;
static float MASS = 0.0008;
static float THERMAL_CONDUCTIVITY = 0.00267;
static float AMBIENT_TEMPERATURE = 40;
static float FLOOR_TEMPERATURE = 65;
static float SPECIFIC_HEAT_CAPACITY_ICE = 2.11;//1.0; //  // units: kJ/kg-K
static float SPECIFIC_HEAT_CAPACITY_WATER = 4.181;//1.0; // // units: kJ/kg-K
static float MAX_TEMPERATURE = 70;
static float MIN_TEMPERATURE = 20;

static float PARTICLE_PRADIUS = 1.1;

// Parameter for the ice-water particle
static float ICE_WATER = -0.5;
static float BOUND_LIQUID = 280;
static const float K_WATER = 1.0;//0.01 ;//71.97;
static const float K_ICE = 20.0; //* 10000;//75.64;rr

#endif DEFINITIONS
#ifndef DEFINITIONS
#define DEFINITIONS


// State of particle

enum State { ICE, WATER};

static float ICE_THERMAL_DIFFUSION_CONSTANT = .5;
static float WATER_THERMAL_DIFFUSION_CONSTANT = .1;
static float MASS = 0.0008;
static float THERMAL_CONDUCTIVITY = 0.00267;
static float AMBIENT_TEMPERATURE = 60;
static float SPECIFIC_HEAT_CAPACITY_ICE = 1.0; // 2.11; // units: kJ/kg-K
static float SPECIFIC_HEAT_CAPACITY_WATER = 1.0; //4.181; // units: kJ/kg-K
static float MAX_TEMPERATURE = 60;
static float MIN_TEMPERATURE = 20;
static float PARTICLE_PRADIUS = 1.1;

#endif DEFINITIONS
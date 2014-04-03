#include "voxel_grid.h"
#include <GL/glut.h>
#include <string>
#include <stdlib.h>
#include "voxel_file_format.h"
#include <iostream>
#include <stdio.h>
#include <vector>

// Init fluid system with OBJ
#define ADJUST_SCALE 1.521587
#define ADJUST_OFFSET_X 0//-30
#define ADJUST_OFFSET_Y 0//-10
#define ADJUST_OFFSET_Z 1//-30

VoxelGrid::~VoxelGrid() {
	for(int i=0;i<theDim[0];i++){
		for(int j=0;j<theDim[2];j++){
			delete[] data[i][j];
		}
		delete[] data[i];
	}
	delete[] data;
}

void VoxelGrid::loadGrid(const char* filename) {  
	FILE* voxel_file = fopen(filename, "rb");
	if (voxel_file != NULL) {
		float scaleFactor[3];
		float zeroCoord[3];

		//Read in Header File
		voxelfile_file_header file_hdr;
		fread(&file_hdr,sizeof(file_hdr),1,voxel_file);

		//Read in Object Header
		voxelfile_object_header object_hdr;
		fread(&object_hdr,sizeof(object_hdr),1,voxel_file);

		//Set Resolution
		theDim[0] = object_hdr.voxel_resolution[0];
		theDim[1] = object_hdr.voxel_resolution[1];
		theDim[2] = object_hdr.voxel_resolution[2];

		//Set Scale Factor
		scaleFactor[0] = object_hdr.model_scale_factor*ADJUST_SCALE;
		scaleFactor[1] = object_hdr.model_scale_factor*ADJUST_SCALE;
		scaleFactor[2] = object_hdr.model_scale_factor*ADJUST_SCALE;

		//Set Voxel Grid Size
		voxelSize[0] = object_hdr.voxel_size[0]*scaleFactor[0];
		voxelSize[1] = object_hdr.voxel_size[1]*scaleFactor[1];
		voxelSize[2] = object_hdr.voxel_size[2]*scaleFactor[2];

		//Set Voxel Grid Size
		offset[0] = object_hdr.model_offset[0]+ADJUST_OFFSET_X;
		offset[1] = object_hdr.model_offset[1]+ADJUST_OFFSET_Y;
		offset[2] = object_hdr.model_offset[2]+ADJUST_OFFSET_Z;

		int size_x = theDim[0];
		int size_y = theDim[2];
		int size_z = theDim[1];

		data = new bool**[size_x];
		for (int i = 0; i < size_x; ++i) {
			data[i] = new bool*[size_y];
			for (int j = 0; j < size_y; ++j) {
				data[i][j] = new bool[size_z];
				for (int k = 0; k < size_z; ++k) {
					data[i][j][k] = false;
				}
			}
		}

		adjacencyList = new short**[theDim[0]];
		for (int i = 0; i < size_x; ++i) {
			adjacencyList[i] = new short*[size_y];
			for (int j = 0; j < size_y; ++j) {
				adjacencyList[i][j] = new short[size_z];
				for (int k = 0; k < size_z; ++k) {
					adjacencyList[i][j][k] = 0;
				}
			}
		}

		char v;
		voxelfile_voxel c_voxel;
		//Alternative Method
		int count=0;
		while(count<object_hdr.num_voxels)
		{
			fread(&v,sizeof(char),1,voxel_file);
			if(v != ASCIIVOXEL_NOVOXEL){
				fread(&c_voxel,sizeof(c_voxel),1,voxel_file);
				data[c_voxel.i][c_voxel.k][c_voxel.j] = true;
				count++;
			}
		}
		std::cout << " count in voxel grid " << count << std::endl;

		fclose(voxel_file);
	} else {
		std::cout << "File Does not Exist" << std::endl;
	}
}

Vector3DF VoxelGrid::getCellCenter(int i, int j, int k)
{
	double x = (i + 0.5f) * voxelSize[0] + offset[0];
	double y = (j + 0.5f) * voxelSize[1] + offset[1];
	double z = (k + 0.5f) * voxelSize[2] + offset[2];

	return Vector3DF(x,y,z);
}

Vector3DF VoxelGrid::inVoxelGrid(double x, double y, double z) {
	float size_x = theDim[0];
	float size_y = theDim[2];
	float size_z = theDim[1];

	int i = (x-offset[0])/voxelSize[0];
	int j = (y-offset[1])/voxelSize[2]; // intentional switch of axis
	int k = (z-offset[2])/voxelSize[1];

	//if (i < 0 || j < 0 || k < 0 || i >= theDim[0]-1 || j >= theDim[2]-1 || k >= theDim[1]-1)
	if (i < 0 || j < 0 || k < 0 || i >= size_x|| j >= size_y|| k >= size_z)
		return Vector3DF(-1.0, -1.0, -1.0);

	//std::cout << "(" << i << "," << j << "," << k << ")" << std::endl;
    if(data[i][j][k])
        return Vector3DF(i,j,k);
    else 
        return Vector3DF(-1.0, -1.0, -1.0);
}
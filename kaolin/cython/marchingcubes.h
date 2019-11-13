// Copyright 2019 Lars Mescheder, Michael Oechsle,
// Michael Niemeyer, Andreas Geiger, Sebastian Nowozin

// Permission is hereby granted, free of charge,
// to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to 
//  in the Software without restriction, including without
//  limitation the rights to use, copy, modify, merge, publish,
//  distribute, sublicense, and/or sell copies of the Software,
//  and to permit persons to whom the Software is furnished to do so,
//  subject to the following conditions:

// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _MARCHING_CUBES_H
#define _MARCHING_CUBES_H

#include <stddef.h>
#include <vector>

namespace mc
{

extern int edge_table[256];
extern int triangle_table[256][16];

namespace private_
{

double mc_isovalue_interpolation(double isovalue, double f1, double f2,
    double x1, double x2);
void mc_add_vertex(double x1, double y1, double z1, double c2,
    int axis, double f1, double f2, double isovalue, std::vector<double>* vertices);
}

template<typename coord_type, typename vector3, typename formula>
void marching_cubes(const vector3& lower, const vector3& upper,
    int numx, int numy, int numz, formula f, double isovalue,
    std::vector<double>& vertices, std::vector<size_t>& polygons)
{
    using namespace private_;

    // typedef decltype(lower[0]) coord_type;

    // numx, numy and numz are the numbers of evaluations in each direction
    --numx; --numy; --numz;

    coord_type dx = (upper[0] - lower[0])/static_cast<coord_type>(numx);
    coord_type dy = (upper[1] - lower[1])/static_cast<coord_type>(numy);
    coord_type dz = (upper[2] - lower[2])/static_cast<coord_type>(numz);

    size_t* shared_indices = new size_t[2*numy*numz*3];
    const int z3 = numz*3;
    const int yz3 = numy*z3;

    for(int i=0; i<numx; ++i)
    {
        coord_type x = lower[0] + dx*i + dx/2;
        coord_type x_dx = lower[0] + dx*(i+1) + dx/2;
        const int i_mod_2 = i % 2;
        const int i_mod_2_inv = (i_mod_2 ? 0 : 1);

        for(int j=0; j<numy; ++j)
        {
            coord_type y = lower[1] + dy*j + dy/2;
            coord_type y_dy = lower[1] + dy*(j+1) + dy/2;
            for(int k=0; k<numz; ++k)
            {
                coord_type z = lower[2] + dz*k + dz/2;
                coord_type z_dz = lower[2] + dz*(k+1) + dz/2;

                double v[8];
                v[0] = f(x,y,z); v[1] = f(x_dx,y,z);
                v[2] = f(x_dx,y_dy,z); v[3] = f(x, y_dy, z);
                v[4] = f(x,y,z_dz); v[5] = f(x_dx,y,z_dz);
                v[6] = f(x_dx,y_dy,z_dz); v[7] = f(x, y_dy, z_dz);

                unsigned int cubeindex = 0;
                for(int m=0; m<8; ++m)
                    if(v[m] <= isovalue)
                        cubeindex |= 1<<m;

                // Generate vertices AVOIDING DUPLICATES.

                int edges = edge_table[cubeindex];
                std::vector<size_t> indices(12, -1);
                if(edges & 0x040)
                {
                    indices[6] = vertices.size() / 3;
                    shared_indices[i_mod_2*yz3 + j*z3 + k*3 + 0] = indices[6];
                    mc_add_vertex(x_dx, y_dy, z_dz, x, 0, v[6], v[7], isovalue, &vertices);
                }
                if(edges & 0x020)
                {
                    indices[5] = vertices.size() / 3;
                    shared_indices[i_mod_2*yz3 + j*z3 + k*3 + 1] = indices[5];
                    mc_add_vertex(x_dx, y, z_dz, y_dy, 1, v[5], v[6], isovalue, &vertices);
                }
                if(edges & 0x400)
                {
                    indices[10] = vertices.size() / 3;
                    shared_indices[i_mod_2*yz3 + j*z3 + k*3 + 2] = indices[10];
                    mc_add_vertex(x_dx, y+dx, z, z_dz, 2, v[2], v[6], isovalue, &vertices);
                }

                if(edges & 0x001)
                {
                    if(j == 0 || k == 0)
                    {
                      indices[0] = vertices.size() / 3;
                      mc_add_vertex(x, y, z, x_dx, 0, v[0], v[1], isovalue, &vertices);
                    }
                    else
                        indices[0] = shared_indices[i_mod_2*yz3 + (j-1)*z3 + (k-1)*3 + 0];
                }
                if(edges & 0x002)
                {
                    if(k == 0)
                    {
                        indices[1] = vertices.size() / 3;
                        mc_add_vertex(x_dx, y, z, y_dy, 1, v[1], v[2], isovalue, &vertices);
                    }
                    else
                        indices[1] = shared_indices[i_mod_2*yz3 + j*z3 + (k-1)*3 + 1];
                }
                if(edges & 0x004)
                {
                    if(k == 0)
                    {
                        indices[2] = vertices.size() / 3;
                        mc_add_vertex(x_dx, y_dy, z, x, 0, v[2], v[3], isovalue, &vertices);
                    }
                    else
                        indices[2] = shared_indices[i_mod_2*yz3 + j*z3 + (k-1)*3 + 0];
                }
                if(edges & 0x008)
                {
                    if(i == 0 || k == 0)
                    {
                        indices[3] = vertices.size() / 3;
                        mc_add_vertex(x, y_dy, z, y, 1, v[3], v[0], isovalue, &vertices);
                    }
                    else
                        indices[3] = shared_indices[i_mod_2_inv*yz3 + j*z3 + (k-1)*3 + 1];
                }
                if(edges & 0x010)
                {
                    if(j == 0)
                    {
                        indices[4] = vertices.size() / 3;
                        mc_add_vertex(x, y, z_dz, x_dx, 0, v[4], v[5], isovalue, &vertices);
                    }
                    else
                        indices[4] = shared_indices[i_mod_2*yz3 + (j-1)*z3 + k*3 + 0];
                }
                if(edges & 0x080)
                {
                    if(i == 0)
                    {
                        indices[7] = vertices.size() / 3;
                        mc_add_vertex(x, y_dy, z_dz, y, 1, v[7], v[4], isovalue, &vertices);
                    }
                    else
                        indices[7] = shared_indices[i_mod_2_inv*yz3 + j*z3 + k*3 + 1];
                }
                if(edges & 0x100)
                {
                    if(i == 0 || j == 0)
                    {
                        indices[8] = vertices.size() / 3;
                        mc_add_vertex(x, y, z, z_dz, 2, v[0], v[4], isovalue, &vertices);
                    }
                    else
                        indices[8] = shared_indices[i_mod_2_inv*yz3 + (j-1)*z3 + k*3 + 2];
                }
                if(edges & 0x200)
                {
                    if(j == 0)
                    {
                        indices[9] = vertices.size() / 3;
                        mc_add_vertex(x_dx, y, z, z_dz, 2, v[1], v[5], isovalue, &vertices);
                    }
                    else
                        indices[9] = shared_indices[i_mod_2*yz3 + (j-1)*z3 + k*3 + 2];
                }
                if(edges & 0x800)
                {
                    if(i == 0)
                    {
                        indices[11] = vertices.size() / 3;
                        mc_add_vertex(x, y_dy, z, z_dz, 2, v[3], v[7], isovalue, &vertices);
                    }
                    else
                        indices[11] = shared_indices[i_mod_2_inv*yz3 + j*z3 + k*3 + 2];
                }

                int tri;
                int* triangle_table_ptr = triangle_table[cubeindex];
                for(int m=0; tri = triangle_table_ptr[m], tri != -1; ++m)
                    polygons.push_back(indices[tri]);
            }
        }
    }

    delete [] shared_indices;
}

template<typename coord_type, typename vector3, typename formula>
void marching_cubes2(const vector3& lower, const vector3& upper,
    int numx, int numy, int numz, formula f, double isovalue,
    std::vector<double>& vertices, std::vector<size_t>& polygons)
{
    using namespace private_;

    // typedef decltype(lower[0]) coord_type;

    // numx, numy and numz are the numbers of evaluations in each direction
    --numx; --numy; --numz;

    coord_type dx = (upper[0] - lower[0])/static_cast<coord_type>(numx);
    coord_type dy = (upper[1] - lower[1])/static_cast<coord_type>(numy);
    coord_type dz = (upper[2] - lower[2])/static_cast<coord_type>(numz);

    size_t* shared_indices = new size_t[2*numy*numz*3];
    const int z3 = numz*3;
    const int yz3 = numy*z3;

    for(int i=0; i<numx; ++i)
    {
        coord_type x = lower[0] + dx*i;
        coord_type x_dx = lower[0] + dx*(i+1);
        const int i_mod_2 = i % 2;
        const int i_mod_2_inv = (i_mod_2 ? 0 : 1);

        for(int j=0; j<numy; ++j)
        {
            coord_type y = lower[1] + dy*j;
            coord_type y_dy = lower[1] + dy*(j+1);
            for(int k=0; k<numz; ++k)
            {
                coord_type z = lower[2] + dz*k;
                coord_type z_dz = lower[2] + dz*(k+1);

                double v[8];
                v[0] = f(x,y,z); v[1] = f(x_dx,y,z);
                v[2] = f(x_dx,y_dy,z); v[3] = f(x, y_dy, z);
                v[4] = f(x,y,z_dz); v[5] = f(x_dx,y,z_dz);
                v[6] = f(x_dx,y_dy,z_dz); v[7] = f(x, y_dy, z_dz);

                unsigned int cubeindex = 0;
                for(int m=0; m<8; ++m)
                    if(v[m] <= isovalue)
                        cubeindex |= 1<<m;

                // Generate vertices AVOIDING DUPLICATES.

                int edges = edge_table[cubeindex];
                std::vector<size_t> indices(12, -1);
                if(edges & 0x040)
                {
                    indices[6] = vertices.size() / 3;
                    shared_indices[i_mod_2*yz3 + j*z3 + k*3 + 0] = indices[6];
                    mc_add_vertex(x_dx, y_dy, z_dz, x, 0, v[6], v[7], isovalue, &vertices);
                }
                if(edges & 0x020)
                {
                    indices[5] = vertices.size() / 3;
                    shared_indices[i_mod_2*yz3 + j*z3 + k*3 + 1] = indices[5];
                    mc_add_vertex(x_dx, y, z_dz, y_dy, 1, v[5], v[6], isovalue, &vertices);
                }
                if(edges & 0x400)
                {
                    indices[10] = vertices.size() / 3;
                    shared_indices[i_mod_2*yz3 + j*z3 + k*3 + 2] = indices[10];
                    mc_add_vertex(x_dx, y+dx, z, z_dz, 2, v[2], v[6], isovalue, &vertices);
                }

                if(edges & 0x001)
                {
                    if(j == 0 || k == 0)
                    {
                      indices[0] = vertices.size() / 3;
                      mc_add_vertex(x, y, z, x_dx, 0, v[0], v[1], isovalue, &vertices);
                    }
                    else
                        indices[0] = shared_indices[i_mod_2*yz3 + (j-1)*z3 + (k-1)*3 + 0];
                }
                if(edges & 0x002)
                {
                    if(k == 0)
                    {
                        indices[1] = vertices.size() / 3;
                        mc_add_vertex(x_dx, y, z, y_dy, 1, v[1], v[2], isovalue, &vertices);
                    }
                    else
                        indices[1] = shared_indices[i_mod_2*yz3 + j*z3 + (k-1)*3 + 1];
                }
                if(edges & 0x004)
                {
                    if(k == 0)
                    {
                        indices[2] = vertices.size() / 3;
                        mc_add_vertex(x_dx, y_dy, z, x, 0, v[2], v[3], isovalue, &vertices);
                    }
                    else
                        indices[2] = shared_indices[i_mod_2*yz3 + j*z3 + (k-1)*3 + 0];
                }
                if(edges & 0x008)
                {
                    if(i == 0 || k == 0)
                    {
                        indices[3] = vertices.size() / 3;
                        mc_add_vertex(x, y_dy, z, y, 1, v[3], v[0], isovalue, &vertices);
                    }
                    else
                        indices[3] = shared_indices[i_mod_2_inv*yz3 + j*z3 + (k-1)*3 + 1];
                }
                if(edges & 0x010)
                {
                    if(j == 0)
                    {
                        indices[4] = vertices.size() / 3;
                        mc_add_vertex(x, y, z_dz, x_dx, 0, v[4], v[5], isovalue, &vertices);
                    }
                    else
                        indices[4] = shared_indices[i_mod_2*yz3 + (j-1)*z3 + k*3 + 0];
                }
                if(edges & 0x080)
                {
                    if(i == 0)
                    {
                        indices[7] = vertices.size() / 3;
                        mc_add_vertex(x, y_dy, z_dz, y, 1, v[7], v[4], isovalue, &vertices);
                    }
                    else
                        indices[7] = shared_indices[i_mod_2_inv*yz3 + j*z3 + k*3 + 1];
                }
                if(edges & 0x100)
                {
                    if(i == 0 || j == 0)
                    {
                        indices[8] = vertices.size() / 3;
                        mc_add_vertex(x, y, z, z_dz, 2, v[0], v[4], isovalue, &vertices);
                    }
                    else
                        indices[8] = shared_indices[i_mod_2_inv*yz3 + (j-1)*z3 + k*3 + 2];
                }
                if(edges & 0x200)
                {
                    if(j == 0)
                    {
                        indices[9] = vertices.size() / 3;
                        mc_add_vertex(x_dx, y, z, z_dz, 2, v[1], v[5], isovalue, &vertices);
                    }
                    else
                        indices[9] = shared_indices[i_mod_2*yz3 + (j-1)*z3 + k*3 + 2];
                }
                if(edges & 0x800)
                {
                    if(i == 0)
                    {
                        indices[11] = vertices.size() / 3;
                        mc_add_vertex(x, y_dy, z, z_dz, 2, v[3], v[7], isovalue, &vertices);
                    }
                    else
                        indices[11] = shared_indices[i_mod_2_inv*yz3 + j*z3 + k*3 + 2];
                }

                int tri;
                int* triangle_table_ptr = triangle_table[cubeindex];
                for(int m=0; tri = triangle_table_ptr[m], tri != -1; ++m)
                    polygons.push_back(indices[tri]);
            }
        }
    }

    delete [] shared_indices;
}

template<typename coord_type, typename vector3, typename formula>
void marching_cubes3(const vector3& lower, const vector3& upper,
    int numx, int numy, int numz, formula f, double isovalue,
    std::vector<double>& vertices, std::vector<size_t>& polygons)
{
    using namespace private_;

    // typedef decltype(lower[0]) coord_type;

    // numx, numy and numz are the numbers of evaluations in each direction
    --numx; --numy; --numz;

    coord_type dx = (upper[0] - lower[0])/static_cast<coord_type>(numx);
    coord_type dy = (upper[1] - lower[1])/static_cast<coord_type>(numy);
    coord_type dz = (upper[2] - lower[2])/static_cast<coord_type>(numz);

    size_t* shared_indices = new size_t[2*numy*numz*3];
    const int z3 = numz*3;
    const int yz3 = numy*z3;

    for(int i=0; i<numx; ++i)
    {
        coord_type x = lower[0] + dx*i - dx/2;
        coord_type x_dx = lower[0] + dx*(i+1) - dx/2;
        const int i_mod_2 = i % 2;
        const int i_mod_2_inv = (i_mod_2 ? 0 : 1);

        for(int j=0; j<numy; ++j)
        {
            coord_type y = lower[1] + dy*j - dy/2;
            coord_type y_dy = lower[1] + dy*(j+1) - dy/2;
            for(int k=0; k<numz; ++k)
            {
                coord_type z = lower[2] + dz*k - dz/2;
                coord_type z_dz = lower[2] + dz*(k+1) - dz/2;

                double v[8];
                v[0] = f(x,y,z); v[1] = f(x_dx,y,z);
                v[2] = f(x_dx,y_dy,z); v[3] = f(x, y_dy, z);
                v[4] = f(x,y,z_dz); v[5] = f(x_dx,y,z_dz);
                v[6] = f(x_dx,y_dy,z_dz); v[7] = f(x, y_dy, z_dz);

                unsigned int cubeindex = 0;
                for(int m=0; m<8; ++m)
                    if(v[m] <= isovalue)
                        cubeindex |= 1<<m;

                // Generate vertices AVOIDING DUPLICATES.

                int edges = edge_table[cubeindex];
                std::vector<size_t> indices(12, -1);
                if(edges & 0x040)
                {
                    indices[6] = vertices.size() / 3;
                    shared_indices[i_mod_2*yz3 + j*z3 + k*3 + 0] = indices[6];
                    mc_add_vertex(x_dx, y_dy, z_dz, x, 0, v[6], v[7], isovalue, &vertices);
                }
                if(edges & 0x020)
                {
                    indices[5] = vertices.size() / 3;
                    shared_indices[i_mod_2*yz3 + j*z3 + k*3 + 1] = indices[5];
                    mc_add_vertex(x_dx, y, z_dz, y_dy, 1, v[5], v[6], isovalue, &vertices);
                }
                if(edges & 0x400)
                {
                    indices[10] = vertices.size() / 3;
                    shared_indices[i_mod_2*yz3 + j*z3 + k*3 + 2] = indices[10];
                    mc_add_vertex(x_dx, y+dx, z, z_dz, 2, v[2], v[6], isovalue, &vertices);
                }

                if(edges & 0x001)
                {
                    if(j == 0 || k == 0)
                    {
                      indices[0] = vertices.size() / 3;
                      mc_add_vertex(x, y, z, x_dx, 0, v[0], v[1], isovalue, &vertices);
                    }
                    else
                        indices[0] = shared_indices[i_mod_2*yz3 + (j-1)*z3 + (k-1)*3 + 0];
                }
                if(edges & 0x002)
                {
                    if(k == 0)
                    {
                        indices[1] = vertices.size() / 3;
                        mc_add_vertex(x_dx, y, z, y_dy, 1, v[1], v[2], isovalue, &vertices);
                    }
                    else
                        indices[1] = shared_indices[i_mod_2*yz3 + j*z3 + (k-1)*3 + 1];
                }
                if(edges & 0x004)
                {
                    if(k == 0)
                    {
                        indices[2] = vertices.size() / 3;
                        mc_add_vertex(x_dx, y_dy, z, x, 0, v[2], v[3], isovalue, &vertices);
                    }
                    else
                        indices[2] = shared_indices[i_mod_2*yz3 + j*z3 + (k-1)*3 + 0];
                }
                if(edges & 0x008)
                {
                    if(i == 0 || k == 0)
                    {
                        indices[3] = vertices.size() / 3;
                        mc_add_vertex(x, y_dy, z, y, 1, v[3], v[0], isovalue, &vertices);
                    }
                    else
                        indices[3] = shared_indices[i_mod_2_inv*yz3 + j*z3 + (k-1)*3 + 1];
                }
                if(edges & 0x010)
                {
                    if(j == 0)
                    {
                        indices[4] = vertices.size() / 3;
                        mc_add_vertex(x, y, z_dz, x_dx, 0, v[4], v[5], isovalue, &vertices);
                    }
                    else
                        indices[4] = shared_indices[i_mod_2*yz3 + (j-1)*z3 + k*3 + 0];
                }
                if(edges & 0x080)
                {
                    if(i == 0)
                    {
                        indices[7] = vertices.size() / 3;
                        mc_add_vertex(x, y_dy, z_dz, y, 1, v[7], v[4], isovalue, &vertices);
                    }
                    else
                        indices[7] = shared_indices[i_mod_2_inv*yz3 + j*z3 + k*3 + 1];
                }
                if(edges & 0x100)
                {
                    if(i == 0 || j == 0)
                    {
                        indices[8] = vertices.size() / 3;
                        mc_add_vertex(x, y, z, z_dz, 2, v[0], v[4], isovalue, &vertices);
                    }
                    else
                        indices[8] = shared_indices[i_mod_2_inv*yz3 + (j-1)*z3 + k*3 + 2];
                }
                if(edges & 0x200)
                {
                    if(j == 0)
                    {
                        indices[9] = vertices.size() / 3;
                        mc_add_vertex(x_dx, y, z, z_dz, 2, v[1], v[5], isovalue, &vertices);
                    }
                    else
                        indices[9] = shared_indices[i_mod_2*yz3 + (j-1)*z3 + k*3 + 2];
                }
                if(edges & 0x800)
                {
                    if(i == 0)
                    {
                        indices[11] = vertices.size() / 3;
                        mc_add_vertex(x, y_dy, z, z_dz, 2, v[3], v[7], isovalue, &vertices);
                    }
                    else
                        indices[11] = shared_indices[i_mod_2_inv*yz3 + j*z3 + k*3 + 2];
                }

                int tri;
                int* triangle_table_ptr = triangle_table[cubeindex];
                for(int m=0; tri = triangle_table_ptr[m], tri != -1; ++m)
                    polygons.push_back(indices[tri]);
            }
        }
    }

    delete [] shared_indices;
}

}

#endif // _MARCHING_CUBES_H

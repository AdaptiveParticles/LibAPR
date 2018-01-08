#include "benchmarks/development/old_structures/structure_parts.h"

#include <bitset>
#include <fstream>

#include "src/io/parameters.h"

bool operator==(const Cell_id & lhs, const Cell_id & rhs){
    //relational operator for particle cell ids

#ifdef data_2D
    //2D
    if (lhs.k != rhs.k) {
        return false;
    }
    else if( lhs.y != rhs.y){
        return false;

    } else {
        return lhs.x == rhs.x;
    }

#else
    //3D

    if (lhs.k != rhs.k) {
        return false;
    }
    else if( lhs.y != rhs.y){
        return false;

    }
    else if( lhs.x != rhs.x){
        return false;

    } else {
        return lhs.z == rhs.z;
    }

#endif



};
bool operator<(const Cell_id & lhs, const Cell_id & rhs){
    //relational operator for particle cell ids

#ifdef data_2D
    //2D
    if (lhs.k != rhs.k) {
        return lhs.k < rhs.k;
    }
    else if( lhs.y != rhs.y){
        return lhs.y < rhs.y;

    } else {
        return lhs.x < rhs.x;
    }

#else
    //3D

    if (lhs.k != rhs.k) {
        return lhs.k < rhs.k;
    }
    else if( lhs.y != rhs.y){
        return lhs.y < rhs.y;

    }
    else if( lhs.x != rhs.x){
        return lhs.x < rhs.x;

    } else {
        return lhs.z < rhs.z;
    }

#endif

};


void Part_rep::get_part_coords_from_cell(Cell_id curr_cell_id,uint8_t status,std::vector<std::vector<int>>& co_ords,int k_max,int& num_parts){
    //
    // Calculates the co_ordinates of the particles in the cell
    //

    int x_incr[8];
    int y_incr[8];
    int z_incr[8];

    x_incr[0] = -3;
    x_incr[1] = -1;
    x_incr[2] = -3;
    x_incr[3] = -1;
    x_incr[4] = -3;
    x_incr[5] = -1;
    x_incr[6] = -3;
    x_incr[7] = -1;

    y_incr[0] = -3;
    y_incr[1] = -3;
    y_incr[2] = -1;
    y_incr[3] = -1;
    y_incr[4] = -3;
    y_incr[5] = -3;
    y_incr[6] = -1;
    y_incr[7] = -1;

    z_incr[0] = -3;
    z_incr[1] = -3;
    z_incr[2] = -3;
    z_incr[3] = -3;
    z_incr[4] = -1;
    z_incr[5] = -1;
    z_incr[6] = -1;
    z_incr[7] = -1;

    int shift_x,shift_y,shift_z;

    float k_factor = pow(2,k_max- curr_cell_id.k);

    shift_y = curr_cell_id.y*k_factor * 4;
    shift_x = curr_cell_id.x*k_factor * 4;
    shift_z = curr_cell_id.z*k_factor * 4;

    if (pars.part_config == 0) {
        // old configuration
        if (status == 2){

            for(int i = 0; i < 8;i++){
                co_ords[0][i] = shift_y + (k_factor)*y_incr[i];
                co_ords[1][i] = shift_x + (k_factor)*x_incr[i];
                co_ords[2][i] = shift_z + (k_factor)*z_incr[i];
            }

            num_parts = 8;
        }
        else if (status == 5){


            for(int i = 0; i < 2;i++){
                co_ords[0][i] = shift_y + (k_factor)*y_incr[i];
                co_ords[1][i] = shift_x + (k_factor)*x_incr[i];
                co_ords[2][i] = shift_z + (k_factor)*z_incr[i];
            }

            num_parts = 2;

        } else {
            //get the num parts form the binary rep of the status
            status = status - 6;
            std::bitset<8> bit_rep(status);

            int counter = 0;

            //add two particles that are always there
            for(int i = 0; i < 2;i++){
                co_ords[0][i] = shift_y + (k_factor)*y_incr[i];
                co_ords[1][i] = shift_x + (k_factor)*x_incr[i];
                co_ords[2][i] = shift_z + (k_factor)*z_incr[i];
            }

            //add the support particles
            for(int i = 2; i < 8;i++){

                if (bit_rep[i-2] == 1){
                    co_ords[0][counter + 2] = shift_y + (k_factor)*y_incr[i];
                    co_ords[1][counter + 2] = shift_x + (k_factor)*x_incr[i];
                    co_ords[2][counter + 2] = shift_z + (k_factor)*z_incr[i];
                    counter++;
                }
            }

            num_parts = (int)bit_rep.count() + 2;

        }

    } else if (pars.part_config == 1) {
        // new configuration
        if (status == 2){

            for(int i = 0; i < 8;i++){
                co_ords[0][i] = shift_y + (k_factor)*y_incr[i];
                co_ords[1][i] = shift_x + (k_factor)*x_incr[i];
                co_ords[2][i] = shift_z + (k_factor)*z_incr[i];
            }

            num_parts = 8;
        }
        else if ((status == 5) | (status == 4) ){

            co_ords[0][0] = shift_y + (k_factor)*(-2);
            co_ords[1][0] = shift_x + (k_factor)*(-2);
            co_ords[2][0] = shift_z + (k_factor)*(-2);

            num_parts = 1;

        }

    }



}
void Part_rep::get_all_part_co_ords(){
    //
    //  Creates x,y,z vectors for the particle locations and returns pointers to them, or just gets the pointers
    //

    //first check if the co_ordinates have already been calculated, then generate them..

    Part_data<uint16_t> *x_coords_p;

    (x_coords_p) = (get_data_ref<uint16_t>("x_coords"));

    if ((x_coords_p) == nullptr) {
        // The dataset doesn't exit, need to create it

        create_uint16_dataset("x_coords", num_parts);
        create_uint16_dataset("y_coords", num_parts);
        create_uint16_dataset("z_coords", num_parts);

        Part_data<uint16_t>& x_coords = *(get_data_ref<uint16_t>("x_coords"));
        Part_data<uint16_t>& y_coords = *(get_data_ref<uint16_t>("y_coords"));
        Part_data<uint16_t>& z_coords = *(get_data_ref<uint16_t>("z_coords"));

        int num_parts;

        std::vector<std::vector<int>> co_ords;

        co_ords.resize(3);
        co_ords[0].resize(8);
        co_ords[1].resize(8);
        co_ords[2].resize(8);


        Cell_index curr_index;
        Cell_id curr_cell;


        for (int i = 0; i < pl_map.cells.size(); i++) {
            if((status.data[i] == 2) | (status.data[i] == 4) | (status.data[i] == 5)){

                curr_index = pl_map.cell_indices[i];
                curr_cell = pl_map.cells[i];

                get_part_coords_from_cell(curr_cell,status.data[curr_index.cindex],co_ords,pl_map.k_max,num_parts);

                std::copy(co_ords[0].begin(),co_ords[0].begin() + num_parts,y_coords.data.begin() + curr_index.first);
                std::copy(co_ords[1].begin(),co_ords[1].begin() + num_parts,x_coords.data.begin() + curr_index.first);
                std::copy(co_ords[2].begin(),co_ords[2].begin() + num_parts,z_coords.data.begin() + curr_index.first);
            }


        }


        x = get_data_ref<uint16_t>("x_coords");
        y = get_data_ref<uint16_t>("y_coords");
        z = get_data_ref<uint16_t>("z_coords");



    } else {
        //data already exists just need to go find it
        x = get_data_ref<uint16_t>("x_coords");
        y = get_data_ref<uint16_t>("y_coords");
        z = get_data_ref<uint16_t>("z_coords");

    }

}
void get_all_part_k(Part_rep& p_rep,Part_data<uint8_t>& k_vec){
    //
    //  Creates k vector for the particle locations
    //

    k_vec.data.resize(p_rep.num_parts);

    Cell_index curr_index;

    for(int k_ = p_rep.pl_map.k_min;k_ < p_rep.pl_map.k_max + 1;k_++){
        for(auto const &cell_ref : p_rep.pl_map.pl[k_]) {
            curr_index = p_rep.pl_map.cell_indices[cell_ref.second];;

            std::fill(k_vec.data.begin() + curr_index.first,k_vec.data.begin() + curr_index.last,k_);
        }
    }

}
void get_all_part_type(Part_rep& p_rep,Part_data<uint8_t>& type_vec){
    //
    //  Creates type vector for the particle locations
    //
    //  1 = seed particle
    //  2 = boundary seed particle
    //  3 = filler particle
    //

    type_vec.data.resize(p_rep.num_parts);

    uint8_t status,type;

    Cell_index curr_index;

    for(int k_ = p_rep.pl_map.k_min;k_ < p_rep.pl_map.k_max + 1;k_++){
        for(auto const &cell_ref : p_rep.pl_map.pl[k_]) {

            curr_index = p_rep.pl_map.cell_indices[cell_ref.second];

            //get the cell status
            status = p_rep.status.data[curr_index.cindex];

            if (status == 2) {
                type = 1;
                //seed
            } else if ( status == 5){
                type = 3;
                //boundary
            } else if (status == 4){
                type = 2;
                //filler
            } else {
                type = 4;
            }

            std::fill(type_vec.data.begin() + curr_index.first,type_vec.data.begin() + curr_index.last,type);
        }
    }

}
int Part_rep::get_num_parts_cell(uint8_t status){
    //
    //  Calculates the number of particles from the status
    //
    //

    if (pars.part_config == 0){
        //old configuration
        if (status == 2){
            return 8;
        }
        else if (status == 5){
            return 2;

        } else {
            //get the num parts form the binary rep of the status
            status = status - 6;
            std::bitset<8> bit_rep(status);
            return (int) (bit_rep.count() + 2);

        }
    } else if (pars.part_config == 1) {
        // new configuration
        if (status == 2){
            return 8;
        }
        else if (status == 5){
            return 1;

        } else if (status == 4){
            return 1;
        } else {
            return 0;
        }
    }

    return 0;
}


void get_cell_properties(Part_rep& p_rep,Part_data<uint16_t>& y_coords_cell,Part_data<uint16_t>& x_coords_cell,
                         Part_data<uint16_t>& z_coords_cell,Part_data<uint8_t>& k_vec,Part_data<uint8_t>& type_vec,
                         int type_or_status,int raw_coords){
    //
    //  Bevan Cheeseman 2016
    //
    //  Get the cell properties
    //
    //

    Cell_index curr_index;
    Cell_id curr_cell;

    unsigned int num_cells = (unsigned int)p_rep.pl_map.cells.size();

    //pre_allocate the datasets
    y_coords_cell.data.resize(num_cells);
    x_coords_cell.data.resize(num_cells);
    z_coords_cell.data.resize(num_cells);
    k_vec.data.resize(num_cells);
    type_vec.data.resize(num_cells);

    uint8_t status;
    uint8_t type = 0;

    float k_factor;

    if (raw_coords==0){

        for (int i = 0; i < num_cells; i++) {
            curr_index = p_rep.pl_map.cell_indices[i];
            curr_cell = p_rep.pl_map.cells[i];

            k_factor = pow(2,p_rep.pl_map.k_max- curr_cell.k);

            y_coords_cell.data[i] = (curr_cell.y*4.0 - 2)*(k_factor);
            x_coords_cell.data[i] = (curr_cell.x*4.0 - 2)*(k_factor);
            z_coords_cell.data[i] = (curr_cell.z*4.0 - 2)*(k_factor);
            k_vec.data[i] = curr_cell.k;

            //get the cell status
            status = p_rep.status.data[i];

            if (type_or_status){

                if (status == 2) {
                    type = 1;
                    //seed
                } else if ( status == 5){
                    type = 3;
                    //filler
                }else if ( status == 0){
                    type = 4;
                    //ghost
                }else if(status ==4){
                    type = 2;
                    //boundary
                }

                type_vec.data[i] = type;

            } else {
                type_vec.data[i] = status;
            }

        }
    } else {
        for (int i = 0; i < num_cells; i++) {
            curr_index = p_rep.pl_map.cell_indices[i];
            curr_cell = p_rep.pl_map.cells[i];

            k_factor = pow(2,p_rep.pl_map.k_max- curr_cell.k);

            y_coords_cell.data[i] = curr_cell.y;
            x_coords_cell.data[i] = curr_cell.x;
            z_coords_cell.data[i] = curr_cell.z;
            k_vec.data[i] = curr_cell.k;

            //get the cell status
            status = p_rep.status.data[i];

            if (type_or_status){

                if (status == 2) {
                    type = 1;
                    //seed
                } else if ( status == 5){
                    type = 3;
                    //filler
                }else if ( status == 0){
                    type = 4;
                    //ghost
                }else if(status ==4){
                    type = 2;
                    //boundary
                }

                type_vec.data[i] = type;

            } else {
                type_vec.data[i] = status;
            }

        }


    }

}




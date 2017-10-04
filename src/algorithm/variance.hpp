#ifndef PARTPLAY_VARIANCE_H
#define PARTPLAY_VARIANCE_H

#include <algorithm>
#include <functional>

#include "../data_structures/meshclass.h"
#include "../data_structures/structure_parts.h"

class Map_calc_cpu{
    //
    //
    //  This is a class containing different processing methods used to estimate the part map on the GPU
    //
    //

public:
    unsigned int dir;  //direction of the filter


    //image parameters
    std::vector<float> real_sigmas;
    std::vector<float> real_window_size;
    float cut_th;

    float var_rescale;
    float window_ref;
    float var_scale;

    float max_filter_size;

    std::vector<float> sampling_delta;


    unsigned int max_filter_len1 ;
    unsigned int max_filter_len2 ;
    unsigned int max_filter_len3 ;

    unsigned int var_window_size;

    unsigned int var_window_size1;
    unsigned int var_window_size2;
    unsigned int var_window_size3;


    Proc_par par;

    Map_calc_cpu(unsigned int dir_,Proc_par par_){
        dir = dir_;
        par = par_;
        max_filter_len2 = 0;
        max_filter_len1 = 0;
        max_filter_len3 = 0;


        max_filter_size = round(par_.xdim/(par_.dx*2.0)-1);

        cut_th = 0.01;

        var_scale = par_.var_scale;
        var_rescale = 10;

        real_sigmas.resize(3);
        real_sigmas[0] = par_.psfy;
        real_sigmas[1] = par_.psfx;
        real_sigmas[2] = par_.psfz;

        sampling_delta.resize(3);

        sampling_delta[0] = par_.dy;
        sampling_delta[1] = par_.dx;
        sampling_delta[2] = par_.dz;

        real_window_size.resize(3);
        //calc the parameters
        for (int i = 0;  i < real_window_size.size(); i++) {
            real_window_size[i] = sqrt(log(1/cut_th)*2*pow(real_sigmas[i],2));
        }


    }


    void set_up_var_filters_3D(){


        var_window_size1 = std::min((float)(1.0+2*round(var_scale*real_window_size[0]/sampling_delta[0])),max_filter_size);

        var_window_size1 = std::max((float)3.0,(float)var_window_size1);

        var_window_size2 = std::min((float)(1.0+2*round(var_scale*real_window_size[1]/sampling_delta[1])),max_filter_size);

        var_window_size2 = std::max((float)3.0,(float)var_window_size2);

        var_window_size3 = std::min((float)(1.0+2*round(var_scale*real_window_size[2]/sampling_delta[2])),max_filter_size);

        var_window_size3 = std::max((float)3.0,(float)var_window_size3);


        max_filter_len1 = std::max(max_filter_len1,(unsigned int)((var_window_size1-1)/2.0));
        max_filter_len2 = std::max(max_filter_len2,(unsigned int)((var_window_size2-1)/2.0));
        max_filter_len3 = std::max(max_filter_len3,(unsigned int)((var_window_size3-1)/2.0));


        window_ref = std::max((float)3.0,(float)(1+2*round(real_window_size[0]/sampling_delta[0])));


        float window_rel1 = 1.0*var_window_size1/(std::max((float)3.0,(float)(1+2*round(real_window_size[0]/sampling_delta[0]))));

        float window_rel2 = 1.0*var_window_size2/(std::max((float)3.0,(float)(1+2*round(real_window_size[1]/sampling_delta[1]))));

        float window_rel3 = 1.0*var_window_size3/(std::max((float)3.0,(float)(1+2*round(real_window_size[2]/sampling_delta[2]))));


        float rescale_par = pow(window_rel1*window_rel2*window_rel3,1.0/3.0);

        window_ref = rescale_par;


        float sig = pow(real_sigmas[0]*real_sigmas[1]*real_sigmas[2],1.0/3.0);


        var_rescale = 1.0/(0.02201*pow(rescale_par,3.0) - 0.146*pow(rescale_par,2.0) + 0.3521*rescale_par - 0.09969);

        if(sig < 0.73){

            var_rescale = 1.0/(0.79*pow(sig,3.0) - 1.7*pow(sig,2.0) + .87*pow(sig,1.0)-.012);


        } else {
            var_rescale = 1.0/(.015 + .046*sig);
        }

        var_rescale = par.var_scale;

        std::vector<std::vector<std::vector<float>>> rescale_store ={{{{16.9769},{8.628},{6.4631},{5.5352},{4.8488},{4.6435},{4.3354},{4.1406},{3.9932},{3.926},{3.7388}},{{19.1518},{9.9501},{6.8968},{5.3009},{4.9696},{4.4534},{4.2216},{4.1036},{4.0919},{4.0657},{3.8522}},{{24.0889},{11.1562},{7.8479},{6.1636},{5.0663},{4.9335},{4.5682},{4.629},{4.1628},{4.3992},{4.512}},{{27.17},{12.6038},{8.8751},{6.8596},{5.7551},{5.1839},{4.8905},{4.8323},{4.9267},{4.8601},{4.6752}},{{31.219},{14.611},{10.3246},{7.4114},{6.528},{5.9386},{5.4942},{5.2487},{5.1634},{5.0597},{5.4911}},{{34.7882},{17.7455},{11.3276},{9.392},{7.6969},{6.2484},{6.3427},{5.6827},{6.2022},{5.8165},{6.497}},{{41.5018},{20.5435},{12.4031},{10.468},{8.6325},{8.3557},{7.0198},{7.229},{6.8005},{6.5261},{7.8893}},{{45.5236},{24.1119},{15.4761},{11.2859},{10.18},{9.1312},{8.6091},{7.9835},{7.5885},{7.2799},{8.4986}},{{48.8501},{28.2228},{17.7487},{14.0864},{11.6638},{10.7459},{9.2374},{9.1674},{8.7252},{9.506},{10.9329}},{{62.5753},{30.8751},{21.2827},{16.4726},{14.8101},{12.476},{11.1306},{11.1009},{10.7719},{10.5892},{11.4628}},{{107.3033},{66.7713},{37.219},{30.7441},{28.8554},{23.4423},{23.2535},{22.9503},{19.7087},{20.002},{19.2833}}},{{{33.4781},{14.1196},{8.6536},{6.5868},{5.8665},{5.0836},{4.9018},{4.7376},{4.7099},{4.5953},{4.6029}},{{35.1485},{15.0989},{9.2577},{7.0929},{5.888},{5.1619},{4.8638},{4.769},{4.62},{4.6044},{4.5247}},{{36.257},{15.6678},{10.0447},{7.478},{6.1507},{5.2646},{5.1032},{4.8673},{4.6768},{4.6948},{4.5828}},{{38.9444},{17.4314},{10.109},{8.0839},{6.3852},{5.6854},{5.2542},{5.0634},{5.0704},{4.779},{5.0556}},{{42.8708},{18.3376},{11.8808},{8.2094},{6.8677},{6.0337},{5.6794},{5.1433},{5.1838},{4.8694},{5.1225}},{{43.7978},{20.8179},{12.575},{9.6992},{7.5577},{6.4328},{6.3287},{5.8712},{5.6428},{5.5762},{5.4446}},{{49.4379},{22.8418},{15.2339},{10.4771},{8.6254},{7.3263},{6.6185},{6.1644},{6.0252},{6.2615},{6.5542}},{{57.2023},{24.8151},{15.7953},{11.8533},{9.7275},{8.1641},{7.7963},{7.0178},{6.7266},{6.4575},{7.0788}},{{62.9177},{28.4823},{18.0211},{14.6673},{10.8502},{9.5671},{8.8263},{8.1035},{7.5138},{7.563},{8.776}},{{64.7786},{32.1582},{20.0236},{16.1231},{11.077},{11.2159},{9.7006},{8.9712},{8.6148},{8.3906},{8.7792}},{{85.6895},{51.2173},{36.2178},{25.4276},{24.1825},{19.2404},{17.7352},{16.1644},{15.5072},{14.9471},{14.3574}}},{{{62.5312},{26.5403},{15.6849},{10.7082},{8.7374},{7.7284},{6.8484},{6.3563},{6.4888},{6.3287},{6.5842}},{{63.1662},{25.949},{16.1523},{11.4998},{9.0375},{8.0432},{6.7468},{6.8743},{6.732},{6.5283},{6.5714}},{{65.3118},{28.1256},{16.5483},{11.9926},{9.169},{8.1917},{7.5096},{6.7009},{6.6708},{6.7333},{6.4407}},{{66.3453},{29.3534},{17.3178},{11.8987},{9.9317},{8.3381},{7.4244},{7.2651},{6.7112},{6.2778},{6.432}},{{66.4447},{30.2962},{17.9124},{12.6181},{10.2125},{8.6758},{7.9063},{7.302},{6.932},{6.6138},{6.5891}},{{72.4833},{30.3346},{19.328},{13.7372},{11.0446},{8.9351},{8.0078},{7.6391},{7.2624},{7.165},{6.8934}},{{78.9294},{31.737},{19.8099},{14.5908},{11.4029},{9.603},{8.5196},{8.1983},{7.8351},{7.1586},{7.0632}},{{78.1269},{35.8937},{21.5537},{16.5003},{11.8525},{11.2098},{8.7255},{8.4921},{8.2504},{8.081},{7.4495}},{{77.9504},{38.5122},{25.0387},{15.9629},{13.5003},{11.5785},{9.798},{8.8316},{8.5456},{8.7507},{7.7912}},{{85.1816},{41.4754},{25.1345},{16.9004},{14.5978},{12.4148},{10.6805},{10.1286},{9.1234},{9.0549},{9.9711}},{{114.3861},{54.1934},{46.8598},{32.6175},{23.0859},{20.2407},{19.7157},{17.5292},{14.9067},{15.615},{12.6046}}},{{{94.8109},{44.4622},{25.7247},{17.8581},{13.6092},{11.4388},{9.8386},{9.2692},{9.0509},{8.6391},{8.6099}},{{93.3508},{44.1559},{25.4598},{17.271},{13.7236},{12.1683},{10.189},{8.9084},{8.5871},{9.1196},{8.7419}},{{95.8238},{43.6554},{26.5658},{18.7297},{13.2068},{11.9815},{9.9654},{9.3015},{8.5465},{9.0356},{8.2819}},{{91.6733},{45.8746},{26.0809},{18.1804},{14.8921},{12.0864},{10.4313},{9.0498},{9.2835},{9.0287},{8.5902}},{{93.6191},{46.5535},{28.4793},{19.0843},{14.2789},{12.5236},{10.8693},{9.8815},{9.5268},{8.5725},{8.6396}},{{99.542},{49.4243},{28.1561},{18.989},{14.4518},{12.7459},{10.5505},{10.1108},{8.9941},{8.6461},{8.0287}},{{97.4517},{46.4779},{29.5205},{20.291},{15.7544},{12.3173},{11.1124},{9.775},{9.2656},{9.182},{8.107}},{{97.1716},{52.5026},{29.9288},{22.1051},{16.7601},{13.8468},{12.1825},{10.8065},{9.9501},{9.4665},{8.655}},{{96.5752},{48.4374},{33.0858},{22.0064},{17.4869},{14.4932},{12.4786},{11.6806},{10.021},{9.9231},{9.5337}},{{102.6604},{55.7241},{33.4545},{22.8598},{18.4556},{15.5798},{12.4689},{11.3605},{11.1079},{10.43},{10.2523}},{{129.6776},{76.4799},{50.8205},{35.7226},{29.5086},{23.0142},{20.1833},{17.9213},{16.3033},{16.357},{12.7985}}},{{{117.4208},{63.0154},{39.4519},{27.0428},{19.8002},{16.4037},{13.5178},{12.5846},{11.8918},{11.4687},{10.3434}},{{116.9444},{65.1213},{38.3341},{25.8749},{20.041},{17.2062},{13.3996},{12.7537},{11.717},{10.471},{10.792}},{{127.4023},{65.9393},{35.8905},{26.3776},{20.6411},{16.8474},{14.9266},{12.4018},{12.4054},{11.6088},{10.5594}},{{118.9327},{66.0998},{39.2791},{27.1236},{20.2958},{16.0581},{13.7522},{12.7608},{11.9829},{10.7957},{9.8103}},{{112.4779},{65.3363},{40.5562},{26.1793},{20.4013},{16.8146},{13.7622},{13.3829},{12.2659},{11.2316},{9.7878}},{{124.0328},{63.5733},{40.2812},{26.2598},{22.4126},{17.1716},{14.7646},{13.3095},{12.2945},{11.2102},{10.2218}},{{119.3259},{62.126},{40.0724},{28.0288},{20.2436},{16.9361},{15.2447},{12.7201},{12.3003},{10.7115},{9.8501}},{{119.3854},{65.666},{43.4689},{28.9977},{22.6039},{17.8696},{14.9515},{13.4381},{12.4905},{12.3098},{10.654}},{{119.2083},{71.5777},{41.995},{28.6811},{22.7042},{18.3172},{15.4148},{14.0024},{13.8569},{12.613},{10.4365}},{{127.7738},{69.8294},{42.4419},{29.3418},{23.9092},{18.669},{16.3655},{15.1559},{13.7514},{13.3507},{11.1398}},{{131.3805},{81.4716},{58.7189},{41.7905},{31.3916},{25.9737},{23.9274},{20.5865},{18.5793},{18.0177},{14.6751}}},{{{144.4743},{84.0982},{53.5642},{35.3847},{26.878},{22.4535},{18.4642},{16.7524},{15.1259},{13.5064},{10.7144}},{{122.2087},{79.8562},{51.0552},{37.4008},{27.1487},{21.4288},{18.5235},{16.2356},{15.2634},{13.5152},{11.8281}},{{128.7447},{88.6027},{54.8615},{38.6451},{28.5666},{23.5458},{19.8605},{16.5489},{15.6579},{13.3829},{12.1039}},{{136.7543},{83.9167},{55.5174},{38.1009},{27.7775},{22.3148},{19.1952},{17.4361},{15.2176},{14.0244},{11.4303}},{{140.9391},{81.2295},{54.262},{37.5398},{29.9033},{21.7904},{18.9679},{17.3035},{14.997},{14.8881},{10.6178}},{{142.2333},{84.081},{55.6834},{39.0426},{30.6918},{23.1069},{19.8058},{16.723},{14.2339},{14.3083},{11.7628}},{{126.0369},{85.3652},{57.4633},{37.631},{28.2711},{22.6387},{20.2672},{17.7201},{15.5866},{14.4709},{11.862}},{{134.4467},{85.5117},{56.5459},{37.4781},{28.0887},{23.6314},{19.1423},{17.4756},{15.6926},{14.789},{11.0529}},{{140.9904},{88.6147},{58.0526},{37.8601},{27.7367},{24.1074},{20.1849},{19.3109},{15.8602},{15.2353},{11.4083}},{{131.0348},{90.1161},{59.2736},{39.3099},{30.5637},{23.5657},{21.3517},{18.8256},{16.2566},{15.3417},{12.6387}},{{141.0241},{108.7515},{70.8173},{50.3984},{41.4976},{33.8933},{28.5599},{25.5554},{22.3193},{19.2624},{15.023}}}};

        //(PSF,window2,window1) with 0 indexing

        int window_ind_1 = std::min(par.padd_dims[0] - 1,10);
        int window_ind_2 = std::min(par.padd_dims[3] - 1,10);
        int psf_ind = std::max(((float) (round(par.psfx/par.dx) - 1)),((float)0.0f));

        psf_ind = std::min(psf_ind,5);

        var_rescale = rescale_store[psf_ind][window_ind_2][window_ind_1];

        var_rescale = 1;

        //var_rescale = par.var_scale;

        std::cout << "**scale: " << var_rescale << std::endl;

        //int stop = 1;

        //var_rescale = 6.9541;
        //var_rescale = 7.1748;

    }

    void set_up_var_filters_2D(){


        var_window_size1 = std::min((float)(1.0+2*round(var_scale*real_window_size[0]/sampling_delta[0])),max_filter_size);

        var_window_size1 = std::max((float)3.0,(float)var_window_size1);

        var_window_size2 = std::min((float)(1.0+2*round(var_scale*real_window_size[1]/sampling_delta[1])),max_filter_size);

        var_window_size2 = std::max((float)3.0,(float)var_window_size2);

        var_window_size3 = std::min((float)(1.0+2*round(var_scale*real_window_size[2]/sampling_delta[2])),max_filter_size);

        var_window_size3 = std::max((float)3.0,(float)var_window_size3);

        var_window_size3 = 1;



        max_filter_len1 = std::max(max_filter_len1,(unsigned int)((var_window_size1-1)/2.0));
        max_filter_len2 = std::max(max_filter_len2,(unsigned int)((var_window_size2-1)/2.0));
        max_filter_len3 = std::max(max_filter_len3,(unsigned int)((var_window_size3-1)/2.0));


        window_ref =std::max((float)3.0,(float)(1+2*round(real_window_size[0]/sampling_delta[0])));


        float window_rel1 = 1.0*var_window_size1/(std::max((float)3.0,(float)(1+2*round(real_window_size[0]/sampling_delta[0]))));

        float window_rel2 = 1.0*var_window_size2/(std::max((float)3.0,(float)(1+2*round(real_window_size[1]/sampling_delta[1]))));

        float window_rel3 = 1.0*var_window_size3/(std::max((float)3.0,(float)(1+2*round(real_window_size[2]/sampling_delta[2]))));
        window_rel3 = 1;


        float rescale_par = pow(window_rel1*window_rel2*window_rel3,1.0/3.0);

        window_ref = rescale_par;

        var_rescale = 1.0/(0.02201*pow(rescale_par,3.0) - 0.146*pow(rescale_par,2.0) + 0.3521*rescale_par - 0.09969);



        //var_rescale = 6.9541;
        //var_rescale = 7.1748;

    }




};

template<typename T>
void calc_sat_mean_y(Mesh_data<T>& input,const int offset){
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculates a O(1) recursive mean using SAT.
    //


    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    std::vector<T> temp_vec;
    temp_vec.resize(y_num,0);


    const int offset_n = offset;
    int i, k, index;
    float counter, temp, divisor = 2*offset_n + 1;

#pragma omp parallel for default(shared) private(i,k,counter,temp,index) firstprivate(temp_vec)
    for(int j = 0;j < z_num;j++){
        for(i = 0;i < x_num;i++){

            index = j*x_num*y_num + i*y_num;

            //first pass over and calculate cumsum
            temp = 0;
            for (k = 0; k < y_num;k++){
                temp += input.mesh[index + k];
                temp_vec[k] = temp;
            }

            input.mesh[index] = 0;
            //handling boundary conditions (LHS)
            for (k = 1; k <= (offset+1);k++){
                input.mesh[index + k] = -temp_vec[0]/divisor;
            }

            //second pass calculate mean
            for (k = offset + 1; k < y_num;k++){
                input.mesh[index + k] = -temp_vec[k - offset - 1]/divisor;
            }


            //second pass calculate mean
            for (k = 0; k < (y_num-offset);k++){
                input.mesh[index + k] += temp_vec[k + offset]/divisor;
            }


            counter = 0;
            //handling boundary conditions (RHS)
            for (k = ( y_num - offset); k < (y_num);k++){
                counter++;
                input.mesh[index + k]*= divisor;
                input.mesh[index + k]+= temp_vec[y_num-1];
                input.mesh[index + k]*= 1.0/(divisor - counter);
            }

            //handling boundary conditions (LHS), need to rehandle the boundary
            for (k = 1; k < (offset + 1);k++){
                input.mesh[index + k] *= divisor/(1.0*k + offset_n);
            }
            //end point boundary condition
            input.mesh[index] *= divisor/(offset_n+1);
        }
    }



}

template<typename T>
void calc_sat_mean_x(Mesh_data<T>& input,const int offset){
    // The same, but in place

    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    std::vector<T> temp_vec;
    temp_vec.resize(y_num*(2*offset + 1),0);

    int i,k;
    float temp;
    int index_modulo, previous_modulo, current_index, jxnumynum;

#pragma omp parallel for default(shared) private(i,k,temp,index_modulo, previous_modulo, current_index, jxnumynum) \
        firstprivate(temp_vec)
    for(int j = 0; j < z_num; j++) {

        jxnumynum = j * x_num * y_num;

        //prefetching

        for(k = 0; k < y_num ; k++){
            // std::copy ?
            temp_vec[k] = input.mesh[jxnumynum + k];
        }

        for(i = 1; i < 2 * offset + 1; i++) {
            for(k = 0; k < y_num; k++) {
                temp_vec[i*y_num + k] = input.mesh[jxnumynum + i*y_num + k] + temp_vec[(i-1)*y_num + k];
            }
        }

        // LHS boundary

        for(i = 0; i < offset + 1; i++){
            for(k = 0; k < y_num; k++) {
                input.mesh[jxnumynum + i * y_num + k] = (temp_vec[(i + offset) * y_num + k]) / (i + offset + 1);
            }
        }

        // middle

        current_index = offset + 1;

        for(i = offset + 1; i < x_num - offset; i++){
            // the current cumsum
            index_modulo = (current_index + offset) % (2*offset + 1); // current_index - offset - 1
            previous_modulo = (current_index + offset - 1) % (2*offset + 1); // the index of previous cumsum

            for(k = 0; k < y_num; k++) {
                temp = input.mesh[jxnumynum + (i + offset)*y_num + k] + temp_vec[previous_modulo*y_num + k];
                input.mesh[jxnumynum + i*y_num + k] = (temp - temp_vec[index_modulo*y_num + k]) /
                        (2*offset + 1);
                temp_vec[index_modulo*y_num + k] = temp;
            }

            current_index = (current_index + 1) % (2*offset + 1);
        }

        // RHS boundary
        current_index = (current_index + offset) % (2*offset + 1);

        for(i = x_num - offset; i < x_num; i++){
            for(k = 0; k < y_num; k++){
                input.mesh[jxnumynum + i*y_num + k] = (temp_vec[index_modulo*y_num + k] -
                        temp_vec[current_index*y_num + k]) / (x_num - i + offset);
            }

            current_index = (current_index + 1) % (2*offset + 1);
        }
    }


}


template<typename T>
void calc_sat_mean_z(Mesh_data<T>& input,const int offset) {

    // The same, but in place

    const int z_num = input.z_num;
    const int x_num = input.x_num;
    const int y_num = input.y_num;

    std::vector<T> temp_vec;
    temp_vec.resize(y_num*(2*offset + 1),0);

    int j,k;
    float temp;
    int index_modulo, previous_modulo, current_index, iynum;
    int xnumynum = x_num * y_num;

#pragma omp parallel for default(shared) private(j,k,temp,index_modulo, previous_modulo, current_index, iynum) \
        firstprivate(temp_vec)
    for(int i = 0; i < x_num; i++) {

        iynum = i * y_num;

        //prefetching

        for(k = 0; k < y_num ; k++){
            // std::copy ?
            temp_vec[k] = input.mesh[iynum + k];
        }

        for(j = 1; j < 2 * offset + 1; j++) {
            for(k = 0; k < y_num; k++) {
                temp_vec[j*y_num + k] = input.mesh[j * xnumynum + iynum + k] + temp_vec[(j-1)*y_num + k];
            }
        }

        // LHS boundary

        for(j = 0; j < offset + 1; j++){
            for(k = 0; k < y_num; k++) {
                input.mesh[j * xnumynum + iynum + k] = (temp_vec[(j + offset)*y_num + k]) / (j + offset + 1);
            }
        }

        // middle

        current_index = offset + 1;

        for(j = offset + 1; j < z_num - offset; j++){

            index_modulo = (current_index + offset) % (2*offset + 1); // current_index - offset - 1
            previous_modulo = (current_index + offset - 1) % (2*offset + 1); // the index of previous cumsum

            for(k = 0; k < y_num; k++) {
                // the current cumsum
                temp = input.mesh[(j + offset) * xnumynum + iynum + k] + temp_vec[previous_modulo*y_num + k];
                input.mesh[j * xnumynum + iynum + k] = (temp - temp_vec[index_modulo*y_num + k]) /
                        (2*offset + 1);
                temp_vec[index_modulo*y_num + k] = temp;
            }

            current_index = (current_index + 1) % (2*offset + 1);
        }

        // RHS boundary
        current_index = (current_index + offset) % (2*offset + 1);

        for(j = z_num - offset; j < z_num; j++){
            for(k = 0; k < y_num; k++){
                input.mesh[j * xnumynum + iynum + k] = (temp_vec[index_modulo*y_num + k] -
                                                        temp_vec[current_index*y_num + k]) / (z_num - j + offset);
            }

            current_index = (current_index + 1) % (2*offset + 1);
        }
    }

}


template<typename T>
void calc_abs_diff(Mesh_data<T>& input_image,Mesh_data<T>& var){
    //
    //  Bevan Cheeseman 2016
    //
    //

    const int z_num = input_image.z_num;
    const int x_num = input_image.x_num;
    const int y_num = input_image.y_num;

    int i,k;

#pragma omp parallel for default(shared) private(i,k)
    for(int j = 0;j < z_num;j++){

        for(i = 0;i < x_num;i++){

            for (k = 0; k < (y_num);k++){
                var.mesh[j*x_num*y_num + i*y_num + k] = std::abs(var.mesh[j*x_num*y_num + i*y_num + k] - input_image.mesh[j*x_num*y_num + i*y_num + k]);
            }

        }
    }


}
template<typename T>
void intensity_th(Mesh_data<T>& input_image,Mesh_data<T>& var,const float threshold,float max_th_input = 60000){
    //
    //  Bevan Cheeseman 2016
    //
    //

    const int z_num = var.z_num;
    const int x_num = var.x_num;
    const int y_num = var.y_num;

    const int z_num_i = input_image.z_num;
    const int x_num_i = input_image.x_num;
    const int y_num_i = input_image.y_num;

    const float max_th = max_th_input;

    int i,k;

#pragma omp parallel for default(shared) private(i,k)
    for(int j = 0;j < z_num;j++){

        for(i = 0;i < x_num;i++){

            for (k = 0; k < (y_num);k++){

                if(input_image.mesh[j*x_num_i*y_num_i + i*y_num_i + k] < threshold){
                    var.mesh[j*x_num*y_num + i*y_num + k] = max_th;
                }

            }

        }
    }


}

template<typename T>
void rescale_var_and_threshold(Mesh_data<T>& var,const float var_rescale,Part_rep& p_rep){
    //
    //  Bevan Cheeseman 2016
    //
    //

    const int z_num = var.z_num;
    const int x_num = var.x_num;
    const int y_num = var.y_num;
    const float max_th = 60000.0;

    int i,k;
    float rescaled;

#pragma omp parallel for default(shared) private(i,k,rescaled)
    for(int j = 0;j < z_num;j++){

        for(i = 0;i < x_num;i++){

            for (k = 0; k < (y_num);k++){

                float rescaled = var.mesh[j*x_num*y_num + i*y_num + k] * var_rescale;
                if(rescaled < p_rep.pars.var_th_max){
                    rescaled = max_th;
                }
                if(rescaled < p_rep.pars.var_th){
                    rescaled = p_rep.pars.var_th;
                }
                var.mesh[j*x_num*y_num + i*y_num + k] = rescaled;
            }

        }
    }

}

template<typename T>
void get_variance_2D(Part_rep &p_rep, Mesh_data<T> &input_image, Mesh_data<T> &var){
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculates the local variance using recursive SAT
    //

    Part_timer timer;


    // first down sample the image by 2, then calculate...
    down_sample(input_image,var,
                [](T x, T y) { return x+y; },
                [](T x) { return x * (1.0/8.0); });

    // copy constructor
    Mesh_data<T> temp = var;

    Map_calc_cpu calc_map(0,p_rep.pars);

    calc_map.set_up_var_filters_2D();

    int win_y = ceil((calc_map.var_window_size1 - 1)/4.0);
    int win_x = ceil((calc_map.var_window_size2 - 1)/4.0);
    int win_z = ceil((calc_map.var_window_size3 - 1)/4.0);

    //Perform first spatial average output to var

    debug_write(temp,"temp");

    timer.start_timer("calc_sat_mean_y");

    calc_sat_mean_y(var,win_y);

    timer.stop_timer();


    timer.start_timer("calc_sat_mean_x");


    calc_sat_mean_x(var,win_x);

    timer.stop_timer();




    timer.start_timer("calc_abs_diff");


    //calculate abs and subtract from original
    calc_abs_diff(temp,var);

    timer.stop_timer();
    //Second spatial average
    calc_sat_mean_y(var,win_y);
    calc_sat_mean_x(var,win_x);



    //if needed threshold the results
    if(p_rep.pars.I_th > 0) {
        intensity_th(temp, var,
                     p_rep.pars.I_th);
    }

    timer.start_timer("rescale_var_and_threshold");

    //rescaling the variance estimate
    rescale_var_and_threshold( var,calc_map.var_rescale,p_rep);

    timer.stop_timer();


}

template<typename T>
void get_variance_3D(Part_rep &p_rep, Mesh_data<T> &input_image, Mesh_data<T> &var){
    //
    //  Bevan Cheeseman 2016
    //
    //  Calculates the local variance using recursive SAT
    //

    Part_timer timer;


    // first down sample the image by 2, then calculate...
    down_sample(input_image,var,
                [](T x, T y) { return x+y; },
                [](T x) { return x * (1.0/8.0); });

    // copy constructor
    Mesh_data<T> temp = var;

    Map_calc_cpu calc_map(0,p_rep.pars);

    calc_map.set_up_var_filters_3D();



    int win_y = ceil((calc_map.var_window_size1 - 1)/4.0);
    int win_x = ceil((calc_map.var_window_size2 - 1)/4.0);
    int win_z = ceil((calc_map.var_window_size3 - 1)/4.0);

    int win_y2 = ceil((calc_map.var_window_size1 - 1)/4.0);
    int win_x2 = ceil((calc_map.var_window_size2 - 1)/4.0);
    int win_z2 = ceil((calc_map.var_window_size3 - 1)/4.0);

    if(p_rep.pars.padd_dims.size() == 6) {

        win_y = p_rep.pars.padd_dims[0];
        win_x = p_rep.pars.padd_dims[1];
        win_z = p_rep.pars.padd_dims[2];

        win_y2 = p_rep.pars.padd_dims[3];
        win_x2 = p_rep.pars.padd_dims[4];
        win_z2 = p_rep.pars.padd_dims[5];

    }

    //Perform first spatial average output to var

    timer.start_timer("calc_sat_mean_y");

    calc_sat_mean_y(var,win_y);

    timer.stop_timer();


    timer.start_timer("calc_sat_mean_x");


    calc_sat_mean_x(var,win_x);

    timer.stop_timer();

    timer.start_timer("calc_sat_mean_z");

    calc_sat_mean_z(var,win_z);

    timer.stop_timer();



    timer.start_timer("calc_abs_diff");


    //calculate abs and subtract from original
    calc_abs_diff(temp,var);

    timer.stop_timer();
    //Second spatial average
    calc_sat_mean_y(var,win_y2);
    calc_sat_mean_x(var,win_x2);
    calc_sat_mean_z(var,win_z2);

    //if needed threshold the results
    if(p_rep.pars.I_th > 0) {
        intensity_th(temp, var,
        p_rep.pars.I_th);
    }

    timer.start_timer("rescale_var_and_threshold");

    //rescaling the variance estimate
    rescale_var_and_threshold( var,calc_map.var_rescale,p_rep);

    timer.stop_timer();


}

#endif
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

        std::vector<int> windows = {1,2,3,4,5,6,7,8,9,10,13,16,20,30};

        int window_ind_1 = 0;
        int window_ind_2 = 0;

        int curr_dist_1 = 99;
        int curr_dist_2 = 99;

        for (int i = 0; i < windows.size(); ++i) {
            if(abs(windows[i] - par.padd_dims[0]) < curr_dist_1){
                window_ind_1 = i;
                curr_dist_1 = abs(windows[i] - par.padd_dims[0]);
            }

            if(abs(windows[i] - par.padd_dims[3]) < curr_dist_2){
                window_ind_2 = i;
                curr_dist_2 = abs(windows[i] - par.padd_dims[3]);
            }


        }


        std::vector<std::vector<std::vector<float>>> rescale_store = {{{{21.8712},{11.3475},{8.2029},{6.8449},{6.1969},{5.7429},{5.1441},{5.2386},{5.2968},{5.3517},{4.9936},{4.9581},{4.9594},{4.7094}},{{24.1372},{12.5505},{8.7225},{6.6269},{6.3298},{5.8459},{5.5231},{5.3945},{5.1393},{5.1417},{5.0151},{5.2201},{5.1307},{4.954}},{{29.8184},{14.3049},{9.4888},{7.7286},{6.2327},{5.7355},{5.6961},{5.3448},{5.1032},{5.489},{5.4648},{5.4845},{5.2237},{5.2158}},{{35.3079},{16.6115},{11.0758},{8.0459},{7.0799},{6.043},{6.1085},{6.0004},{5.7718},{5.561},{5.7644},{5.9296},{5.5886},{5.8014}},{{41.0414},{18.8932},{12.2474},{9.9592},{8.7891},{7.0745},{6.5979},{6.5939},{6.2863},{6.5275},{6.5264},{6.2005},{6.8697},{7.0316}},{{45.2437},{21.9481},{14.047},{10.9749},{9.0858},{7.2288},{7.0953},{7.3146},{7.3465},{7.0436},{7.6394},{7.3948},{8.3803},{8.2901}},{{50.4008},{24.4009},{16.01},{12.8603},{10.8298},{9.6412},{8.7187},{7.9302},{7.8932},{8.2912},{9.1262},{8.6032},{10.123},{9.4156}},{{63.187},{27.4839},{18.815},{14.6712},{11.8884},{11.7012},{11.1197},{10.2848},{9.9433},{9.5193},{9.432},{9.8971},{10.5318},{10.4675}},{{70.1255},{32.0089},{24.1094},{17.3636},{13.7413},{12.5908},{11.0803},{12.1848},{10.4079},{12.0014},{10.6457},{12.234},{12.7139},{11.7483}},{{75.6331},{38.7046},{24.206},{18.623},{14.9424},{12.5374},{13.744},{13.0704},{11.9171},{12.0562},{12.2139},{12.3602},{13.6859},{15.1146}},{{132.8905},{60.8608},{37.4598},{25.5599},{23.9681},{18.8553},{16.5947},{14.2513},{14.5883},{16.8338},{14.4524},{16.2667},{16.8578},{16.0348}},{{140.853},{62.4615},{48.865},{31.1825},{27.2167},{24.1998},{25.465},{18.0199},{19.3349},{18.0585},{18.7379},{16.9801},{18.6853},{22.4217}},{{156.0072},{82.4835},{50.6702},{40.853},{32.0573},{26.8327},{28.0142},{21.5992},{22.7753},{21.9197},{22.7164},{19.7104},{18.0837},{21.5494}},{{187.0682},{90.8115},{62.1232},{45.8227},{37.0191},{38.1417},{32.0205},{30.9501},{24.4604},{25.1366},{27.1657},{23.716},{21.7931},{26.3089}}},{{{33.7668},{14.0021},{8.7773},{6.6154},{5.4836},{5.1193},{4.5587},{4.5053},{4.317},{4.3235},{4.3884},{4.3794},{4.5082},{4.4521}},{{35.0934},{14.8065},{9.1291},{6.5686},{5.7626},{5.1767},{4.747},{4.5671},{4.4033},{4.4642},{4.4149},{4.4627},{4.4794},{4.3621}},{{37.5025},{15.5716},{9.6533},{7.1924},{5.748},{5.2072},{4.7447},{4.5692},{4.3077},{4.4038},{4.4433},{4.5542},{4.5398},{4.417}},{{39.9126},{16.3702},{10.029},{7.7536},{6.3696},{5.2349},{5.0573},{4.9427},{4.702},{4.6287},{4.7911},{4.8594},{4.5976},{4.5292}},{{45.0468},{18.2406},{11.2004},{8.1869},{6.8266},{5.7521},{5.3952},{5.0871},{4.9473},{4.8062},{5.1419},{4.8676},{4.963},{5.09}},{{49.3254},{20.5647},{11.9391},{8.2251},{7.1618},{6.46},{5.8841},{5.5126},{5.5393},{5.5232},{5.1008},{5.0228},{5.1104},{5.5794}},{{55.2265},{21.9659},{13.7618},{9.4171},{7.9478},{7.0581},{6.892},{5.8256},{6.1395},{5.8102},{5.9404},{5.4464},{5.8324},{5.4376}},{{59.0427},{26.1059},{15.8714},{11.6133},{8.2072},{7.6587},{7.4989},{6.9643},{6.8454},{6.426},{6.1583},{7.1597},{6.5818},{7.3287}},{{70.3604},{29.9269},{18.5125},{11.2888},{10.4531},{8.5162},{8.8686},{6.8441},{6.6747},{6.3439},{6.8237},{6.9559},{6.5687},{7.6255}},{{73.7245},{32.2119},{19.2399},{15.2922},{10.5251},{10.596},{9.4189},{8.5336},{8.1348},{7.9511},{7.9778},{7.6018},{8.7104},{7.8265}},{{90.8949},{43.7023},{25.3167},{20.1892},{16.1558},{14.3434},{12.0804},{10.5268},{10.9515},{9.9447},{8.8738},{9.8977},{9.5741},{10.5934}},{{101.4286},{50.3042},{28.2661},{22.3494},{16.9847},{16.3889},{14.6021},{12.0777},{11.0127},{12.272},{9.4762},{10.6808},{12.2806},{10.6785}},{{139.9784},{57.1585},{33.5377},{29.3547},{21.0713},{19.0745},{16.2923},{13.6059},{12.5611},{13.7547},{10.407},{12.7631},{10.4623},{11.3659}},{{152.5909},{65.8075},{38.6669},{30.0082},{22.7189},{19.4086},{19.3116},{16.221},{14.7568},{17.3114},{14.9049},{15.3965},{13.7937},{15.104}}},{{{54.2393},{20.2722},{11.8921},{8.5984},{7.3198},{6.3027},{5.8096},{5.2957},{5.2429},{5.1952},{5.1171},{4.763},{4.9778},{5.0131}},{{55.1351},{21.2293},{12.7899},{9.169},{7.0199},{6.1514},{5.5453},{5.2323},{5.2304},{5.1579},{4.931},{5.0916},{5.0293},{4.6993}},{{57.6869},{21.4265},{13.0984},{9.2239},{7.333},{6.429},{5.6666},{5.447},{5.1472},{5.2011},{4.8253},{4.8738},{4.9605},{4.7907}},{{61.4797},{24.0969},{13.7219},{9.7664},{7.2963},{6.7211},{6.0614},{5.4541},{5.2081},{5.3464},{5.2094},{5.303},{5.408},{4.9118}},{{67.5115},{25.2493},{15.2233},{10.4532},{8.206},{6.7931},{6.2288},{6.02},{5.9083},{5.2938},{5.4893},{5.523},{5.7606},{5.4829}},{{74.7121},{28.6316},{16.7978},{11.4987},{9.4723},{7.8686},{6.5029},{6.1657},{5.5739},{5.8073},{5.6572},{5.7573},{5.6201},{5.7056}},{{78.7771},{30.7183},{17.9938},{12.0055},{10.5571},{7.6292},{7.3024},{7.0236},{6.0949},{6.4093},{6.0596},{6.3011},{6.3684},{6.3608}},{{85.4553},{34.0796},{20.0702},{13.0713},{11.2645},{9.7722},{7.8795},{7.3611},{6.9251},{6.537},{6.9716},{7.0271},{7.1215},{7.4538}},{{97.133},{40.1325},{20.8241},{15.678},{12.7715},{10.555},{8.9434},{8.5558},{7.9645},{7.5857},{6.5518},{7.857},{7.7866},{8.1651}},{{91.5077},{41.6946},{21.0388},{16.1652},{13.5588},{10.6568},{9.1117},{9.3844},{8.5619},{8.3649},{8.8288},{8.5856},{7.9046},{8.1927}},{{120.8429},{51.3194},{31.7402},{23.537},{16.7056},{13.6521},{12.4574},{9.988},{11.3712},{10.6029},{9.412},{9.5861},{10.1486},{10.0801}},{{143.7078},{65.8955},{38.834},{27.9511},{19.9754},{15.9195},{15.768},{11.4972},{11.1785},{13.4465},{10.6383},{10.8762},{10.9998},{11.817}},{{180.7034},{71.2881},{37.5171},{33.5842},{24.6971},{17.4467},{15.6515},{16.5885},{12.1953},{13.0271},{12.1092},{11.5799},{12.0188},{11.6296}},{{168.5441},{83.7475},{56.42},{38.7432},{31.7104},{22.8922},{24.1318},{20.3337},{16.6357},{15.8415},{14.2212},{14.343},{12.8106},{11.5359}}},{{{71.8792},{27.5171},{14.9127},{10.6669},{8.5157},{7.2579},{6.5554},{6.1093},{5.6624},{5.4644},{5.0386},{5.1248},{4.8711},{4.7247}},{{73.0386},{28.2711},{15.9644},{10.3504},{8.4519},{7.2885},{6.386},{5.885},{5.6525},{5.4635},{4.9849},{5.2202},{5.0113},{4.6001}},{{74.0069},{28.2212},{15.6298},{10.8128},{8.3402},{7.1005},{6.2887},{5.9199},{5.5696},{5.2618},{5.0551},{5.0693},{4.9274},{4.5307}},{{76.9933},{29.7364},{16.6376},{11.3603},{8.6398},{7.3662},{6.6202},{6.0687},{5.6338},{5.429},{5.2384},{5.2461},{5.0722},{5.0701}},{{82.3437},{30.2805},{17.9918},{12.3361},{9.3971},{7.6048},{6.5505},{6.303},{5.5783},{5.4186},{5.1996},{5.3075},{5.2776},{5.1803}},{{84.7133},{33.8577},{18.9271},{12.5927},{9.9455},{7.9207},{7.5271},{6.518},{6.2081},{5.9971},{5.5923},{6.105},{5.5349},{5.7909}},{{94.9706},{36.8767},{20.8195},{14.3431},{11.0164},{8.8307},{7.5038},{7.1595},{6.652},{6.3118},{6.822},{6.3429},{5.631},{5.6354}},{{102.3097},{39.5874},{22.3408},{14.9718},{11.2008},{10.2291},{8.7684},{7.8621},{6.9994},{6.9984},{6.3965},{6.29},{6.0262},{6.7757}},{{107.8217},{41.7518},{23.7995},{16.932},{11.7531},{11.5099},{8.9894},{7.9003},{8.1109},{7.1914},{7.0177},{7.2227},{6.5925},{7.0887}},{{118.0902},{50.9053},{30.931},{19.6455},{15.2693},{11.6706},{9.7404},{9.007},{8.0203},{7.882},{7.3797},{7.1701},{7.2681},{6.6861}},{{152.0845},{56.3437},{35.4982},{24.2324},{15.7499},{13.6684},{13.3628},{11.3062},{10.7142},{9.8594},{10.5512},{8.336},{7.9867},{9.0251}},{{144.3112},{68.6853},{39.8336},{27.6967},{19.5441},{16.7964},{14.1612},{13.2136},{10.8882},{10.7532},{10.5711},{9.9212},{9.2986},{10.5423}},{{189.5481},{80.3813},{48.6339},{27.0163},{27.3222},{20.3803},{18.2824},{15.7282},{14.3775},{13.8993},{11.3588},{11.1365},{10.7046},{11.193}},{{223.9542},{93.0996},{60.7837},{41.6107},{31.4053},{25.4162},{22.7699},{20.4844},{19.5185},{15.13},{14.4715},{14.7544},{13.3189},{11.7747}}},{{{82.8056},{30.8108},{18.289},{11.7798},{9.1113},{7.4379},{6.612},{6.0334},{5.2758},{5.3367},{4.7213},{4.6915},{4.6513},{4.1915}},{{83.5994},{32.0241},{17.1673},{12.2708},{9.1051},{7.4807},{6.6563},{5.9806},{5.6212},{5.2109},{4.8108},{4.5045},{4.5061},{4.1352}},{{82.7106},{32.1886},{16.836},{12.1303},{9.2196},{7.4234},{6.3706},{5.8045},{5.5789},{5.2714},{4.7758},{4.4758},{4.4593},{4.2361}},{{84.9198},{34.1878},{17.9338},{12.1528},{8.9265},{7.5239},{6.4954},{6.0363},{5.5666},{5.3277},{4.8908},{4.7504},{4.3457},{4.3051}},{{90.174},{34.7671},{18.1992},{12.7934},{9.458},{7.7951},{6.6642},{5.9915},{5.3644},{4.9833},{4.9493},{4.9247},{4.3523},{4.9689}},{{95.0982},{36.577},{19.5229},{13.3463},{9.8852},{8.0088},{6.6793},{6.5064},{5.9957},{5.3342},{5.0747},{5.1925},{4.7089},{4.9096}},{{97.86},{38.2271},{20.7056},{15.1647},{10.6916},{8.7883},{7.7685},{6.6923},{6.1778},{5.8452},{5.7308},{4.9987},{5.0182},{4.9739}},{{109.115},{41.8334},{21.4906},{16.3975},{11.5029},{9.0037},{8.6838},{7.0601},{6.8847},{6.0001},{5.8077},{5.2308},{5.6235},{5.2638}},{{111.4722},{42.822},{24.5319},{15.0039},{12.5979},{10.3923},{8.9172},{8.1247},{6.9851},{6.7599},{6.1624},{5.648},{5.7217},{5.4115}},{{115.1866},{48.6417},{28.0892},{17.7574},{13.2283},{9.9637},{9.5767},{7.7822},{7.0152},{7.1892},{6.5053},{6.4699},{6.1909},{6.2592}},{{132.6006},{58.6056},{31.8077},{24.6168},{16.3374},{13.1421},{11.4916},{9.6892},{9.7928},{8.2016},{7.1258},{7.7645},{7.566},{6.8699}},{{147.6585},{66.8003},{38.5095},{25.1516},{18.2773},{16.2663},{13.5329},{11.4167},{11.7138},{9.1254},{9.4503},{8.1261},{7.8743},{8.3974}},{{191.014},{78.0731},{44.456},{31.5084},{25.3488},{20.1063},{14.5143},{12.599},{12.8993},{11.7269},{10.3138},{9.9775},{9.4987},{9.0874}},{{195.0718},{94.0806},{62.4445},{48.725},{30.894},{25.1178},{23.1181},{21.3263},{17.4186},{16.0023},{14.5871},{12.5957},{12.6446},{10.5473}}},{{{82.8396},{32.3602},{19.1028},{11.9705},{9.8451},{7.6936},{6.4696},{5.589},{5.3673},{4.879},{4.2975},{4.1953},{3.7708},{3.0576}},{{89.8308},{34.122},{18.3433},{11.9207},{8.8617},{7.211},{6.6638},{5.7835},{5.2102},{4.9613},{4.2824},{4.1317},{3.7981},{3.6078}},{{88.0161},{32.2863},{18.8203},{12.6632},{9.0364},{7.6453},{6.2469},{5.567},{5.3684},{4.9305},{4.6857},{4.1221},{3.8413},{3.3936}},{{87.402},{34.1287},{18.4087},{12.0995},{9.7922},{6.9253},{6.458},{5.6456},{5.468},{4.9387},{4.4142},{4.0822},{3.8719},{3.2733}},{{85.9661},{35.8293},{19.6734},{12.5878},{9.3574},{6.9738},{6.5443},{5.4182},{5.5343},{5.107},{4.761},{4.0751},{3.9195},{3.6786}},{{91.9479},{38.2158},{20.0227},{12.0302},{9.751},{8.085},{6.5097},{6.0243},{5.2467},{5.2866},{4.6107},{4.5151},{4.0979},{3.5372}},{{100.0497},{36.7103},{20.8488},{12.9523},{10.5386},{7.8514},{7.1111},{5.9954},{5.487},{5.2533},{4.5158},{4.2925},{4.272},{3.889}},{{104.3862},{40.9694},{22.1305},{14.307},{10.8098},{8.3429},{7.8029},{6.5284},{5.7903},{5.4125},{4.4683},{4.6804},{4.3186},{4.1805}},{{110.0215},{44.6417},{23.4827},{15.1913},{11.0129},{8.9073},{7.5668},{7.1556},{6.5138},{5.7692},{5.5239},{4.755},{4.572},{4.193}},{{110.3175},{47.3018},{24.6174},{16.1985},{11.8728},{9.9088},{8.5181},{7.0532},{6.6338},{5.9667},{5.3681},{5.0904},{4.7482},{4.6217}},{{125.618},{52.1555},{26.9539},{19.7156},{14.9998},{11.9354},{10.3845},{8.7063},{8.1414},{7.0253},{6.2466},{6.6572},{5.0704},{5.5302}},{{142.5428},{66.388},{37.7479},{23.1064},{17.9055},{15.5402},{12.1223},{10.0439},{9.0733},{8.8042},{7.3181},{7.1607},{6.0396},{6.0969}},{{163.4435},{68.127},{44.9849},{28.3884},{19.4734},{17.2209},{14.2943},{12.8593},{10.7612},{9.906},{8.7165},{8.1156},{7.6136},{6.2718}},{{177.7164},{110.7729},{60.6643},{44.0968},{32.4216},{24.0201},{21.8686},{22.621},{18.2108},{17.1753},{12.4437},{11.7871},{10.9015},{9.3194}}}};

        //(PSF,window2,window1) with 0 indexing

        int psf_ind = std::max(((float) (round(par.psfx/par.dx) - 1)),((float)0.0f));

        psf_ind = std::min(psf_ind,5);

        var_rescale = rescale_store[psf_ind][window_ind_2][window_ind_1];

        //var_rescale = 1;

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
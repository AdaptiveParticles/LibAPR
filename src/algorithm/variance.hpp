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


        std::vector<std::vector<std::vector<float>>> rescale_store = {{{{22.1341},{10.7272},{8.4599},{7.4075},{6.7677},{6.3736},{5.7483},{5.8956},{5.2944},{5.6046},{5.2514},{5.0158},{5.2669},{4.6109}},{{27.1515},{12.0975},{9.4366},{7.4239},{6.9205},{5.6849},{5.1602},{6.0029},{5.6898},{5.8345},{5.1524},{5.1948},{5.8673},{5.7513}},{{29.4627},{15.8537},{9.6843},{8.3647},{6.9943},{6.6692},{6.3145},{5.7784},{5.1064},{5.7305},{6.0878},{6.1164},{5.3947},{6.3264}},{{38.7984},{18.1885},{13.1586},{8.8905},{7.8875},{7.1447},{5.528},{6.2221},{6.3296},{6.2671},{6.7953},{6.0299},{5.4832},{7.1419}},{{46.8447},{19.529},{13.1086},{9.3571},{9.3253},{7.9606},{7.0095},{6.6039},{5.7789},{5.9196},{7.6258},{5.9254},{7.8243},{6.9332}},{{47.346},{18.7463},{14.6966},{11.6527},{11.0271},{7.7114},{8.3285},{7.512},{7.8781},{8.3528},{7.9861},{8.4467},{7.6817},{8.1957}},{{55.4137},{30.4497},{16.5702},{15.3488},{12.4887},{10.6769},{8.5109},{9.2872},{7.9521},{7.1643},{10.2575},{8.9625},{7.2073},{10.2709}},{{69.2428},{33.8699},{18.3317},{13.7272},{14.2662},{10.2674},{9.8524},{9.9586},{8.2536},{10.097},{10.1831},{10.3933},{8.5894},{11.2861}},{{59.173},{34.8976},{23.0065},{20.6407},{18.452},{13.5364},{12.8864},{11.6075},{12.4616},{11.6157},{10.315},{13.2381},{9.2214},{13.8052}},{{82.7881},{51.4311},{28.8984},{21.1391},{21.1125},{16.0706},{14.1677},{17.2966},{13.5951},{14.0022},{12.6532},{9.812},{14.2146},{16.4416}},{{135.1822},{43.5409},{45.0016},{33.8689},{14.99},{19.8864},{15.9427},{17.5091},{18.7969},{16.559},{21.5635},{17.3592},{17.9955},{24.6541}},{{116.6189},{65.3223},{37.7854},{45.5298},{30.0113},{37.8003},{23.0122},{17.3266},{22.0513},{20.3009},{19.1271},{19.0797},{20.8943},{19.4972}},{{124.2206},{58.971},{64.3689},{44.7286},{26.996},{30.5738},{25.9192},{31.4525},{23.2633},{23.0336},{23.0555},{19.6951},{18.1947},{29.6539}},{{206.404},{72.1232},{67.8921},{61.9302},{56.0803},{38.9458},{35.1053},{27.9257},{27.683},{29.0236},{22.8653},{20.9668},{23.9519},{27.4798}}},{{{34.0956},{13.7486},{8.921},{6.0269},{5.7974},{5.068},{4.6797},{4.7997},{4.0517},{4.4852},{4.5691},{4.3302},{4.2325},{4.2418}},{{35.1283},{13.7125},{9.2874},{6.9333},{5.8781},{5.1451},{4.827},{4.1459},{4.528},{4.4207},{4.5094},{4.4148},{4.5999},{4.5841}},{{34.8954},{15.4161},{9.7378},{6.6931},{5.9903},{5.2966},{4.7744},{4.2595},{4.3939},{4.6288},{4.8407},{4.2428},{4.7064},{4.2423}},{{38.9458},{16.6195},{10.5179},{7.7781},{6.14},{5.9245},{5.1647},{4.6352},{4.8925},{4.712},{5.0419},{4.6201},{4.7422},{4.5615}},{{43.6769},{17.6705},{10.9827},{8.0963},{6.6923},{6.1559},{5.7003},{5.4521},{4.9638},{5.0267},{5.5251},{5.0812},{5.0562},{4.928}},{{38.5334},{18.0371},{12.9159},{9.3561},{7.5231},{6.0412},{6.2639},{5.8621},{5.175},{5.7424},{5.8003},{4.5084},{5.5582},{4.3337}},{{46.7325},{22.2289},{13.7821},{11.0412},{8.0065},{6.9437},{6.9975},{6.8936},{5.6917},{6.3882},{6.5471},{6.3391},{5.6761},{5.6301}},{{64.8187},{26.1732},{15.5811},{9.102},{9.3435},{9.0741},{7.3805},{6.565},{6.47},{6.5281},{5.7881},{5.187},{6.9064},{6.0206}},{{52.1329},{26.8127},{16.8263},{13.4387},{8.5881},{9.2889},{7.8075},{7.7328},{7.6422},{5.63},{7.6743},{6.6875},{8.0889},{7.0043}},{{71.2941},{32.4913},{22.9727},{15.71},{11.9633},{11.4444},{9.9392},{7.8004},{7.456},{8.5639},{7.7765},{9.3823},{8.7279},{7.914}},{{87.7357},{37.1896},{27.2362},{18.7841},{12.2499},{15.8463},{12.4795},{10.5092},{12.4836},{10.7369},{12.0379},{9.9918},{11.7043},{9.2691}},{{86.9017},{44.6789},{30.8584},{26.8835},{17.2521},{9.0298},{10.2059},{13.3152},{9.5519},{11.9832},{9.4452},{11.8381},{11.5238},{13.3762}},{{86.012},{35.9734},{45.7545},{32.2907},{23.8879},{14.7273},{15.7767},{14.8546},{18.3297},{12.7811},{12.2491},{12.3923},{10.9555},{12.1976}},{{103.1811},{82.056},{42.7185},{32.3101},{26.1117},{15.7645},{16.5317},{13.8239},{13.3107},{12.6101},{16.4273},{17.1946},{17.2011},{16.2449}}},{{{53.9066},{20.4303},{12.1878},{9.0107},{7.519},{5.994},{5.7115},{5.235},{5.6024},{5.4422},{4.8929},{4.9429},{5.1146},{4.477}},{{50.9132},{21.07},{12.4241},{9.0103},{7.3904},{6.6217},{5.9419},{5.365},{5.3506},{4.6407},{5.2287},{4.518},{5.019},{4.786}},{{51.7423},{22.855},{13.0421},{9.8385},{7.9349},{6.1988},{5.9807},{5.6514},{5.0194},{5.2076},{5.4332},{4.6381},{4.6545},{5.1224}},{{55.9819},{23.0454},{12.6913},{9.6899},{6.7381},{7.0138},{6.1379},{5.7382},{5.4149},{5.7916},{5.4625},{5.09},{5.6176},{5.3173}},{{66.4472},{25.982},{14.905},{11.0642},{8.3666},{7.0616},{6.8309},{6.2094},{5.8003},{5.8675},{5.2523},{5.8157},{5.6569},{4.9352}},{{59.0862},{29.5162},{16.4301},{10.5241},{7.4444},{7.7961},{7.5817},{5.9986},{6.8619},{5.9015},{6.1929},{5.4575},{5.9689},{4.979}},{{75.0687},{32.512},{19.4697},{10.923},{9.6024},{7.3457},{7.5734},{7.4005},{6.0657},{7.1834},{5.7101},{6.5641},{6.6076},{7.4123}},{{59.4072},{33.3115},{17.064},{15.9129},{11.5189},{9.2616},{8.8868},{7.5995},{7.6447},{7.356},{7.8066},{5.285},{7.9866},{7.5302}},{{103.9628},{33.5942},{20.7467},{13.1011},{14.6599},{9.6728},{8.8752},{9.5014},{8.6198},{8.5316},{7.7363},{7.5056},{8.0346},{7.758}},{{82.7788},{35.8808},{26.9908},{19.6365},{15.1172},{12.2403},{9.1945},{10.0381},{9.7577},{9.0728},{8.4241},{9.4527},{8.8459},{8.6938}},{{126.8777},{57.6107},{29.0784},{22.2849},{15.9603},{12.4388},{16.4078},{10.9086},{10.9676},{10.2654},{9.6993},{10.1393},{11.2046},{9.6676}},{{116.8094},{61.3013},{48.4281},{23.9909},{22.8985},{21.6415},{16.4691},{14.5013},{11.4697},{15.8899},{9.3635},{10.9561},{13.6456},{11.9933}},{{198.6054},{69.9951},{50.8716},{41.1538},{24.4671},{22.4017},{21.4472},{16.1775},{10.46},{12.5546},{11.007},{10.3663},{12.4789},{11.0248}},{{163.1896},{71.8837},{46.3142},{32.4356},{25.2136},{23.4824},{15.8062},{17.3038},{18.6051},{18.0486},{13.5283},{16.6629},{11.8379},{12.5729}}},{{{71.2287},{29.4273},{15.4916},{11.3656},{9.3247},{7.8809},{7.0521},{6.518},{6.1248},{5.4318},{5.8803},{4.9008},{4.9367},{4.9581}},{{78.7496},{29.3704},{16.5242},{12.2956},{9.3376},{7.9641},{7.0142},{5.5482},{5.5513},{5.7242},{5.8488},{5.4936},{5.1461},{5.2881}},{{75.318},{30.0627},{17.8154},{12.5498},{9.1206},{8.141},{5.712},{6.4559},{6.1207},{5.5607},{5.5648},{5.5081},{5.0902},{5.0107}},{{76.4579},{31.8455},{17.8512},{11.8416},{9.7313},{7.3186},{5.7107},{6.4394},{6.669},{6.0782},{4.9381},{6.101},{5.2177},{5.0834}},{{89.4883},{34.1093},{17.4676},{12.331},{10.2915},{8.4218},{7.7488},{6.6905},{6.2192},{6.62},{5.5874},{5.6957},{5.8744},{5.2848}},{{85.0241},{35.7193},{22.1309},{12.9303},{11.7388},{9.604},{7.3456},{7.4661},{7.0828},{7.1805},{6.1494},{6.2863},{6.0322},{5.7604}},{{94.7371},{40.2817},{24.5586},{15.2748},{11.2814},{9.6202},{6.993},{8.8186},{8.0045},{6.6235},{7.2554},{7.411},{6.8217},{6.2892}},{{108.5113},{43.0657},{28.6859},{18.832},{11.3116},{10.3472},{9.0642},{7.8705},{7.2368},{7.9264},{7.7433},{7.1806},{7.8222},{7.2954}},{{93.7898},{43.3185},{28.1636},{15.8597},{11.8695},{8.8306},{9.5531},{6.9469},{7.5852},{8.744},{8.0974},{8.1741},{7.8598},{7.4394}},{{113.0281},{52.4502},{30.5313},{20.0941},{14.0366},{11.5113},{11.6419},{13.1239},{9.3603},{8.1144},{8.5367},{8.2703},{7.9276},{7.7726}},{{112.2077},{64.2642},{34.5971},{22.8582},{22.1313},{16.2347},{16.6225},{14.7425},{11.9736},{10.5842},{12.3801},{10.7812},{11.9228},{7.7619}},{{154.7442},{65.0444},{38.1896},{30.9245},{21.087},{16.662},{19.1345},{14.9156},{10.8514},{10.8315},{10.8654},{11.1686},{10.6401},{13.2264}},{{203.0087},{87.5359},{57.3583},{34.9574},{20.1434},{18.3287},{19.9266},{12.7569},{13.3424},{13.1069},{12.811},{12.4976},{10.0088},{11.1173}},{{167.9787},{85.5648},{68.0387},{41.8167},{34.2698},{31.3971},{24.3774},{22.7024},{24.0996},{18.8842},{14.0965},{16.4037},{12.664},{15.7549}}},{{{85.5163},{37.8455},{21.2197},{14.1069},{10.7748},{8.6188},{7.2526},{7.1487},{6.6262},{6.5336},{5.3929},{4.9181},{4.9395},{4.3447}},{{98.8773},{36.9656},{21.4329},{13.9988},{10.1102},{8.5694},{7.8686},{6.8639},{6.8433},{5.6703},{5.1825},{4.3916},{4.7006},{4.5048}},{{97.6612},{37.0275},{21.7639},{14.2406},{11.2192},{8.2323},{7.4799},{7.2592},{6.7853},{6.5963},{5.1501},{5.5197},{5.529},{4.9854}},{{100.8472},{40.0726},{22.5336},{14.4639},{10.1672},{9.5901},{7.4117},{6.9107},{6.644},{6.4351},{5.7433},{5.9856},{5.0446},{5.1273}},{{97.5477},{43.8973},{22.6218},{16.0012},{11.4455},{10.3553},{7.9678},{7.3089},{5.963},{6.7902},{6.5446},{6.0006},{6.6307},{5.0176}},{{104.771},{45.9085},{22.4576},{14.7865},{12.1136},{10.0555},{8.3089},{7.7474},{7.8843},{7.6018},{6.4482},{6.3128},{6.4038},{5.1828}},{{108.8113},{44.817},{26.4037},{18.3256},{13.9107},{11.2798},{9.8922},{9.1827},{8.1448},{7.6868},{7.2163},{6.7892},{6.5372},{6.9556}},{{96.434},{52.7984},{29.0675},{20.29},{12.8256},{10.8555},{11.5334},{8.2675},{7.9195},{7.5491},{6.2541},{6.0373},{6.7368},{5.8921}},{{127.5041},{52.6385},{31.986},{23.7632},{19.7447},{14.5399},{11.2111},{10.3881},{8.7805},{8.3423},{8.155},{6.9705},{7.1623},{6.5126}},{{127.4393},{60.3241},{36.5352},{24.899},{18.5792},{17.5704},{10.8918},{9.5355},{10.0843},{10.5894},{8.0324},{10.3214},{6.077},{7.8494}},{{151.4042},{74.35},{39.5004},{30.6685},{21.6606},{17.4931},{14.9958},{12.5449},{10.2357},{12.3239},{10.2485},{10.1251},{9.8071},{9.9791}},{{180.0001},{93.2944},{75.3555},{37.9852},{25.3393},{17.4267},{18.2217},{14.0766},{12.8283},{14.3663},{11.9858},{12.6424},{8.1456},{12.6791}},{{206.3426},{77.9534},{51.7925},{35.6894},{32.9117},{19.6114},{19.7948},{20.1717},{12.8536},{15.4548},{15.4605},{13.5446},{12.976},{12.5672}},{{284.2492},{140.7756},{92.8408},{68.0216},{33.9464},{36.7797},{33.273},{29.4653},{28.4274},{21.0246},{21.6262},{21.9979},{15.6343},{10.978}}},{{{100.2765},{40.6658},{22.5166},{15.8876},{11.8849},{9.3696},{7.1693},{6.924},{5.3803},{4.8142},{4.7468},{3.808},{3.981},{3.8654}},{{102.6856},{44.5771},{22.1179},{14.4077},{11.2203},{8.3801},{7.4633},{6.326},{6.288},{5.5668},{4.5885},{3.8742},{4.0636},{3.3556}},{{96.0067},{43.8433},{23.0055},{15.2262},{11.5528},{8.8113},{9.4028},{6.3972},{6.3039},{5.0847},{4.7505},{4.9456},{3.9835},{3.9524}},{{117.0312},{38.6059},{23.3474},{15.3512},{9.4405},{9.0471},{7.5024},{6.4632},{5.6381},{5.7985},{4.7238},{4.7656},{4.4496},{3.8186}},{{43.6014},{43.6014},{24.4907},{14.5308},{11.4531},{10.0961},{8.7854},{7.5357},{6.1355},{6.442},{4.757},{4.4155},{4.5006},{3.9338}},{{128.9444},{46.5314},{27.3412},{16.6536},{11.1643},{10.4389},{8.1132},{8.1354},{6.6618},{7.4732},{4.8917},{5.3468},{4.5887},{4.5344}},{{96.3896},{57.0438},{25.7986},{21.0297},{12.6315},{9.4289},{9.5462},{8.3361},{7.5249},{8.2726},{6.3575},{6.0292},{5.77},{5.0524}},{{137.9497},{52.3061},{30.9575},{19.8604},{13.9372},{14.1115},{12.077},{7.7474},{8.8489},{8.5591},{6.317},{4.9571},{5.1295},{3.7051}},{{145.2448},{61.9979},{32.5141},{20.0425},{16.8134},{10.3306},{11.0876},{8.5296},{9.532},{7.5515},{6.618},{7.5335},{6.5369},{6.7472}},{{157.3623},{54.6489},{36.8586},{24.89},{18.2836},{10.1631},{12.3121},{9.5373},{9.565},{10.3901},{8.2041},{6.458},{7.2421},{6.7746}},{{149.3257},{95.9255},{51.0052},{38.0189},{29.6291},{21.2164},{14.2796},{11.3299},{11.4976},{11.7419},{9.6285},{10.1665},{12.376},{8.2337}},{{182.8687},{100.9566},{46.4212},{44.4595},{22.3603},{24.8464},{15.1743},{19.0367},{20.5776},{12.9326},{12.325},{7.3321},{10.5403},{13.1214}},{{146.4793},{118.6857},{83.4564},{97.5604},{46.6203},{37.5851},{13.416},{25.4297},{28.8487},{15.2701},{14.6429},{16.3045},{14.3522},{13.2959}},{{326.649},{209.7137},{74.3526},{90.5873},{67.405},{88.5866},{41.3695},{43.1656},{36.0642},{25.9717},{26.0389},{22.95},{19.162},{17.2291}}}};

        //(PSF,window2,window1) with 0 indexing

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
%
%
%   Shows how to get the analysis data
%
%
%

[name,analysis_root] = uigetfile('*.h5');

%get the data
analysis_data = load_analysis_data([analysis_root,name]);

%% Plot th


%% Plot out some scaling results no image quality

figure;
plot(analysis_data.information_content,analysis_data.apr_comp_size,'x')
title('info content vs comp')

figure;plotyy(1:length(analysis_data.information_content),analysis_data.information_content,1:length(analysis_data.information_content),analysis_data.num_parts)
title('info content vs. parts')

figure;plotyy(1:length(analysis_data.information_content),analysis_data.information_content,1:length(analysis_data.information_content),analysis_data.apr_comp_size)
title('info content vs. comp size')

figure;
plot(analysis_data.information_content,analysis_data.num_parts,'x')
title('info content vs num parts')

figure;
plot(analysis_data.num_pixels,analysis_data.num_parts,'x')
title('pixels vs num parts')


figure;
plot(analysis_data.num_pixels,analysis_data.num_seed_cells,'x')
title('pixels vs seed cells') 


figure;
plot(analysis_data.information_content,'Displayname','info content');
title('information content')

hold on
plot(analysis_data.num_parts,'Displayname','num_parts');
legend('show')
title('num parts')

figure;plotyy(1:length(analysis_data.information_content),analysis_data.num_parts,1:length(analysis_data.information_content),analysis_data.num_pixels)
title('number pixels vs. number parts')

figure;
plot(analysis_data.image_size./analysis_data.apr_comp_size)
title('compression ratio')


figure;
plot(analysis_data.image_size,'Displayname','Original Image');

hold on
plot(analysis_data.apr_comp_size,'Displayname','APR');
legend('show')
title('File Size image vs. APR comp.')



%% Plot some stuffs out image quality


figure;plot(analysis_data.psnr_pc,'x','Displayname','pc')
hold all
plot(analysis_data.psnr_org,'Displayname','org');
plot(analysis_data.psnr_lin,'Displayname','lin');
legend('show')
title('PSNR')


figure;plot(analysis_data.ssim_pc,'Displayname','pc')
hold all
plot(analysis_data.ssim_org,'Displayname','org');
plot(analysis_data.ssim_lin,'Displayname','lin');
legend('show')
title('SSIM')


figure;plot(analysis_data.snr_pc,'Displayname','pc')
hold all
plot(analysis_data.snr_org,'Displayname','org');
plot(analysis_data.snr_lin,'Displayname','lin');
legend('show')
title('SNR')


figure;plot(analysis_data.rel_l2_pc,'Displayname','l2')
legend('show')
title('rel error')

figure;
plot(analysis_data.rel_linf_pc,'Displayname','linf');
legend('show')
title('rel error')


figure;
plot(analysis_data.information_content,'Displayname','info content');
legend('show')
title('information content')

hold on
plot(analysis_data.num_parts,'Displayname','num_parts');
legend('show')
title('num parts')

figure;
plot(analysis_data.information_content,analysis_data.apr_comp_size,'x')
title('info content vs comp')

figure;plotyy(1:length(analysis_data.information_content),analysis_data.information_content,1:length(analysis_data.information_content),analysis_data.num_parts)
title('info content vs. parts')

figure;plotyy(1:length(analysis_data.information_content),analysis_data.information_content,1:length(analysis_data.information_content),analysis_data.apr_comp_size)
title('info content vs. comp size')

figure;
plot(analysis_data.information_content,analysis_data.num_parts,'x')
title('info content vs num parts')


figure;plot(analysis_data.rel_l2_pc,'Displayname','pc')
hold on
plot(analysis_data.rel_l2_lin,'Displayname','lin')
%plot(analysis_data.rel_l2_org,'Displayname','org')
legend('show')
title('rel error l2')

figure;plot(analysis_data.rel_linf_pc,'Displayname','pc')
hold on
plot(analysis_data.rel_linf_lin,'Displayname','lin')
%plot(analysis_data.rel_linf_org,'Displayname','org')
legend('show')
title('rel error linf')

figure;
plot(analysis_data.rel_linf_org,'Displayname','org')
legend('show')
title('rel error linf')





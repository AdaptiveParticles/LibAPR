%
%   Bevan Cheeseman 2018
%
%   Plotting output of APR benchmarkrs   
%
%   Requires the Matlab folder to be in path.

[name,analysis_root] = uigetfile('*.h5');

%get the data
ad = load_analysis_data([analysis_root,name]);

%% DO SOMETHING

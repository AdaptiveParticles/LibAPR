%
%   Bevan Cheeseman 2018 (Adaptive Particle Representation)
%
%   Demo: showing syntax for reading in apr_paraview file into matlab. Note
%   this stored explictly all Particle Cell data and is therefore not
%   memory efficient. (Please use the C++ code for more data intensive tasks)
%

%find the path to APR dataset produced by Example_produce_paraview_file
[name,analysis_root] = uigetfile('*.h5');

apr = load_apr_full([analysis_root,name]);
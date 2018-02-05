%
%   Bevan Cheeseman 2018 (Adaptive Particle Representation)
%
%   Demo: showing syntax for reading in apr_paraview file into matlab. Note
%   this stored explictly all Particle Cell data and is therefore not
%   memory efficient. (Please use the C++ code for more data intensive tasks)
%

%add your path to APR dataset produced by Example_produce_paraview_file
filename = '/Users/cheesema/PhD/ImageGenData/Exemplar_aprs/spheres/120/sphere_apr_paraview.h5';

apr_sphere = load_apr_full(filename);
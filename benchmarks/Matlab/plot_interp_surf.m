function plot_interp_surf(X,Y,Z,varargin)
%
%   Bevan Cheeseman 2016
%
%   Interpolated Surface, rescales the two axis
%

ma = max(double(X));
mb  = max(double(Y));

va = double(X);
vb = double(Y);
vz = double(Z);
N = 500;

[xq,yq] = meshgrid(linspace(min(va)/ma,1,N),linspace(min(vb)/mb,1,N));
vq = griddata(va/ma,vb/mb,vz,xq,yq);

[xqt,yqt] = meshgrid(linspace(min(va),ma,N),linspace(min(vb),mb,N));


if ~isempty(varargin)
    figure
    mesh(xqt,yqt,vq);
    
else
    figure
    mesh(xqt,yqt,vq);
    hold on
    plot3(X,Y,Z,'rx')
end

    
    

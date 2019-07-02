function plot_surf_unique(X,Y,Z,uQ,type)
%
%   Bevan Cheeseman 2016
%
%   Interpolated Surface, rescales the two axis
%

va = double(X);
vb = double(Y);
vz = double(Z);
N = 500;

v = double(uQ);

%gets the unique vector
un_v = unique(v);

indx_v = cell(length(un_v),1);

for i = 1:length(un_v)
   indx_v{i} = find(v == un_v(i)); 
end



if (type == 1)
    
    ma = max(double(X));
    mb  = max(double(Y));
    
    va = double(X);
    vb = double(Y);
    vz = double(Z);
    
    [xq,yq] = meshgrid(linspace(min(va)/ma,1,N),linspace(min(vb)/mb,1,N));
    vq = griddata(va/ma,vb/mb,vz,xq,yq);
    
    figure
    mesh(xqt,yqt,vq);
    
elseif (type == 2)
    for i = 1:length(un_v)
        
        ma = max(double(X(indx_v{i})));
        mb  = max(double(Y(indx_v{i})));
        
        va = double(X(indx_v{i}));
        vb = double(Y(indx_v{i}));
        vz = double(Z(indx_v{i}));
        
        [xq,yq] = meshgrid(linspace(min(va)/ma,1,N),linspace(min(vb)/mb,1,N));
        vq = griddata(va/ma,vb/mb,vz,xq,yq);

        [xqt,yqt] = meshgrid(linspace(min(va),ma,N),linspace(min(vb),mb,N));
        
        figure
        imagesc(vq)
        
    end
elseif (type ==3)
    figure
    for i = 1:length(un_v)
        
        ma = max(double(X(indx_v{i})));
        mb  = max(double(Y(indx_v{i})));
        
        va = double(X(indx_v{i}));
        vb = double(Y(indx_v{i}));
        vz = double(Z(indx_v{i}));
        
        [xq,yq] = meshgrid(linspace(min(va)/ma,1,N),linspace(min(vb)/mb,1,N));
        vq = griddata(va/ma,vb/mb,vz,xq,yq);

        [xqt,yqt] = meshgrid(linspace(min(va),ma,N),linspace(min(vb),mb,N));
        
        figure
        %mesh(xqt,yqt,vq);
        %hold on
        plot3(va,vb,vz,'x')
        title(num2str(un_v(i)))
    end
    
    
end

    
    

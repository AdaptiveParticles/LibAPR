function h = plot_ind(X,Y,V,val,type,col)
%
%
%   Bevan Cheeseman 2016
%
%   Plots two variables for one of the other variables, averaging over the
%   repeatitions.
%
%
%

X = double(X);
Y = double(Y);

V = double(V);

indx_v = find(round(V*1000) == round(val*1000));

X = X(indx_v);
Y = Y(indx_v);
%avg plot take the average over duplicates on the X values

%average over repeats

un_x = unique(X);

temp_x_un = zeros(length(un_x),1);
temp_y_un = zeros(length(un_x),1);
temp_y_un_sd = zeros(length(un_x),1);

for j = 1:length(un_x)
    indx = find(X == un_x(j));
    
    temp_x_un(j) = un_x(j);
    temp_y_un(j) = mean(Y(indx));
    temp_y_un_sd(j) = sqrt(var(Y(indx)));
end

if(type == 0)
    if ~isempty(temp_y_un)
        max_val = temp_y_un(1);
    else
        max_val = 1;
    end
else
   max_val = 1;
end

h = shadedErrorBar(temp_x_un,temp_y_un/max_val,temp_y_un_sd/max_val,{'Color',col},1);





end
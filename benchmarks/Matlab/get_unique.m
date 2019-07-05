function [out] = get_unique(X,Y,v,type)
%
%
%   Bevan Cheeseman 2016
%
%   Gets all the unique values of the variable and outputs the average data
%   for each unique value
%
%

X = double(X);
Y = double(Y);
v = double(v);

%gets the unique vector
un_v = unique(v);

indx_v = cell(length(un_v),1);

for i = 1:length(un_v)
    indx_v{i} = find(v == un_v(i));
end


if type ~= 3
    
    %avg plot take the average over duplicates on the X values
    
    
    for i = 1:length(un_v)
        %average over repeats
        
        temp_x = X(indx_v{i});
        temp_y = Y(indx_v{i});
        
        un_x = unique(temp_x);
        
        temp_x_un = zeros(length(un_x),1);
        temp_y_un = zeros(length(un_x),1);
        temp_y_un_sd = zeros(length(un_x),1);
        
        for j = 1:length(un_x)
            indx = find(temp_x == un_x(j));
            
            temp_x_un(j) = un_x(j);
            temp_y_un(j) = mean(temp_y(indx));
            temp_y_un_sd(j) = sqrt(var(temp_y(indx)));
        end
        
        out(i).x = temp_x_un;
        out(i).y = temp_y_un;
        out(i).y_sd = temp_y_un_sd;
        
    end
    
    
elseif type == 3
    
    %avg plot take the average over duplicates on the X values, max
    %normalized
    
    
    for i = 1:length(un_v)
        %average over repeats
        
        temp_x = X(indx_v{i});
        temp_y = Y(indx_v{i});
        
        un_x = unique(temp_x);
        
        temp_x_un = zeros(length(un_x),1);
        temp_y_un = zeros(length(un_x),1);
        temp_y_un_sd = zeros(length(un_x),1);
        
        for j = 1:length(un_x)
            indx = find(temp_x == un_x(j));
            
            temp_x_un(j) = un_x(j);
            temp_y_un(j) = mean(temp_y(indx));
            temp_y_un_sd(j) = sqrt(var(temp_y(indx)));
        end
        
        max_val = max(temp_y_un);
        max_val = temp_y_un(1);
        
        out(i).x = temp_x_un;
        out(i).y = temp_y_un/max_val;
        out(i).y_sd = temp_y_un_sd;
    end
    
    
end



end
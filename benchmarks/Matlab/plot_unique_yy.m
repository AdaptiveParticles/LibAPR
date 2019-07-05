function plot_unique_yy(X,Y1,Y2,v,type)
%
%
%   Bevan Cheeseman 2016
%
%   Gets all the unique values of the variable and plots a scatter and an
%   average plot for them
%
%
%

X = double(X);
Y1 = double(Y1);
Y2 = double(Y2);
v = double(v);

%gets the unique vector
un_v = unique(v);

indx_v = cell(length(un_v),1);

for i = 1:length(un_v)
   indx_v{i} = find(v == un_v(i)); 
end

if type == 1
    figure;
    %scatter plot
    hold on
    for i = 1:length(un_v)
        plotyy(X(indx_v{i}),Y1(indx_v{i}),X(indx_v{i}),Y2(indx_v{i}),'x','Displayname',num2str(un_v(i)))
    end
    
elseif type == 2
    figure;
    %avg plot take the average over duplicates on the X values
    
    hold on
    for i = 1:length(un_v)
        %average over repeats
        
        temp_x = X(indx_v{i});
        temp_y = Y1(indx_v{i});
        temp_y2 = Y2(indx_v{i});
        
        un_x = unique(temp_x);
        
        temp_x_un = zeros(length(un_x),1);
        temp_y_un = zeros(length(un_x),1);
        temp_y_un_sd = zeros(length(un_x),1);
        
        temp_y2_un = zeros(length(un_x),1);
        temp_y2_un_sd = zeros(length(un_x),1);
        
        for j = 1:length(un_x)
            indx = find(temp_x == un_x(j));
            
            temp_x_un(j) = un_x(j);
            temp_y_un(j) = mean(temp_y(indx));
            temp_y_un_sd(j) = sqrt(var(temp_y(indx)));
            
            temp_y2_un(j) = mean(temp_y2(indx));
            temp_y2_un_sd(j) = sqrt(var(temp_y2(indx)));
        end
        
        shadedErrorBaryy(temp_x_un,temp_y_un,temp_y_un_sd,{},temp_x_un,temp_y2_un,temp_y2_un_sd,{})
    end
    
end
    




end
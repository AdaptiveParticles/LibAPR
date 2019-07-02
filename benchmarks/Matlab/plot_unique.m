function [h] = plot_unique(X,Y,v,type,varargin)
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
Y = double(Y);
v = double(v);

if ~isempty(varargin)
   cmp = varargin{1};
else
   cmp = colormap(); 
end

%gets the unique vector
un_v = unique(v);

indx_v = cell(length(un_v),1);

for i = 1:length(un_v)
   indx_v{i} = find(v == un_v(i)); 
end

if type == 1
  
    %scatter plot
    hold on
    for i = 1:length(un_v)
        h(i) = plot(X(indx_v{i}),Y(indx_v{i}),'x','Displayname',num2str(un_v(i)),'Color',cmp(i,:));
    end
    
elseif type == 2

    %avg plot take the average over duplicates on the X values
    
    hold on
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
        
        h(i) = shadedErrorBar(temp_x_un,temp_y_un,temp_y_un_sd,{'Displayname',num2str(un_v(i)),'Color',cmp(i,:)},1);
    end
    
    
elseif type == 3

    %avg plot take the average over duplicates on the X values, max
    %normalized
    
    hold on
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
        
        h(i) = shadedErrorBar(temp_x_un,temp_y_un/max_val,temp_y_un_sd/max_val,{'Displayname',num2str(un_v(i)),'Color',cmp(i,:)},1);
    end
    
    
end
    


end
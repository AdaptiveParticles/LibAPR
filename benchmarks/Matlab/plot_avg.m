function h = plot_avg(X,Y,name,varargin)
%
%
%   Bevan Cheeseman 2016
%
%   Gets all the unique values of the variable and plots a scatter and an
%   average plot for them
%
%
%

if(~isempty(varargin))
    
   if(strcmp(varargin{1},'none'))
       error_bar = 0;
   elseif (strcmp(varargin{1},'bars'))
       error_bar = 1;
   elseif(strcmp(varargin{1},'shaded'))
        error_bar = 2;
   else
       error_bar = 2;
   end
   
   if(length(varargin) > 1)
       
      sym = varargin{2}; 
       
   else
       sym = '-x';
   end
   
   if(length(varargin) > 2)
       
      Color = varargin{3}; 
       
   else
       Color = [];
   end
   
    
else 
   error_bar = 2;
end

X = double(X);
Y = double(Y);

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

if(error_bar ==2)
    h = shadedErrorBar(temp_x_un,temp_y_un,temp_y_un_sd,{'Displayname',name},0);
end

if(error_bar ==1)
    h = errorbar(temp_x_un,temp_y_un,temp_y_un_sd,sym,'Displayname',name);
end

if(error_bar ==0)
    h = plot(temp_x_un,temp_y_un,sym,'Displayname',name);
end

if(length(Color) > 0)
    h.Color = Color;
end


end
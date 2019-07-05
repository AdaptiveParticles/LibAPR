function [temp_x_un,temp_y_un] =  get_avg(X,Y)
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
end
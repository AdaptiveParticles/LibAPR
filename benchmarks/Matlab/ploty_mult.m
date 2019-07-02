function [hAx,hLine1,hLine2] = ploty_mult(x_axis,y_axis)


%first average the data

[x_axis.x,x_axis.y] = get_avg(x_axis.x,x_axis.y);

for i = 1:length(y_axis)
    [y_axis(i).x,y_axis(i).y] = get_avg(y_axis(i).x,y_axis(i).y);
end

%first get the maximum length of any of the vectors
max_len = length(x_axis.x);

for i = 1:length(y_axis)
    max_len = max(max_len,length(y_axis(i).x));
    
end

% then padd the vectors

x_axis.x = [x_axis.x;nan(max_len - length(x_axis.x),1)];
x_axis.y = [x_axis.y;nan(max_len - length(x_axis.y),1)];

y_vec_x = [];
y_vec_y = [];

for i = 1:length(y_axis)
    y_axis(i).x = [y_axis(i).x',nan(1,max_len - length(y_axis(i).x))];
    y_axis(i).y = [y_axis(i).y',nan(1,max_len - length(y_axis(i).y))];
    
    y_vec_x = [y_vec_x;y_axis(i).x];
    y_vec_y = [y_vec_y;y_axis(i).y];
end


[hAx,hLine1,hLine2] = plotyy(gca,x_axis.x,x_axis.y,y_vec_x',y_vec_y');


end

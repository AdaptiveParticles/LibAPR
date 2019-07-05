function format_figure_scatter(cf)
%
%   Bevan Cheeseman 2017
%
%   Formatting For Part Paper
%


%% Axis Properties

cf.Colormap = colormap(jet);

for i = 1:length(cf.Children)
   a = cf.Children(i); 
    
   a.FontSize = 20;
   
   a.YColor = [0,0,0];
   
   a.YTickMode = 'auto';
   
   a.XMinorTick = 'on';
   a.YMinorTick = 'on';
   
   % Line Properties
   
   for j = 1:length(a.Children)
        l = a.Children(j);
        
        l.LineWidth = 1.5;
        
        
   end
   
    
end


cf.Clipping = 'off';
cf.Position(3) = 1.5*250;
cf.Position(4) = 1.5*200; 




end
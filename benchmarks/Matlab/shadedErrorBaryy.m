function ax = shadedErrorBaryy(x1,y1,er1,c1,x2,y2,er2,c2)
% ax = SHADEDERRORBARYY(x1,y1,er1,c1,x2,y2,er2,c2) is a wrapper for the
% popular File Exchange function shadedErrorBar which allows you to put two
% of these beautiful objects on two y-scales using the plotyy format. This
% is a bit of a hack and requires shadedErrorBar to already be in your
% path. The output, ax, is a vector containing the left and right axes
% handles.

%opengl software %work-around to fix the fact that transparency tends to
% make the y-axis disapear. Only works on PCs

ca = gca;

a = figure('visible','off');
H(1) = shadedErrorBar(x1,y1,er1,c1,1);
b = figure('visible','off');
H(2) = shadedErrorBar(x2,y2,er2,c2,1);

ax = plotyy(ca,x1,y1,x2,y2);

for iAx = 1:2
    handle_vec = [H(iAx).patch H(iAx).mainLine H(iAx).edge(1) H(iAx).edge(2)];
    set(handle_vec,'Parent',ax(iAx));
end

axis(ax,'tight');

set(ax,'box','off'); %gets rid of pesky right-side tickmarks

%set(ax(1),'YColor',c1);
%set(ax(2),'YColor',c2);

close(a,b);
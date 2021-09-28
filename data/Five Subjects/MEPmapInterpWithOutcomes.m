function [xi,yi,zi,VarOutput] =  MEPmapInterpWithOutcomes(NavData,MEPamp,MEPampThresh)

format short g
MeshRes = .1;
xMESH = min(NavData(:,1)):MeshRes:max(NavData(:,1));
yMESH = min(NavData(:,2)):MeshRes:max(NavData(:,2));
[xi, yi] = meshgrid(xMESH,yMESH);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
zi = griddata(NavData(:,1),NavData(:,2),MEPamp, xi,yi,'cubic');
[zinan] = find(isnan(zi));
zi(zinan) = 0;
[zithresh] = find(zi<MEPampThresh);
zi(zithresh) = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
stem3(xi,yi,zi)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MapVol = trapz(xMESH',trapz(yMESH',zi',2),1);
MapMean = mean(nonzeros(zi));
MapArea = MapVol/MapMean;
CoGx = sum(sum( xi .* zi)) ./ sum(sum(zi));
CoGy = sum(sum( yi .* zi)) ./ sum(sum(zi));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

VarOutput = [MapVol,MapMean,MapArea,CoGx,CoGy];
beep









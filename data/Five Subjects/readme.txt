
Folder Contents:
5 Subjects each with 3 maps (GRID,RAND,USER)

GRID MAP: 49 locations with 6 stimulations at each location (294 total stimulation)
USER MAP: center + 4 corners defined remaining 289 stimulations chosen with user expertise (294 total stimulations)
RANDOM MAP: center + 4 corners defined remaining 289 stimulations chosen from a uniform random distirubution (294 stimulations total)

Each data file (ie AS_3MAPS.mat) contains:
NavDataGRID: xyz data of grid locations (cm)
MEPampGRID: amplitude at each grid location (uV)
NavDataRAND: xyz data of rand locations (cm)
MEPampRAND: amplitude at each rand location (uV)
NavDataUSER: xyz data of user locations (cm)
MEPampUSER: amplitude at each user location (uV)

*Note: No thresholding has been done on data. 
*Note: Some data sets will contain less than 294 stimulations. Stimulations were removed for poor coil location, subject head movement, etc


There is a figure provided showing maps for each subject


Code: 
"MEPmapInterpWithOutcomes.m"  
Inputs: NavData, MEPamp, AmpThresh (usually set to 50)
Outputs: Interpolated maps, map volume, map area, mean amplitude, center of gravity in x coordinate (COGx), center of gravity in y coordinate (COGy)
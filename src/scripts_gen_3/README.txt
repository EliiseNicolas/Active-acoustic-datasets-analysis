Filtering Data Gen 3

Updates from gen 2 : 

- Filtering by bathymetry based on gebco bathymetry data

Filtering order : 

1) Extract variables of interest from IMOS file => same as gen2
	- 1 pkl file created for one folder of IMOS files
	- resulting pkl files of this shape : Tuple(name_IMOS_trajectory, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "DAY", "SEASONS", "in_ROI"
	- Sv a 2D array no matters the number of initial dimensions in the IMOS file : only one channel is kept
	
2) Filter data by season => same as gen2
	- 4 different pkl files created, one per season : winter, spring, summer, fall
	- resulting pkl files of this shape : Tuple (name_IMOS_trajectory, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "DAY", "SEASONS", "in_ROI"
	
3) filter data by bathymetry 

3) Filter data by period(day/sunset/sunrise/night) => same as gen2
	- 4 different pkl files created : one per period : day, sunset, sunrise, night
	- resulting pkl files of this shape : Tuple (name_IMOS_trajectory, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "DAY", "SEASONS", "in_ROI"
	
5) Filter data by dates => same as gen 2
	- 1 pkl file created for one source file
	- resulting pkl file of this shape : Tuple(date, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "DAY", "SEASONS", "in_ROI"
	

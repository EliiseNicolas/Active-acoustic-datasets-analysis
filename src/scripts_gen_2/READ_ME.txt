Filtering Data Gen 2

Updates from gen 1 : 

- Scripts are now in functions callable from the terminal
- A final script is created to call every filtering script to convert IMOS file to final processed data
- Scripts are modified to be more general callable for files of different channels
- filtering order is different

Filtering order : 

1) Extract variables of interest from IMOS file
	- 1 pkl file created for one folder of IMOS files
	- resulting pkl files of this shape : Tuple(name_IMOS_trajectory, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "DAY", "SEASONS", "in_ROI"
	- Sv a 2D array no matters the number of initial dimensions in the IMOS file : only one channel is kept
	
2) Filter data by bathymetry : only datas in pkl where Sv values go to depth of 1000m are kept
	- 1 pkl file created per source file
	- resulting pkl file of this shape : Tuple (name_IMOS_trajectory, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "DAY", "SEASONS", "in_ROI"
	
3) Filter data by season 
	- 4 different pkl files created, one per season : winter, spring, summer, fall
	- resulting pkl files of this shape : Tuple (name_IMOS_trajectory, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "DAY", "SEASONS", "in_ROI"
	
4) Filter data by period(day/sunset/sunrise/night)
	- 4 different pkl files created : one per period : day, sunset, sunrise, night
	- resulting pkl files of this shape : Tuple (name_IMOS_trajectory, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "DAY", "SEASONS", "in_ROI"
	
5) Filter data by dates
	- 1 pkl file created for one source file
	- resulting pkl file of this shape : Tuple(date, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "DAY", "SEASONS", "in_ROI"
	

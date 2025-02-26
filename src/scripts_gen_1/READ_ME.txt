Filtrage supposed to be ordered as so : 

1) by Season 
	pkl = Tuple (name_IMOS_file, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "CHANNEL", "DAY", "SEASONS"
	
1) by Day (24h) 
	pkl = Tuple (date, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "CHANNEL", "DAY", "SEASONS"
	
2) by bathymetry : only datas in pkl where Sv values go to depth of 1000m are kept
	pkl = Tuple (date, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "CHANNEL", "DAY", "SEASONS"
	
3) Labels : new key "in_ROI" is putted to the dictionnary. Is True if there is a point (longitude, latitude) in the dict that is in the ROI
	pkl = Tuple (date, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "CHANNEL", "DAY", "SEASONS", "in_ROI"
	
4) by periods (day, sunset, sunrise, night) : 4 pkl created : one per period
	pkl = Tuple (date, Dict) where keys are "TIME", "DEPTH", "Sv", "LATITUDE", "LONGITUTE", "CHANNEL", "DAY", "SEASONS", "in_ROI"
	
	
The issue of this filtering is that when cropping datas by 24h, the period isn't respected : 
for instance, in an echogram "day", I can have the backscattering of "day" period of day 2017-11-21 from 00:00 to 20:00 and then the period day of the "next day" (still the 2017-11-21 (but night period was between those two day period) from 21:00 to 23:59.




directory = directory;
samplename = samplename;
savedirectory = savedirectory;


firstsample = 01;
lastsample = 09;

for (i=firstsample;i<lastsample; i++){
	if (i < 10){
		filename = samplename + "0" + i;
		filename = samplename + "_0" + i;
	}
	if (i >= 10){
		filename = samplename + i;
		filename = samplename + "_" + i;
	}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
savedirectory = savedirectory + "Size Constraints";
watershedtolerance = 1;
for (watershedtolerance=1;watershedtolerance<10;watershedtolerance++){
open(directory + filename + ".TIF");
run("Set Scale...", "distance=68 known=300 unit=nm");
makeRectangle(1, 1, 1022, 697);
run("Crop");
run("Median...", "radius=2");
run("Subtract Background...", "rolling=20 light sliding");
setAutoThreshold("Moments dark");
run("Convert to Mask");
watershedtolerancecorrected = watershedtolerance/10000;
run("adjustable watershed", "tolerance=watershedtolerancecorrected");
run("Set Measurements...", "area mean modal min centroid center perimeter bounding fit shape feret's integrated stack redirect=None decimal=3");
run("Analyze Particles...", "size=2000-Infinity circularity=0.5-1.00 show=Outlines display exclude");
saveAs("Jpeg", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected + ".jpg");
saveAs("Results", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected +  "-Results.csv");
run("Clear Results");
selectWindow(filename + ".TIF");
close();
selectWindow("Drawing of " + filename + ".TIF");
close();
	}
	
watershedtolerance = 1;
for (watershedtolerance=1;watershedtolerance<10;watershedtolerance++){
open(directory + filename + ".TIF");
run("Set Scale...", "distance=68 known=300 unit=nm");
makeRectangle(1, 1, 1022, 697);
run("Crop");
run("Median...", "radius=2");
run("Subtract Background...", "rolling=20 light sliding");
setAutoThreshold("Moments dark");
run("Convert to Mask");
watershedtolerancecorrected = watershedtolerance/1000;
run("adjustable watershed", "tolerance=watershedtolerancecorrected");
run("Set Measurements...", "area mean modal min centroid center perimeter bounding fit shape feret's integrated stack redirect=None decimal=3");
run("Analyze Particles...", "size=2000-Infinity circularity=0.5-1.00 show=Outlines display exclude");
saveAs("Jpeg", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected + ".jpg");
saveAs("Results", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected +  "-Results.csv");
run("Clear Results");
selectWindow(filename + ".TIF");
close();
selectWindow("Drawing of " + filename + ".TIF");
close();
	}
	
watershedtolerance = 1;
for (watershedtolerance=1;watershedtolerance<10;watershedtolerance++){
open(directory + filename + ".TIF");
run("Set Scale...", "distance=68 known=300 unit=nm");
makeRectangle(1, 1, 1022, 697);
run("Crop");
run("Median...", "radius=2");
run("Subtract Background...", "rolling=20 light sliding");
setAutoThreshold("Moments dark");
run("Convert to Mask");
watershedtolerancecorrected = watershedtolerance/100;
run("adjustable watershed", "tolerance=watershedtolerancecorrected");
run("Set Measurements...", "area mean modal min centroid center perimeter bounding fit shape feret's integrated stack redirect=None decimal=3");
run("Analyze Particles...", "size=2000-Infinity circularity=0.5-1.00 show=Outlines display exclude");
saveAs("Jpeg", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected + ".jpg");
saveAs("Results", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected +  "-Results.csv");
run("Clear Results");
selectWindow(filename + ".TIF");
close();
selectWindow("Drawing of " + filename + ".TIF");
close();
	}
	
watershedtolerance = 1;
for (watershedtolerance=1;watershedtolerance<10;watershedtolerance++){
open(directory + filename + ".TIF");
run("Set Scale...", "distance=68 known=300 unit=nm");
makeRectangle(1, 1, 1022, 697);
run("Crop");
run("Median...", "radius=2");
run("Subtract Background...", "rolling=20 light sliding");
setAutoThreshold("Moments dark");
run("Convert to Mask");
watershedtolerancecorrected = watershedtolerance/10;
run("adjustable watershed", "tolerance=watershedtolerancecorrected");
run("Set Measurements...", "area mean modal min centroid center perimeter bounding fit shape feret's integrated stack redirect=None decimal=3");
run("Analyze Particles...", "size=2000-Infinity circularity=0.5-1.00 show=Outlines display exclude");
saveAs("Jpeg", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected + ".jpg");
saveAs("Results", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected +  "-Results.csv");
run("Clear Results");
selectWindow(filename + ".TIF");
close();
selectWindow("Drawing of " + filename + ".TIF");
close();
	}

watershedtolerance = 1;
for (watershedtolerance=1;watershedtolerance<10;watershedtolerance++){
open(directory + filename + ".TIF");
run("Set Scale...", "distance=68 known=300 unit=nm");
makeRectangle(1, 1, 1022, 697);
run("Crop");
run("Median...", "radius=2");
run("Subtract Background...", "rolling=20 light sliding");
setAutoThreshold("Moments dark");
run("Convert to Mask");
watershedtolerancecorrected = watershedtolerance/1;
run("adjustable watershed", "tolerance=watershedtolerancecorrected");
run("Set Measurements...", "area mean modal min centroid center perimeter bounding fit shape feret's integrated stack redirect=None decimal=3");
run("Analyze Particles...", "size=2000-Infinity circularity=0.5-1.00 show=Outlines display exclude");
saveAs("Jpeg", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected + ".jpg");
saveAs("Results", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected +  "-Results.csv");
run("Clear Results");
selectWindow(filename + ".TIF");
close();
selectWindow("Drawing of " + filename + ".TIF");
close();
	}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
savedirectory = savedirectory + "No Size Constraints";
watershedtolerance = 1;
for (watershedtolerance=1;watershedtolerance<10;watershedtolerance++){
open(directory + filename + ".TIF");
run("Set Scale...", "distance=68 known=300 unit=nm");
makeRectangle(1, 1, 1022, 697);
run("Crop");
run("Median...", "radius=2");
run("Subtract Background...", "rolling=20 light sliding");
setAutoThreshold("Moments dark");
run("Convert to Mask");
watershedtolerancecorrected = watershedtolerance/10000;
run("adjustable watershed", "tolerance=watershedtolerancecorrected");
run("Set Measurements...", "area mean modal min centroid center perimeter bounding fit shape feret's integrated stack redirect=None decimal=3");
run("Analyze Particles...", "size=0-Infinity circularity=0-1.00 show=Outlines display exclude");
saveAs("Jpeg", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected + "-No Size Constraints.jpg");
saveAs("Results", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected +  "-No Size Constraints-Results.csv");
run("Clear Results");
selectWindow(filename + ".TIF");
close();
selectWindow("Drawing of " + filename + ".TIF");
close();
	}
	
watershedtolerance = 1;
for (watershedtolerance=1;watershedtolerance<10;watershedtolerance++){
open(directory + filename + ".TIF");
run("Set Scale...", "distance=68 known=300 unit=nm");
makeRectangle(1, 1, 1022, 697);
run("Crop");
run("Median...", "radius=2");
run("Subtract Background...", "rolling=20 light sliding");
setAutoThreshold("Moments dark");
run("Convert to Mask");
watershedtolerancecorrected = watershedtolerance/1000;
run("adjustable watershed", "tolerance=watershedtolerancecorrected");
run("Set Measurements...", "area mean modal min centroid center perimeter bounding fit shape feret's integrated stack redirect=None decimal=3");
run("Analyze Particles...", "size=0-Infinity circularity=0-1.00 show=Outlines display exclude");
saveAs("Jpeg", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected + "-No Size Constraints.jpg");
saveAs("Results", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected +  "-No Size Constraints-Results.csv");
run("Clear Results");
selectWindow(filename + ".TIF");
close();
selectWindow("Drawing of " + filename + ".TIF");
close();
	}
	
watershedtolerance = 1;
for (watershedtolerance=1;watershedtolerance<10;watershedtolerance++){
open(directory + filename + ".TIF");
run("Set Scale...", "distance=68 known=300 unit=nm");
makeRectangle(1, 1, 1022, 697);
run("Crop");
run("Median...", "radius=2");
run("Subtract Background...", "rolling=20 light sliding");
setAutoThreshold("Moments dark");
run("Convert to Mask");
watershedtolerancecorrected = watershedtolerance/100;
run("adjustable watershed", "tolerance=watershedtolerancecorrected");
run("Set Measurements...", "area mean modal min centroid center perimeter bounding fit shape feret's integrated stack redirect=None decimal=3");
run("Analyze Particles...", "size=0-Infinity circularity=0-1.00 show=Outlines display exclude");
saveAs("Jpeg", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected + "-No Size Constraints.jpg");
saveAs("Results", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected +  "-No Size Constraints-Results.csv");
run("Clear Results");
selectWindow(filename + ".TIF");
close();
selectWindow("Drawing of " + filename + ".TIF");
close();
	}
	
watershedtolerance = 1;
for (watershedtolerance=1;watershedtolerance<10;watershedtolerance++){
open(directory + filename + ".TIF");
run("Set Scale...", "distance=68 known=300 unit=nm");
makeRectangle(1, 1, 1022, 697);
run("Crop");
run("Median...", "radius=2");
run("Subtract Background...", "rolling=20 light sliding");
setAutoThreshold("Moments dark");
run("Convert to Mask");
watershedtolerancecorrected = watershedtolerance/10;
run("adjustable watershed", "tolerance=watershedtolerancecorrected");
run("Set Measurements...", "area mean modal min centroid center perimeter bounding fit shape feret's integrated stack redirect=None decimal=3");
run("Analyze Particles...", "size=0-Infinity circularity=0-1.00 show=Outlines display exclude");
saveAs("Jpeg", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected + "-No Size Constraints.jpg");
saveAs("Results", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected +  "-No Size Constraints-Results.csv");
run("Clear Results");
selectWindow(filename + ".TIF");
close();
selectWindow("Drawing of " + filename + ".TIF");
close();
	}

watershedtolerance = 1;
for (watershedtolerance=1;watershedtolerance<10;watershedtolerance++){
open(directory + filename + ".TIF");
run("Set Scale...", "distance=68 known=300 unit=nm");
makeRectangle(1, 1, 1022, 697);
run("Crop");
run("Median...", "radius=2");
run("Subtract Background...", "rolling=20 light sliding");
setAutoThreshold("Moments dark");
run("Convert to Mask");
watershedtolerancecorrected = watershedtolerance/1;
run("adjustable watershed", "tolerance=watershedtolerancecorrected");
run("Set Measurements...", "area mean modal min centroid center perimeter bounding fit shape feret's integrated stack redirect=None decimal=3");
run("Analyze Particles...", "size=0-Infinity circularity=0-1.00 show=Outlines display exclude");
saveAs("Jpeg", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected + "-No Size Constraints.jpg");
saveAs("Results", savedirectory + "Drawing of " + filename + " Watershed Tolerance" + watershedtolerancecorrected +  "-No Size Constraints-Results.csv");
run("Clear Results");
selectWindow(filename + ".TIF");
close();
selectWindow("Drawing of " + filename + ".TIF");
close();
	}
}



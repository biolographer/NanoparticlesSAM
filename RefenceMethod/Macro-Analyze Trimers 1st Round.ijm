

directory = directory;
samplename = samplename;
firstsample = 01;
lastsample = 09;

for (i=firstsample;i<lastsample; i++){
filename = samplename + 0 + i;
open(directory + filename + ".TIF");

run("Set Scale...", "distance=68 known=300 unit=nm");

makeRectangle(1, 1, 1022, 697);
run("Crop");

run("Median...", "radius=2");
run("Subtract Background...", "rolling=20 light sliding");

setAutoThreshold("Moments dark");
run("Convert to Mask");
////////////////////////////////////////////////////////////////////
run("adjustable watershed", "tolerance=10");
///////////////////////////////////////////////////////////////////

run("Set Measurements...", "area mean modal min centroid center perimeter bounding fit shape feret's integrated stack redirect=None decimal=3");
run("Analyze Particles...", "size=2000-Infinity circularity=0.50-1.00 show=Outlines display exclude");

saveAs("Jpeg", directory + "Drawing of " + filename + ".jpg");
saveAs("Results", directory + "Drawing of " + filename + "-Results.csv");
run("Clear Results");

selectWindow(filename + ".TIF");
close();
selectWindow("Drawing of "+ filename + ".TIF");
close();
}
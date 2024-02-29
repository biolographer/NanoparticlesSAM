

directory = directory;
samplename = samplename;


open(directory + filename + ".TIF");

run("Set Scale...", "distance=68 known=300 unit=nm");

makeRectangle(1, 1, 1022, 697);
run("Crop");

run("Median...", "radius=2");
run("Subtract Background...", "rolling=20 light sliding");
run("Set Measurements...", "area mean modal min centroid center perimeter bounding fit shape feret's integrated stack redirect=None decimal=3");
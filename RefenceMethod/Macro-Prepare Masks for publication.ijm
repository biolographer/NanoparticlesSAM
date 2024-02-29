

//close();

directory = directory;
samplename = samplename;


open(directory + samplename + ".jpg");

run("8-bit");

setAutoThreshold("Default");
setThreshold(0, 60);
setOption("BlackBackground", false);
run("Convert to Mask");

run("Dilate");

startx = 250;
starty = 200;
//use even number for width and height
width = 512;
height = 348;
//image width - 1024 when they come directly from the SEM
totaldistance = 1024;
run("Set Scale...", "distance=totaldistance known=4517 unit=nm");

startx = startx*totaldistance/1024;
starty = starty*totaldistance/1024;
width = width*totaldistance/1024;
height = height*totaldistance/1024;

//scale bar properties
scalebarwidth = 250;
scalebarheight = 15*totaldistance/1024;
scalebarfontsize = 20;
writelenght = "False";

makeRectangle(startx, starty, width, height);
run("Crop");

savedirectory = directory;

saveAs("PNG", savedirectory + samplename + " Binary.png");
//change color to red for better visualization over the SEM images

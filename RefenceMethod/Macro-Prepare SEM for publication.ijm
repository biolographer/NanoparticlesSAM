

//close();

directory = directory;
samplename = samplename;

open(directory + samplename + ".TIF");

//for the images generated using ImageJ
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

if (writelenght == "False"){
run("Scale Bar...", "width=scalebarwidth height=scalebarheight color=White background=Black location=[Lower Right] bold hide overlay");
}

savedirectory = directory;

saveAs("Jpeg", savedirectory + samplename + " - Prepared.TIF");



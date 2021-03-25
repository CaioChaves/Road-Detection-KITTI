cd '/Users/CaioChaves/PRE/data_semantics/training/semantic_rgb'

imgRGBdouble = double(imread('000000_10.png'));
pixel = reshape(imgRGBdouble(1,1,:),[1,3]);  %% 1,1 --> i,i

green = [107,142,35];        
blue = [0,0,142];
fucsia = [128,64,128];
yellow = [220,220,0];
black = [0,0,0];
pink = [244,35,232];
red = [220,20,60];

switch pixel
    case green
        binarySol = 0;
    case blue
        binarySol = 0;
    case fucsia
        binarySol = 1;
    case yellow 
        binarySol = 0;
    case black
        binarySol = 0;
    case pink
        binarySol = 1;
    case red
        binarySol = 0;
end


cd ../../../matlab
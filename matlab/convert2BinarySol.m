cd '/Users/CaioChaves/ENSTA-PRE-2019/PRe/data_semantics/training/semantic'

files = dir('*.png');    % get all the .png files from the directory
numberofFiles = length(files);

% Semantic classifications
green = [107,142,35];        
blue = [0,0,142];
fucsia = [128,64,128];
yellow = [220,220,0];
black = [0,0,0];
pink = [244,35,232];
red = [220,20,60];

for i=1:2
    
    file = files(i).name;
    imgdouble = double(imread(file));
    pxl_X = size(imgdouble,1);
    pxl_Y = size(imgdouble,2);
    
    binarySol = zeros(pxl_X,pxl_Y);  
%     for j = 1:pxl_X
%         for k = 1:pxl_Y
%             pixel = imgdouble(j,k);
%             if pixel == 4
%                 binarySol(j,k) = 0;
%             elseif pixel == 7
%                 binarySol(j,k) = 1;
%             elseif pixel == 8
%                 binarySol(j,k) = 1;
%             elseif pixel == 11
%                 binarySol(j,k) = 0;
%             elseif pixel == 12
%                 binarySol(j,k) = 0;
%             elseif pixel == 13
%                 binarySol(j,k) = 0;
%             elseif pixel == 17
%                 binarySol(j,k) = 0;
%             elseif pixel == 20
%                 binarySol(j,k) = 0;
%             elseif pixel == 21
%                 binarySol(j,k) = 0;
%             elseif pixel == 23
%                 binarySol(j,k) = 0;
%             elseif pixel == 24
%                 binarySol(j,k) = 0;
%             elseif pixel == 26
%                 binarySol(j,k) = 0;
%             elseif pixel == 28
%                 binarySol(j,k) = 0;
%             else 
%                 display('ERROR! Pixel value does not match any class');
%                 break
%             end
%         end   
%     end

    binarySol = imgdouble>5.*imgdouble<10;
    
    cd ../semantic_binarySol
    imwrite(binarySol,['binary',file],'png');
    display(['Image ',int2str(i),' over ',int2str(numberofFiles),' converted!'])
    
    cd ../semantic
    
end



cd ../../../matlab
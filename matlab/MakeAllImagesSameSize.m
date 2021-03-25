currDir = pwd;
cd ../data_semantics/training/image_2/

images = dir('*.png');
PX = zeros(length(images),1);
PY = zeros(length(images),1);

for i = 1:length(images)
    name = images(i).name;
    image = double(imread(name))/255.0;
    PY(i) = size(image,1);
    PX(i) = size(image,2);
end

minPX = min(PX)
minPY = min(PY)

for i = 1:length(images)
    name = images(i).name;
    image = imread(name);
    image_cropped = image(end-minPY+1:end,end-minPX+1:end,:);
    cd ../image_2
    imwrite(image_cropped,[name],'png');
    disp(['Image ',int2str(i),' over ',int2str(length(images)),' converted!'])
    cd ../image_2
end
   

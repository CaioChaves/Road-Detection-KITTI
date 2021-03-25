cd /Users/CaioChaves/ENSTA-PRE-2019/PRe/data_semantics/training/semantics_binary_ground/
clear all

images = dir('*.png');
crop_x_esq = 300;
crop_x_dir = 920;
crop_y_sup = 150;
crop_y_inf = 370;

for i = 1:length(images)
    name = images(i).name;
    image = double(imread(name))/255.0;
    image_cropped = image(crop_y_sup:crop_y_inf,crop_x_esq:crop_x_dir);
    cd ../semantic_binary_ground_balanced
    imwrite(image_cropped,[name],'png');
    disp(['Image ',int2str(i),' over ',int2str(length(images)),' converted!'])
    cd ../semantics_binary_ground
end

cd /Users/CaioChaves/ENSTA-PRE-2019/PRe/data_semantics/training/image_2/

images = dir('*.png');

for i = 1:length(images)
    name = images(i).name;
    image = imread(name);
    image_cropped = image(crop_y_sup:crop_y_inf,crop_x_esq:crop_x_dir);
    cd ../image_2_balanced
    imwrite(image_cropped,[name],'png');
    disp(['Image ',int2str(i),' over ',int2str(length(images)),' converted!'])
    cd ../image_2
end
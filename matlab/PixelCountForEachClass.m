cd /Users/CaioChaves/ENSTA-PRE-2019/PRe/data_semantics/training/semantic_binary_ground_balanced/
clear all

images = dir('*.png');
GroundCount = zeros(1,length(images));
NotGroundCount = zeros(1,length(images));

for i = 1:length(images)
    name = images(i).name;
    image = double(imread(name))/255.0;
    totalPixels = numel(image);
    groundCount = sum(image(:) == 1.0);
    notGroundCount = sum(image(:) == 0.0);
    if (groundCount + notGroundCount ~= totalPixels)
        disp('Something is weird...')
    end
    GroundCount(i) = groundCount;
    NotGroundCount = notGroundCount;
end

RatioNG = NotGroundCount./(NotGroundCount+GroundCount);

average = mean(RatioNG);
std = std(RatioNG);
maxNG = max(RatioNG);
minNG = min(RatioNG);

disp('Number of images:'+string(length(images)))
disp('Average proportion of not ground over total:'+string(average))
disp('Standard Deviation proportion of not ground over total:'+string(std))
disp('Maximum proportion of not ground over total:'+string(maxNG))
disp('Minimum proportion of not ground over total:'+string(minNG))


cd ../../../matlab
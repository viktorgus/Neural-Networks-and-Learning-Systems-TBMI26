%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 100;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 1000;
% Number of weak classifiers
nbrWeakClassifiers = 80;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));
% 
% figure(1);
% colormap gray;
% for k=1:25
%     subplot(5,5,k), imagesc(faces(:,:,10*k));
%     axis image;
%     axis off;
% end
% 
% figure(2);
% colormap gray;
% for k=1:25
%     subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
%     axis image;
%     axis off;
% end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);
% 
% figure(3);
% colormap gray;
% for k = 1:25
%     subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
%     axis image;
%     axis off;
% end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
D = 1/size(xTrain,2)*ones(1,size(xTrain,2));
T = ones(nbrWeakClassifiers,1); %skapa placeholder för threshold
P = ones(nbrWeakClassifiers,1); %skapa placeholder för polarity
alphas = ones(nbrWeakClassifiers, 1);
haarfeaturevector = ones(nbrWeakClassifiers,1); 

for i=1:nbrWeakClassifiers %loopa över antal classifiers
    Etot = inf; 
    polarity = 1;     
    for k = 1:nbrHaarFeatures %loopa över filter
        
     for j = 1:size(xTrain,2) %loopa över alla trösklar
       threshold = xTrain(k,j); %trösklar sätts till datapunkter obs ska ej va 1
       C = WeakClassifier(threshold, polarity, xTrain(k,:)); %ska ej va xTrain(1,)
       E = WeakClassifierError(C, D, yTrain);
       
       if E > 0.5 %om felet är större än 0.5, switcha polariteten & felet
           polarity = -polarity; 
           E = 1-E; 
       end
       
       if E < Etot %om felet är mindre än det "bästa" felet så sparas det 
           Etot = E;
           P(i) = polarity; 
           T(i) = threshold;
           haarfeaturevector(i) = k;
 
       end
       
       
     end
    end
   
   disp(i);
   disp(Etot);
       
   alpha = 0.5*log((1-Etot)/Etot);
   predLabel = WeakClassifier(T(i), P(i), xTrain(haarfeaturevector(i),:));
   D = D.*exp(-alpha*yTrain.*predLabel); %update weights
   D = D / sum(D);
   alphas(i) = alpha; %alpha sparas 
   
end


%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

errorsTest = zeros(nbrWeakClassifiers,1);
errorsTrain = zeros(nbrWeakClassifiers,1);

for k= 1:nbrWeakClassifiers
[H, errorsTest(k)] = classifyFace(T(1:k),P(1:k),xTest,yTest,alphas(1:k),haarfeaturevector(1:k));
[H, errorsTrain(k)] = classifyFace(T(1:k),P(1:k),xTrain,yTrain,alphas(1:k),haarfeaturevector(1:k));
end

%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

figure(1);
plot(1:k,errorsTest, "r",1:k,errorsTrain, "b");
%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.


optWeak = find(errorsTest==min(errorsTest));
disp("Optimal ammount of weak classifiers: ");
disp(optWeak);
[HTest, errorsTestOpt] = classifyFace(T(1:optWeak),P(1:optWeak),xTest,yTest,alphas(1:optWeak),haarfeaturevector(1:optWeak));
[HTrain, errorsTrainOpt] = classifyFace(T(1:optWeak),P(1:optWeak),xTrain,yTrain,alphas(1:optWeak),haarfeaturevector(1:optWeak));

disp("Test Err: ");
disp(errorsTestOpt);

whichWrongFace = find(((HTest ~= yTest) == 1) & (yTest == 1));
whichWrongNonface = find(((HTest ~= yTest) == 1) & (yTest == -1));


%Plot wrongly classified faces
figure(6);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(testImages(:,:,whichWrongFace(1,k)));
    axis image;
    axis off;
end


figure(7);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(testImages(:,:,whichWrongNonface(1,k)));
    axis image;
    axis off;
end

%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.

subplotDim = ceil(sqrt(optWeak));
figure(8);
colormap gray;
for k = 1:optWeak
    subplot(subplotDim ,subplotDim ,k),imagesc(haarFeatureMasks(:,:,haarfeaturevector(k,1)),[-1 2]);
    axis image;
    axis off;
end


function [Class, Err] = classifyFace(T,P,X,Y,alphas,haarfeaturevector)
Class = zeros(1,size(X,2));

for i =1:size(X,2)
    classifiers = WeakClassifier(T,P,X(haarfeaturevector,i));
    Class(i) = sign(sum(alphas.*classifiers));
end

Err = sum(Class~=Y)/size(Y,2);
end
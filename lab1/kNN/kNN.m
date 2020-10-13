function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
LPred  = zeros(size(X,1),1);
distances = zeros(size(XTrain,1),1);


% Classify i'th sample from X 
for i = 1:size(X,1)
    for q = 1:size(XTrain,1)
       distances(q, 1) = sqrt(sum((XTrain(q,:)-X(i,:)).^2));     
    end
   
    C = [distances LTrain]; 
    C = sortrows(C,1);
    
    closestLabels = C(1:k,2);
    uniqueClosestLabels = unique(closestLabels);
   
    
    if length(uniqueClosestLabels) == 1
        LPred(i) = uniqueClosestLabels(1);
        continue
    end
    
    Dict = containers.Map('KeyType','int32', 'ValueType', 'any');
    
    for n = 1:k
       label = C(n,2);
       distance = C(n,1);
       if Dict.isKey(label)
          Dict(label) = [Dict(label), distance];
       else
           Dict(label)=distance;
       end  
    end
    
    
    %Finding the label with most distances stored
    numberOfDistancesPerLabel = cellfun('size', Dict.values, 2);
    maximum = max(max(numberOfDistancesPerLabel));
    equalLabels = sum(numberOfDistancesPerLabel == maximum);
  
    
    %Handling equally good labels
    bestDistance = inf;
    for classIndex = 1:length(uniqueClosestLabels)
        class = uniqueClosestLabels(classIndex);
        classDistances = Dict(class);
        if length(classDistances) == maximum
            if equalLabels>1
                totClassDistance = sum(classDistances);
                if totClassDistance<bestDistance
                    bestDistance = totClassDistance;
                    predLabel = class;
                end
            else
                predLabel = class;
                break
            end
        end
    end
 
    LPred(i) = predLabel;
    
end


end

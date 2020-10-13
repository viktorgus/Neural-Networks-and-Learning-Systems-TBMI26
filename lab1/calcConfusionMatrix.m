function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTrue);
NClasses = length(classes);

% Add your own code here
cM = zeros(NClasses);

for i = 1:length(LPred)
    predictionIndex = find(classes == LPred(i));
    labelIndex = find(classes == LTrue(i));
    
    
    cM(predictionIndex,labelIndex ) = cM(predictionIndex,labelIndex )+1;
end


end


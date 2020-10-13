function [Wout, ErrTrain, ErrTest] = trainSingleLayer(XTrain,DTrain,XTest,DTest,W0,numIterations,learningRate)
% TRAINSINGLELAYER Trains the single-layer network (Learning)
%    Inputs:
%                X* - Training/test samples (matrix)
%                D* - Training/test desired output of net (matrix)
%                W0 - Initial weights of the neurons (matrix)
%                numIterations - Number of learning steps (scalar)
%                learningRate  - The learning rate (scalar)
%    Output:
%                Wout - Weights after training (matrix)
%                ErrTrain - The training error for each iteration (vector)
%                ErrTest  - The test error for each iteration (vector)

% Initialize variables
ErrTrain = nan(numIterations+1, 1);
ErrTest  = nan(numIterations+1, 1);
NTrain = size(XTrain, 1);
NTest  = size(XTest , 1);
Wout = W0;

% Calculate initial error
YTrain = runSingleLayer(XTrain, Wout);
YTest  = runSingleLayer(XTest , Wout);
ErrTrain(1) = sum(sum((YTrain - DTrain).^2)) / NTrain;
ErrTest(1)  = sum(sum((YTest  - DTest ).^2)) / NTest;



for n = 1:numIterations
    % Add your own code here
%     dEdW = zeros(size(W0,1),size(W0,2));
%     
%     for i = 1:size(XTrain,1) 
%         for q = 1:size(W0,1)
%             dEdW(q,:) = 2*(YTrain(i,q)-DTrain(i,q))*XTrain(i,:)+dEdW(q,:);
%         end
%         %dEdW(1,:) = 2*(YTrain(i,1)-DTrain(i,1))*XTrain(i,:)+dEdW(1,:);
%         %dEdW(2,:) = 2*(YTrain(i,2)-DTrain(i,2))*XTrain(i,:)+dEdW(2,:);
%         
%         %dEdA = diag(2*(YTrain(i,:)-DTrain(i,:)));
%         %dAdW = repmat(XTrain(i,:),size(W0,1),1);
%         %dEdW = dEdA*dAdW+dEdW;
%     end
%     dEdW = dEdW/size(XTrain,1);
    
    %Updating Y with the new weights
    YTrain = XTrain * Wout; 
    %Calculating cost gradient
    grad_cost_w = 2 * (YTrain - DTrain)' * XTrain;
    
    
    % Take a learning step
    Wout = Wout - learningRate * grad_cost_w'/NTrain;
    
    % Evaluate errors
    YTrain = runSingleLayer(XTrain, Wout);
    YTest  = runSingleLayer(XTest , Wout);
    ErrTrain(n+1) = sum(sum((YTrain - DTrain).^2)) / NTrain;
    ErrTest(n+1)  = sum(sum((YTest  - DTest ).^2)) / NTest;
end
end


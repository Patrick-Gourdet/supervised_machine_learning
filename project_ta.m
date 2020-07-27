% COT 4501 project - deliverable two
% supervised learning - multiple classes
% data from UCI repository: 
clear all;
% parameters
% ... define conditioning parameter
lambda = [0.01 .001 0.0005];
% ... data inputs
N = 5;
trainPercent = 0.5;
vLength = 10;
nCategories = 3;
% read data
% ... for ta
filename = 'taEvalData.txt';
formatSpec = '%d %d %d %d %d %s\n';
delimiter = ',';
dataFileID = fopen(filename,'r');
data = textscan(dataFileID, formatSpec, 'Delimiter', delimiter);
fclose(dataFileID);
% build indices for shuffling data rows
perm = randperm(length(data{1}));
% build all data
% ... for seeds
x = cell2mat(data(:,1:N))';
dataSize = length(x);
x = [ones(1,dataSize);x];
% build classification list
% ... for seeds
classes = unique(data{1,6});
for i = 1:dataSize
    y(:,i) = ismember(classes,data{1,6}(i));
end
% shuffle data
x2 = []; y2 = [];
for i = 1:dataSize
    x2(:,i) = x(:,perm(i));
    y2(:,i) = y(:,perm(i));
end
% seperate training and validation data
nTraining = ceil(trainPercent*dataSize);
xt = x2(:,1:nTraining); yt = y2(:,1:nTraining);
% set aside three sets of validation data
xv1 = x2(:,nTraining+1:nTraining+vLength); yv1 = y2(:,nTraining+1:nTraining+vLength);
xv2 = x2(:,nTraining+vLength+1:nTraining+2*vLength); yv2 = y2(:,nTraining+vLength+1:nTraining+2*vLength);
xv3 = x2(:,nTraining+2*vLength+1:nTraining+3*vLength); yv3 = y2(:,nTraining+2*vLength+1:nTraining+3*vLength);
xv = {xv1 xv2 xv3}; yv = {yv1 yv2 yv3};
for j = 1:length(lambda)
    % solve for weights
    W = inv(xt*xt'+lambda(j))*(xt*yt');
    % results for training data
    results = {};
    fprintf('training...\n');
    for i = 1:nTraining
        [val, idx] = max(W'*xt(:,i));
        results{i} = classes(idx);
        [valC, idxC] = max(yt(:,i));
%         if strcmp(results{i}, classes(idxC))
%             fprintf('Hit: %s, %s\n',char(results{i}), char(classes(idxC)));
%         else
%             fprintf('Miss: %s, %s\n',char(results{i}), char(classes(idxC)));
%         end
    end
    % results for validation data
    % ... also count misses in categorization
    validationResults = {};
    misses = 0; hits = 0;
    fprintf('validating...\n');
    for i = 1:vLength
        [val, idx] = max(W'*xv{j}(:,i));
        validationResults{i} = classes(idx);
        [valC, idxC] = max(yv{j}(:,i));
        if strcmp(validationResults{i}, classes(idxC))
            hits = hits + 1;
            % fprintf('Hit: %s, %s\n',char(validationResults{i}), char(classes(idxC)));
        else
            misses = misses + 1;
            % fprintf('Miss: %s, %s\n',char(validationResults{i}), char(classes(idxC)));
        end
    end
    fprintf('Training %%\tlambda\t\thits\tmisclassifications\n');
    fprintf('%d\t\t%f\t%d\t%d\n ', trainPercent * 100, lambda(j), hits, misses);
end
% testing
tLambda = 0.0005;
xtest = x2(:,nTraining+3*vLength+1:dataSize); 
ytest = y2(:,nTraining+3*vLength+1:dataSize);
W = inv(xt*xt'+tLambda)*(xt*yt');
% results for test data
tresults = {};
fprintf('testing... with lambda = %f\n', tLambda);
confusionMatrix = zeros(nCategories,nCategories);
misses = 0; hits = 0;
for i = 1:length(xtest)
    [val, idx] = max(W'*xtest(:,i));
    results{i} = classes(idx);
    [valC, idxC] = max(ytest(:,i));
    if strcmp(results{i}, classes(idxC))          
        %fprintf('Hit: %s, %s\n',char(results{i}), char(classes(idxC)));
        hits = hits + 1;
    else
        %fprintf('Miss: %s, %s\n',char(results{i}), char(classes(idxC)));
        confusionMatrix(idx,idxC) = confusionMatrix(idx,idxC) + 1;
        misses = misses + 1;
    end
end
fprintf('Testing %%\tlambda\t\thits\tmisclassifications\n');
fprintf('%d\t\t%f\t%d\t%d\n ', trainPercent * 100, tLambda, hits, misses);
disp(classes');
disp(confusionMatrix);

disp('WINE DATA SET PORTION - THIRD PART')
% COT 4501 Team project - NR 3
% UCI repository: Wine
clear all;

A = csvread('wine.csv');

[m,n]=size(A);

y = A(:,1); 
x = A(:,2:n);
num = ceil(0.5*m);

perm = randperm(length(x));
x = x(perm,:);
y = y(perm,:);

xt = x(1:num,: );
yt = y(1:num,:);

t = templateSVM('Standardize',1);
%Standard onevone
Mdl = fitcecoc(xt,yt,'Learners',t,'FitPosterior',1,...
    'ClassNames',{'1','2','3'});
% One v All coding
Mdl2 = fitcecoc(xt,yt,'Coding','onevsall','Learners',t,'FitPosterior',1,...
    'ClassNames',{'1','2','3'});
disp('OUTCOME One vs One')
CVMdl = crossval(Mdl);
oosLoss = kfoldLoss(CVMdl)
disp('OUTCOME One vs All')
CVMdl2 = crossval(Mdl2)
oosLoss2 = kfoldLoss(CVMdl2)

[label,~,~,Posterior] = resubPredict(Mdl);
[label2,~,~,Posterior2] = resubPredict(Mdl2);
Mdl.BinaryLoss;
idx = randsample(size(xt,1),10,1);
disp('OUTCOME One vs One')
Mdl.ClassNames
table(yt(idx),label(idx),Posterior(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})
disp('OUTCOME One vs All')
Mdl2.ClassNames
table(yt(idx),label2(idx),Posterior2(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})

disp('IRIS DATA SET PORTION - THIRD PART')
% COT 4501 Team project - NR 3
% UCI repository: Iris
clear all;
%Use pre-saved Iris dataset 
A = load('iris.mat');

[m,n]=size(A.iris);

y = A.iris(2:end,5); 
x = A.iris(2:end,1:n-1);
num = ceil(0.1*m);
y = table2array(y);
perm = randperm(height(x));
x = x(perm,:);
y = y(perm,:);

xt = x(1:num,: );
yt = y(1:num,:);

t = templateSVM('Standardize',1);
Mdl = fitcecoc(xt,yt,'Learners',t,'FitPosterior',1,...
    'ClassNames',{'setosa','versicolor','virginica'});
Mdl2 = fitcecoc(xt,yt,'Coding','onevsall','Learners',t,'FitPosterior',1,...
    'ClassNames',{'setosa','versicolor','virginica'});
disp('OUTCOME One vs One')
CVMdl = crossval(Mdl)
oosLoss = kfoldLoss(CVMdl)
disp('OUTCOME One vs All')
CVMdl2 = crossval(Mdl2)
oosLoss2 = kfoldLoss(CVMdl2)

[label,~,~,Posterior] = resubPredict(Mdl);
[label2,~,~,Posterior2] = resubPredict(Mdl2);
Mdl.BinaryLoss
idx = randsample(size(xt,1),10,1);
disp('OUTCOME One vs One')
Mdl.ClassNames
table(yt(idx),label(idx),Posterior(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})
disp('OUTCOME One vs All')
Mdl2.ClassNames
table(yt(idx),label2(idx),Posterior2(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})
disp('TA DATA SET PORTION - THIRD PART')
% COT 4501 Team project - NR 3
% UCI repository: TA
clear all;
A = csvread('ta.csv');

[m,n]=size(A);

y = A(:,6); 
x = A(:,1:n-1);
num = ceil(0.1*m);

perm = randperm(length(x));
x = x(perm,:);
y = y(perm,:);

xt = x(1:num,: );
yt = y(1:num,:);

t = templateSVM('Standardize',1);
Mdl = fitcecoc(xt,yt,'Learners',t,'FitPosterior',1,...
    'ClassNames',{'1','2','3'});
Mdl2 = fitcecoc(xt,yt,'Coding','onevsall','Learners',t,'FitPosterior',1,...
    'ClassNames',{'1','2','3'});
disp('OUTCOME One vs One')
CVMdl = crossval(Mdl)
oosLoss = kfoldLoss(CVMdl)
disp('OUTCOME One vs All')
CVMdl2 = crossval(Mdl2)
oosLoss2 = kfoldLoss(CVMdl2)

[label,~,~,Posterior] = resubPredict(Mdl);
[label2,~,~,Posterior2] = resubPredict(Mdl2);
Mdl.BinaryLoss
idx = randsample(size(xt,1),10,1);
disp('OUTCOME One vs One')
Mdl.ClassNames
table(yt(idx),label(idx),Posterior(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})
disp('OUTCOME One vs All')
Mdl2.ClassNames
table(yt(idx),label2(idx),Posterior2(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})
% COT 4501 Team project - NR 3
% UCI repository: SEED Data
A = importdata('seeds_dataset.csv');

[m,n]=size(A);

y = A(:,8); 
x = A(:,1:n-1);
num = ceil(0.1*m);

perm = randperm(length(x));
x = x(perm,:);
y = y(perm,:);

xt = x(1:num,: );
yt = y(1:num,:);

t = templateSVM('Standardize',1);
Mdl = fitcecoc(xt,yt,'Learners',t,'FitPosterior',1,...
    'ClassNames',{'1','2','3'});
Mdl2 = fitcecoc(xt,yt,'Learners',t,'FitPosterior',1,...
    'ClassNames',{'1','2','3'});
disp('OUTCOME One vs One')
CVMdl = crossval(Mdl)
oosLoss = kfoldLoss(CVMdl)
disp('OUTCOME One vs All')
CVMdl2 = crossval(Mdl2)
oosLoss2 = kfoldLoss(CVMdl2)

[label,~,~,Posterior] = resubPredict(Mdl);
[label2,~,~,Posterior2] = resubPredict(Mdl2);
Mdl.BinaryLoss
idx = randsample(size(xt,1),10,1);
disp('OUTCOME One vs All')
Mdl.ClassNames
table(yt(idx),label(idx),Posterior(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})
disp('OUTCOME One vs All')
Mdl2.ClassNames
table(yt(idx),label2(idx),Posterior2(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})

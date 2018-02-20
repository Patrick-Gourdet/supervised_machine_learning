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
x = x(perm,:)
y = y(perm,:)

xt = x(1:num,: );
yt = y(1:num,:);

t = templateSVM('Standardize',1)
Mdl = fitcecoc(x,y,'Learners',t,'FitPosterior',1,...
    'ClassNames',{'setosa','versicolor','virginica'},'Verbose',2);

CVMdl = crossval(Mdl)
oosLoss = kfoldLoss(CVMdl)

[label,~,~,Posterior] = resubPredict(Mdl,'Verbose',1);
Mdl.BinaryLoss
idx = randsample(size(x,1),10,1);
Mdl.ClassNames
table(y(idx),label(idx),Posterior(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})

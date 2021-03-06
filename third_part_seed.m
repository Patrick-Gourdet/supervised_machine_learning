A = importdata('seeds_dataset.csv');

[m,n]=size(A);

y = A(:,8); 
x = A(:,1:n-1);
num = ceil(0.1*m);

perm = randperm(length(x));
x = x(perm,:)
y = y(perm,:)

xt = x(1:num,: );
yt = y(1:num,:);

t = templateSVM('Standardize',1)
Mdl = fitcecoc(x,y,'Learners',t,'FitPosterior',1,...
    'ClassNames',{'1','2','3'},'Verbose',2);

CVMdl = crossval(Mdl)
oosLoss = kfoldLoss(CVMdl)

[label,~,~,Posterior] = resubPredict(Mdl,'Verbose',1);
Mdl.BinaryLoss
idx = randsample(size(x,1),10,1);
Mdl.ClassNames
table(y(idx),label(idx),Posterior(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})

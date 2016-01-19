load W1L1;
load b1L1;
load W1L2;
load b1L2;
load dataAftRemv;

x=zeros(1440,638);
m=size(x,2);
for i=1:638
    x(:,i)=reshape(recordingAftRemv((i-1)*72+1:i*72,:),[1440,1]);
end

z2=W1L1*x+repmat(b1L1,1,m);
%a2=sigmoid(z2);
a2=1./(1+exp(-1*z2));

z3=W1L2*a2+repmat(b1L2,1,m);
%a3=sigmoid(z3);
a3=1./(1+exp(-1*z3));
data_svm=a3';
save('data_svm.mat','data_svm');

%function sigm = sigmoid(x)
%    sigm = 1 ./ (1 + exp(-x));
%end
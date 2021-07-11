function [pingce,mm]=cross_performance(final_adj,gtrue_g,gtrue_q)
%%% final_adj similarity matrix, size is  gallery number X query number
%%% if query is 100, gallery is 200, the simialrity matrix is 200x100
%%%gtrue_g  gallery label, size is 1 x gallery sample number)
%%%gtrue_q  query sample label, size is 1 x query sample number)
[~,index]=sort(final_adj,'ascend');
rresult=zeros(length(gtrue_q),length(gtrue_g));
for i=1:length(gtrue_q)
    lable=gtrue_q(1,i);
    same_class=find(gtrue_g(:)==lable);
    
    for j=1:length(same_class)
        rresult(i,index(:,i)==same_class(j))=1;
    end
    
end

a=rresult;
for i=1:size(a,2)
    n = a(:,1:i);
    s = sum(sum(n));
    ETH_p(i) = s/(size(a,1)*i);
    ETH_rr(i) = s/sum(sum(a));
end

AUC_0 = trapz(ETH_rr,ETH_p);

mm=[ETH_rr' ETH_p' ];
groundtruth=gtrue_g;
groundtruth_q=gtrue_q;
max_m=max(max(groundtruth));
number_m=[];
for i=1:max_m
    number_m(1,i)=size(find(groundtruth==i),2);
end

max_q=max(max(groundtruth_q));
number_q=[];
for i=1:max_q
    number_q(1,i)=size(find(groundtruth_q==i),2);
end
%% 其他几种评测方法

%% NN
NN=sum(rresult(:,1))/sum(number_q);

%% FT ST

for i=1:size(rresult,1)
    ft(1,i)=sum(rresult(i,1:number_m(1,groundtruth_q(1,i)))/number_m(1,groundtruth_q(1,i)));
    st(1,i)=sum(rresult(i,1:number_m(1,groundtruth_q(1,i))*2-1)/number_m(1,groundtruth_q(1,i)));
end
FT=sum(sum(ft))/size(rresult,1);
ST=sum(sum(st))/size(rresult,1);

%% 20-Measure
temp123=sum(sum(rresult));
n = rresult(:,1:20);
s = sum(sum(n));
p = s/(size(rresult,1)*20);
rr = s/(temp123);
F_measure=2/(1/p+1/rr);

%% DCG
count=0;
for i=1:size(rresult,1)
    count=count+1;
    DCG_k=rresult(count,1);
    DCG_data=1;
    if number_m(1,groundtruth_q(1,i))>1
        for k=2:number_m(1,groundtruth_q(1,i))
            DCG_k=DCG_k+rresult(count,k)/log2(k);
            DCG_data=DCG_data+1/log2(k);
        end
    end
    DCG_child(count)=DCG_k/DCG_data;
end
DCG=sum(DCG_child)/size(rresult,1);

%% ANMRR
T_max=max(number_m);
count=0;
for i=1:size(rresult,1)
    count=count+1;
    S_k=min(4*number_m(1,groundtruth_q(1,i)),2*T_max);
    r=[];
    for k=1:number_m(1,groundtruth_q(1,i))
        if rresult(count,k)==1
            r(k)=k;
        else
            r(k)=S_k+1;
        end
    end
    NMRR(count)=( (sum(r)/number_m(1,groundtruth_q(1,i))) - number_m(1,groundtruth_q(1,i))/2 -0.5 ) / ( S_k - number_m(1,groundtruth_q(1,i))/2 + 0.5 );
end
ANMRR=sum(NMRR)/count;

pingce=[NN FT ST F_measure DCG ANMRR AUC_0];

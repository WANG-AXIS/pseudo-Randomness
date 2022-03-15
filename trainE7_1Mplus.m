load('ESource.mat')

N=7; %The length of series
M=40000;%The number of samples
%Series_Clock=zeros(M,6);      %Record the time seed generating the random number
Series_training=zeros(M,N);
%Series_training(1,:)=Source(1,1:101);

for iter=1:M
 Series_training(iter,:)=ESource(1,iter:iter+N-1);
 %Series_training(iter,:)=Gen_Rand_01(N);
end

Series_training=Series_training';
P1=Series_training(1:N-1,:);  %训练集，预测集合
T1=Series_training(N,:);

% P1(1:(N+1)/2-1,:)=Series_training(1:(N+1)/2-1,:);  %训练集，预测集合
% P1((N+1)/2:N-1,:)=Series_training((N+1)/2+1:N,:);  %训练集，预测集合
% T1=Series_training((N+1)/2,:);

tra_data=P1;
tra_label=T1;
Stat_low1=0;
Stat_low2=0;
Stat_up1=0.6;
Stat_up2=0.6;
while (Stat_low1<0.5)&&(Stat_low2<0.5)

net=newff(tra_data,tra_label,[30,20],{'logsig','logsig','purelin'});
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false;    
   net.trainParam.epochs=40;
   net.trainParam.mc=0.95;                    % 附加动量因子
   net.trainParam.lr=0.05;                       % 学习速率，这里设置为0.05
   net.trainParam.min_grad=1e-4;        % 最小性能梯度
   net.trainParam.max_fail=200;                 % 最大确认失败次数

   net=train(net,tra_data,tra_label);
%view(net);
Result=hardlim(sim(net,tra_data)-0.5);
Pfm=length(find(Result==T1));
Pfm=Pfm/M;

%%
Index=sum(ESource(1,100000:1000000))/900000;
M_pred=100000;%The number of samples
%Series_Clock_pred=zeros(M_pred,6);      %Record the time seed generating the random number
Series_pred=zeros(M_pred,N);
Stat_overall=zeros(1,9);
for j=1:9
  for iter=1:M_pred
      
     Series_pred(iter,:)=ESource(1,iter+100000*j+1000000:iter+100000*j+1000000+N-1);
  end
  Series_pred1=Series_pred';
  P2=Series_pred1(1:N-1,:);  %训练集，预测集合
% figure()
% imshow(P2(:,100:1000),[])

T2=Series_pred1(N,:);

% P2=zeros(N-1,M_pred);
% P2(1:(N+1)/2-1,:)=Series_pred1(1:(N+1)/2-1,:);
% P2((N+1)/2:N-1,:)=Series_pred1((N+1)/2+1:N,:); 
% T2=zeros(1,M_pred);
% T2=Series_pred1((N+1)/2,:);

Result_pre=hardlim(sim(net,P2)-0.5);
Pfm_pre=length(find(Result_pre==T2));
Pfm_pre=Pfm_pre/M_pred;
Stat_overall(j)=Pfm_pre;
end

%%

Stat_mean=mean(Stat_overall);
Stat_var=var(Stat_overall);
Stat_Norvar=sqrt(Stat_var)/sqrt(length(Stat_overall));
Stat_low1=Stat_mean-4.502*Stat_Norvar
Stat_up1=Stat_mean+4.502*Stat_Norvar
Stat_low2=Stat_mean-2.897*Stat_Norvar
Stat_up2=Stat_mean+2.897*Stat_Norvar
end


%%
 inputWeights=net.IW{1,1};
   inputbias=net.b{1};
   layerWeights=net.LW{2,1};
   layerbias=net.b{2};
   outputWeights=net.LW{3,2};
   outputbias=net.b{3};
   
   
   net=newff(tra_data,tra_label,[30,20],{'logsig','logsig','purelin'});
    
 net.IW{1,1}=inputWeights;
  net.b{1}= inputbias;
   net.LW{2,1}=layerWeights;
   net.b{2}=layerbias;
   net.LW{3,2}=outputWeights;
   net.b{3}=outputbias;
   
   Result_pp=hardlim(sim(net,P2)-0.5);
Pfm_pp=length(find(Result_pre==T2));



%%
Stat_mean=mean(Stat_overall);
Stat_var=var(Stat_overall);
Stat_Norvar=sqrt(Stat_var)/sqrt(length(Stat_overall));
Stat_low=Stat_mean-5*Stat_Norvar
Stat_up=Stat_mean+5*Stat_Norvar;


save('E7_5sigma')


%%









    

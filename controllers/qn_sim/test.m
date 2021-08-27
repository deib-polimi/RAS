clear
tf=1000;
rep=5000;
dt=10^-1;
T=linspace(0,(tf+1)*dt,(tf+1));
X=mean(qn_sim([0,0,0],[0,10,0],[100,10,0],[0,1,0;0,0,1;0,0,0;],tf*dt,rep,dt),3);
[t,y] = ode45(@(t,y) odefcn(t,y,[0,10],[10]),T,[0]);

figure 
hold on
plot(T,y)
plot(T,X(2,:))

% Tr=X(3,:)./T;
% plot(Tr)
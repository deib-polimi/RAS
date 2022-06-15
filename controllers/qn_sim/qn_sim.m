function X=qn_sim(X0,S,MU,P,TF,rep,dt)
import Gillespie.*

%% Rate constants
p.MU=MU;      
p.S=max(S,zeros(1,size(S,2)));
p.P=P;

% disp(MU)
% disp(S)
% disp(P)

[stoich_matrix,propensities_2state]=genStoich(P);

%% Initial state
tspan = [0, TF]; %seconds

%% Specify reaction network
%pfun = @propensities_2state;
%disp(propensities_2state)
pfun = eval(propensities_2state);

p.stoich_matrix=stoich_matrix;

%% Run simulation
X=zeros(length(X0),round(TF/dt,1)+1,rep);
for i=1:rep
    [t,x] = directMethod(stoich_matrix, pfun, tspan, X0, p);
    tsin = timeseries(x,t);
    tsout = resample(tsin,linspace(0,TF,round(TF/dt,1)+1),'zoh');
    X(:,:,i)=tsout.Data';
end
%[t,x] = firstReactionMethod(stoich_matrix, pfun, tspan, x0, p);
end

function dydt = odefcn(t,y,MU,S)
dydt = zeros(2,1);
dydt(1) = MU(1)-MU(2)*min(y(1),S(1));
dydt(2) = MU(2)*min(y(1),S(1));
end
function [stoich_matrix,a]=genStoich(P)
stoich_matrix=zeros((size(P,1))*(size(P,1)),size(P,1));
k=0;
a=sprintf('@(x, p) [');
for i=1:size(P,1)
    for j=1:size(P,2)
        k=k+1;
        if(j~=i)
            stoich_matrix(k,i)=-1;
            stoich_matrix(k,j)=1;
        end
        if(i~=1)
            a=sprintf('%s\tp.P(%d,%d)*p.MU(%d)*min(x(%d),p.S(%d));',a,i,j,i,i,i);
        else
            %a=sprintf('%s\tp.P(%d,%d)*p.MU(%d)*x(%d);',a,i,j,i,i);
            a=sprintf('%s\tp.P(%d,%d)*p.MU(%d);',a,i,j,i);
        end
    end
end
a=sprintf('%s];',a);
end


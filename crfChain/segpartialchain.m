function subchain = segpartialchain(y_s)

nNodes = length(y_s);
labelflag = 0;
subchain = zeros(0,3);
scnt = 0;
for n = 1:nNodes
    if (labelflag == 0)
        if (y_s(n) == -1)                                    
            scnt = scnt + 1;
            subchain(scnt,1) = -1;
            subchain(scnt,2) = n;
            labelflag = -1;
        else
            scnt = scnt + 1;
            % beginning of the labeled subchain
            subchain(scnt,1) = 1;
            subchain(scnt,2) = n;
            labelflag = 1;
         end
    else
        if ((y_s(n) == -1) && (labelflag == 1)) ||((y_s(n) ~=-1) && (labelflag == -1))
            subchain(scnt,3) = n-1;
            labelflag = -labelflag;
            scnt = scnt + 1;
            subchain(scnt,1) = labelflag;
            subchain(scnt,2) = n;       
        end
    end 
end
subchain(scnt,3) = nNodes;

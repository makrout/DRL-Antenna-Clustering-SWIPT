clc; clear all; close all;
    
Iterations=10;
M=1; N=1; 

Alpha=0.5;    
PdB=  20:5:60; 
P_D1=10.^(PdB/10)*10^(-3);
     
for Count_P=1:length(P_D1)
 
    Count_Iterations_Op=0;
    Count_Iterations_NONOp=0;
 
    for Count_I=1:Iterations
        %% Channel Generations
        H_h=raylrnd(1,N,M); H_I=H_h'; 
       
        Rician=1;
        H_1= ricernd(1, Rician,M ,M);H_2= ricernd(1, Rician,N ,N);
 
        
        
        Sigma_1=eye(M); Sigma_2=eye(N);
 
        
        %% Transmission Power at Device # 2
        if Count_I==1
            P_D2(Count_I)=P_D1(Count_P);
        else 
            x=2^(log_det(eye(M)+ H_I*Q_1*H_I' )       / (log(2)));
            P_D2(Count_I)=  min(x,P_D1(Count_P));
        end
        %=========================CVX Begin========================
        cvx_begin
        variables Q_1(M,M) Q_2(N,N)
        
        maximize(   Alpha*log_det(eye(N)+ H_I*Q_2*H_I')        / (log(2))...
                +(1-Alpha)*log_det(eye(M)+ H_h*Q_1*H_h'+eye(M)) / (log(2)) )
        subject to
        Q_1(:)>=0
        Q_2(:)>=0
                          
        trace(Q_1)<=P_D1(Count_P)
        trace(Q_2)<=P_D2(Count_I)
        cvx_end
        %=========================CVX End========================
        
        %% Optimal Rate
        R_i_OP(Count_I)= Alpha*log_det(eye(M)+ H_I*Q_2*H_I')        / (log(2));
        %% Non-Optimized Rate
        Q_NO_Preco=(P_D2(Count_P)/N)*eye(N);
         
        R_i_NO(Count_I)=  Alpha*log_det(eye(M)+ H_I*Q_NO_Preco*H_I')        / (log(2)) ;
        if abs(R_i_OP(Count_I)) ~= inf && ~isnan(abs(R_i_OP(Count_I)))
        Count_Iterations_Op=Count_Iterations_Op+1;
        elseif abs(R_i_OP(Count_I)) == inf || abs(R_i_OP(Count_I))<0 || isnan(abs(R_i_OP(Count_I)))
            R_i_OP(Count_I)=0;
        end
        
        if abs(R_i_NO(Count_I)) ~= inf && ~isnan(abs(R_i_NO(Count_I)))
        Count_Iterations_NONOp=Count_Iterations_NONOp+1;
        elseif abs(R_i_NO(Count_I)) == inf || abs(R_i_NO(Count_I))<0 || isnan(abs(R_i_NO(Count_I)))
            R_i_NO(Count_I)=0;
        end
    end
    
        R_OP(Count_P)=sum(R_i_OP)/(2*Count_Iterations_Op);
        R_NO(Count_P)=sum(R_i_NO)/(2*Count_Iterations_NONOp);
 
end

figure
semilogy(PdB,R_OP,'b-o');
hold on
semilogy(PdB,R_NO,'r-*');
grid
xlabel('SNR (dB)');
title('M=4');
ylabel('Rate (bps/Hz)');
legend('Optimal Precoding (Using CVX)','No Precoding');
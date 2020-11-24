clc; clear all; close all;
    
Iterations=1;
M=4;M_h=1;%    1<M_h<M-1
N=4;N_h=1;%    1<N_h<N-1
M_I=M-M_h;N_I=N-N_h;
Alpha=0.5;
    
PdB=  20:5:60;%:10:90;%10:5:20;
P_D1    = 10.^(PdB/10)*10^(-3);
Epslion = 0.1*P_D1;
     
for Count_P=1:length(P_D1)
 
    Count_Iterations_Op=0;
    Count_Iterations_NONOp=0;
 
 
    for Count_I=1:Iterations
        %% Channel Generations
        tic
        H = raylrnd(1,N,M);
        %% Antenna Splitting
        [H_I, H_h]=Antenna_Splitting(H,M_h,N_h);
        %% For Random Splitting without antenna selection, the line above should be 
        % commented and the next four lines should be uncommented
        %H_I=H;
        %H_I(1:M_h,:)=[];
        %H_I(:,1:N_h)=[];
        %H_h=H(1:M_h,1:N_h);

        Rician=1;
        H_1= ricernd(1, Rician,M_I,M_h);H_2= ricernd(1, Rician,N_h,N_I);

        Q_o=(P_D1(Count_P)/M_h)*eye(M_h);

        Sigma_1=eye(M_I); Sigma_2=eye(N_h);
        Term_1=eye(M_I)/(Sigma_1+H_1*Q_o*H_1');
        
        %% Transmission Power at Device # 2
        if Count_I==1
            P_D2(Count_I)=P_D1(Count_P) - Epslion(Count_P);
        else
            x=log_det(Sigma_2+( H_h*Q_1*H_h'+H_2*Q_2*H_2'+Sigma_2) );
            P_D2(Count_I)= min(2^(x)-Epslion(Count_P),P_D1(Count_P)-Epslion(Count_P));
        end
        %=========================CVX Begin========================
        cvx_begin
        variables Q_1(M_h,M_h) Q_2(N_I,N_I)
        
        maximize(  Alpha*...
            ( log_det(Sigma_1+H_1*Q_1*H_1'+H_I*Q_2*H_I')/(log(2)) ...
            + trace( Term_1*H_1*(Q_o-Q_1)*H_1' )/(log(2)) ...
            - log_det( Sigma_1+H_1*Q_o*H_1')/(log(2))  )...
            + (1-Alpha)*...
            log_det(Sigma_2+( H_h*Q_1*H_h'+H_2*Q_2*H_2'+Sigma_2) ) )
        subject to
        Q_1(:)>=0
        Q_2(:)>=0
    
        trace(Q_1)<=P_D1(Count_P)
        trace(Q_2)<=P_D2(Count_I)
        cvx_end
        %=========================CVX End========================
         toc 
        %% Optimal Rate
        R_i_OP(Count_I)= Alpha*...
            ( log_det(Sigma_1+H_1*Q_1*H_1'+H_I*Q_2*H_I')/(log(2)) ...
            + trace( Term_1*H_1*(Q_o-Q_1)*H_1' )/(log(2)) ...
            - log_det( Sigma_1+H_1*Q_o*H_1')/(log(2))  );
        %% Non-Optimized Rate
        Q_NO_Preco_1=(P_D1(Count_P)/M_h)*eye(M_h);
        Q_NO_Preco_2=(P_D2(Count_I)/N_I)*eye(N_I);
        R_i_NO(Count_I)=  Alpha*...
            ( log_det(Sigma_1+H_1*Q_NO_Preco_1*H_1'+H_I*Q_NO_Preco_2*H_I')/(log(2)) ...
            + trace( Term_1*H_1*(Q_o-Q_NO_Preco_1)*H_1' )/(log(2)) ...
            - log_det( Sigma_1+H_1*Q_o*H_1')/(log(2))  ) ;
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
  
 
        R_OP(Count_P)=sum(R_i_OP)/Count_Iterations_Op;
        R_NO(Count_P)=sum(R_i_NO)/Count_Iterations_NONOp;
 
end

figure
plot(PdB,R_OP,'b-o');
hold on
plot(PdB,R_NO,'r-*');
grid
xlabel('SNR (dB)');
title('M=4');
ylabel('Rate (bps/Hz)');
legend('Optimal Precoding (Using CVX)','No Precoding');

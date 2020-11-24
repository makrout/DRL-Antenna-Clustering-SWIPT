function [H_I H_h]=Antenna_Splitting(H,M_h,N_h)

M=length(H(1,:));N=length(H(:,1));


m_h=[];n_h=[];
H_Update=H;
if M_h>=N_h
    
    while length(m_h)<M_h
        
        %% Selecting m_h
        norm_m_h=sum(H_Update.^2);
        [~, Index_m_h]=min(norm_m_h);
        m_h=[m_h Index_m_h];
        H_Update(:,Index_m_h)=[];
        
        while length(n_h)<N_h
            
            %% Selecting n_h
            norm_n_h=sum(H_Update.^2');
            [~, Index_n_h]=min(norm_n_h);
            n_h=[n_h Index_n_h];
            H_Update(Index_n_h,:)=[];
            
        end
        
    end
    
else
    
    while length(n_h)<N_h
        
        
        %% Selecting n_h
        norm_n_h=sum(H_Update.^2');
        [~, Index_n_h]=min(norm_n_h);
        n_h=[n_h Index_n_h];
        H_Update(Index_n_h,:)=[];
        
        while length(m_h)<M_h
            
            %% Selecting m_h
            norm_m_h=sum(H_Update.^2);
            [~, Index_m_h]=min(norm_m_h);
            m_h=[m_h Index_m_h];
            H_Update(:,Index_m_h)=[];
            
        end
        
    end
    
    
end

H_I=H_Update';
H_h=H(n_h,m_h);

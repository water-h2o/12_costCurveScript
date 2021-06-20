filesHere = dir('*.xlsx');
in_addresses = {filesHere.name};

for i = 1:2:length(in_addresses)
    
    in_address_FP = in_addresses{i};
    in_address_TP = in_addresses{i+1};
    
    disp(in_address_FP)
    disp(in_address_TP)
    
    fname_curr_FP = in_address_FP;
    fname_curr_TP = in_address_TP;
    
    %opts_b2b_in = detectImportOptions(fname_curr);
    curr_M_in_FP = readmatrix(fname_curr_FP);
    curr_M_in_FP_sz = size(curr_M_in_FP);
    
    curr_M_in_TP = readmatrix(fname_curr_TP);
    curr_M_in_TP_sz = size(curr_M_in_TP);
    
    %n_sample = curr_M_in(2, 1:end);
    fp = fliplr(curr_M_in_FP(14,1:end));
    tp = fliplr(curr_M_in_TP(14,1:end));
    %err_bars = curr_M_in(15,1:end); 
    
    E_Co_0 = 0;
    PC_P_0 = 0;
    
    E_Co_ = zeros(1,(length(fp)-1));
    PC_P_ = zeros(1,(length(fp)-1));
    
    for j = 1:length(E_Co_)
       
        A_ = fp(j) - fp(j+1);
        B_ = tp(j) - tp(j+1);
        
        PC_P_(j) = A_/(A_+B_);
        E_Co_(j) = fp(j) + (1 - tp(j) - fp(j)) * PC_P_(j);
    end
    
    E_Co_end = 0; 
    PC_P_end = 1;
    
    E_Co = [E_Co_0 E_Co_ E_Co_end];
    PC_P = [PC_P_0 PC_P_ PC_P_end];
    
    acutes = zeros(1,length(E_Co));
    
    for j = 2:(length(acutes)-1)
    
        P_L = [PC_P(j-1) E_Co(j-1)];
        P_C = [PC_P(j) E_Co(j)];
        P_R = [PC_P(j+1) E_Co(j+1)];
        
        if vecAngle(P_L, P_C, P_R) < 90
            
            acutes(j) = 1;
        end
        
    end 
    
    if sum(acutes) > 1
        
        offend = find(acutes);
        
        x_L_l = PC_P(offend(1) - 1);
        x_L_r = PC_P(offend(1));
        y_L_l = E_Co(offend(1) - 1);
        y_L_r = E_Co(offend(1));
        slp_L = (y_L_r - y_L_l)/(x_L_r - x_L_l);
        stt_L = y_L_l - slp_L * x_L_l;
        
        x_R_l = PC_P(offend(2));
        x_R_r = PC_P(offend(2) + 1);
        y_R_l = E_Co(offend(2));
        y_R_r = E_Co(offend(2) + 1);
        slp_R = (y_R_r - y_R_l)/(x_R_r - x_R_l);
        stt_R = y_R_l - slp_R * x_R_l;
        
        PC_P_corr = (stt_L - stt_R)/(slp_R - slp_L);
        E_Co_corr = stt_L + slp_L * PC_P_corr;
        
        
        PC_P = [0 PC_P(1:(offend(1)-1)) PC_P_corr PC_P((offend(2)+1):end)];
        E_Co = [0 E_Co(1:(offend(1)-1)) E_Co_corr E_Co((offend(2)+1):end)];
    end
    
    PC_P(isnan(PC_P)) = 0;
    E_Co(isnan(E_Co)) = 0;
    
    AUC = trapz(PC_P, E_Co);
    
    figure(1);
    clf;
    
    hold on
    grid on
    title(['Cost curve: ' in_address_FP(1:end-5)])
    xlabel('PC(P)')
    ylabel('E[Co]')

    fill(PC_P, E_Co,[0.8 0.8 1])
    
    plot([0 1], [0 1], 'r', 'LineWidth', 1)
    plot([0 1], [1 0], 'r', 'LineWidth', 1)
    for j = 1:10
        
        line([0 1], [fp(j) 1-tp(j)])
    end
    
    plot(PC_P,E_Co,'b', 'LineWidth',1.5)
    
    rectangle('Position', [0.36 0.85 0.28 0.1], 'EdgeColor', [0 0 0.7],...
              'LineWidth', 1.5, 'FaceColor', 'w')
    text(0.385,0.9, ['AUC = ' num2str(AUC)],'Color','b','FontSize',7)
    
    ylim([0 1])
    xlim([0 1])
    
    hold off
    
    set(gcf, 'PaperUnits', 'inches');
    x_width = 4;
    y_width = 3;
    set(gca,'fontsize', 7);
    set(gcf, 'PaperPosition', [0 0 x_width y_width]); %
    %set(gcf, 'PaperPositionMode', 'auto');
    saveas(gcf,  ['./imgs_cost/' in_address_FP(1:end-4) 'png']);
    disp(['saved figure for ' in_address_FP])
end

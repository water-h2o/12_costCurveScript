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
    
    AUC = trapz(fp,tp) - 0.5;
    
    figure(1);
    clf;
    
    hold on
    grid on
    title(['ROC plot: ' in_address_FP(1:end-5)])
    xlabel('fp')
    ylabel('tp')
    fill(cat(2,fp,0),cat(2,tp,0),[0.8 0.8 1])
    plot(fp,tp,'b','LineWidth', 1.75)
    scatter(fp,tp,150,'b.')
    fill([0 1 1], [0 1 0], [1 0.8 0.8])
    plot([0 1], [0 1],'r', 'LineWidth', 1)
    %errorbar(n_sample, fp_or_tp, 2*err_bars, 'LineWidth', 0.75, 'Color', '#4DBEEE');
    %errorbar(n_sample, fp_or_tp, err_bars, 'LineWidth', 1.5, 'Color', '#0072BD');
    %scatter(n_sample, fp_or_tp, 150, '.k')
    %yline(0.5,'-.r','LineWidth', 2);
    %yline(0,'-.g','LineWidth', 2);
    %yline(1,'-.g','LineWidth', 2);
    %set(gca, 'XScale', 'log')
    rectangle('Position', [0.025 0.85 0.28 0.1], 'EdgeColor', [0 0 0.7],...
              'LineWidth', 1.5, 'FaceColor', 'w')
    text(0.05,0.9, ['AUC = ' num2str(AUC)],'Color','b','FontSize',7)
    ylim([0 1])
    xlim([0 1])
    
    hold off
    
    set(gcf, 'PaperUnits', 'inches');
    x_width = 4;
    y_width = 3;
    set(gca,'fontsize', 7);
    set(gcf, 'PaperPosition', [0 0 x_width y_width]); %
    %set(gcf, 'PaperPositionMode', 'auto');
    saveas(gcf,  ['./imgs_ROC/' in_address_FP(1:end-4) 'png']);
    disp(['saved figure for ' in_address_FP])
end
